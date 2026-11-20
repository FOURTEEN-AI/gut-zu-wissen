import json
from datetime import UTC, datetime

import pytest

from gut_zu_wissen.dialog import Statement, Turn
from gut_zu_wissen.explicit import ExplicitCheck, Finding
from gut_zu_wissen.extraction import Fact
from gut_zu_wissen.github import Comment
from gut_zu_wissen.judge import TurnJudgement
from gut_zu_wissen.ranking import RankedFact
from gut_zu_wissen.review import (
    Draw,
    RateState,
    Review,
    apply_issue,
    apply_labels,
    build_review,
    closes_issue,
    draw_values,
    escape,
    explicit_accuracy,
    explicit_confusion,
    explicit_pairs,
    find_review,
    is_authorized,
    load_rate_state,
    load_review,
    next_rate,
    open_issue,
    pack,
    parse_comment,
    rate_due,
    render_issue,
    render_reply,
    review_population,
    review_status,
    reviewer_flag,
    sample,
    save_review,
    update_rate,
)
from gut_zu_wissen.run import load_config

NOW = datetime(2026, 11, 20, 8, 0, tzinfo=UTC)
REVIEW_CONFIG = {
    **load_config()["review"],
    "adjust_rate": True,
}
REVIEWER = "ajFourteen"
SMALL_ISSUES = {**REVIEW_CONFIG, "issue_max_chars": 2500}  # forces follow-up comments

TURNS = [
    Turn(0, "Lena", "Der Rotfuchs jagt vor allem Mäuse.", ["5-0000"],
         [Statement("Der Rotfuchs jagt vor allem Mäuse.", ["5-0000"])]),
    Turn(1, "Felix", "Echt? Erzähl mehr!", []),
    Turn(2, "Lena", "Er lebt in Europa, schreib an @someone <b>.", ["5-0001", "7-0000"],
         [Statement("Er lebt in Europa.", ["5-0001", "7-0000"])]),
    Turn(3, "Felix", "Und wo noch?", ["5-0002"], [Statement("Er lebt woanders.", ["5-0002"])]),
]  # fmt: skip
JUDGEMENTS = [
    TurnJudgement(0, True, "jev", {"aussage_0": 0.99, "vollstaendig": 0.97}),
    TurnJudgement(1, True, "jev", {"vollstaendig": 0.98}),
    TurnJudgement(2, True, "llm_judge", {"aussage_0": 0.6}, reason="Folgt aus 5-0001."),
    TurnJudgement(3, False, "jev", {"aussage_0": 0.01}, reason="Nicht belegt."),
]


def fact(segment_id: str, plain: str, context: list[str] | None = None) -> Fact:
    return Fact(segment_id, "sentence", "", [], plain, context or [])


FACTS = {
    "5-0000": RankedFact(1, "Rotfuchs", 5, fact("5-0000", "Der Rotfuchs frisst Mäuse.")),
    "5-0001": RankedFact(2, "Rotfuchs", 5, fact("5-0001", "Er lebt in Europa.", ["Der Fuchs."])),
    "7-0000": RankedFact(1, "Europa", 7, fact("7-0000", "Europa ist ein Kontinent.")),
}


def comment(cid: int, body: str, login: str = REVIEWER, at: str = "2026-11-19T11:00:00Z"):
    return Comment(cid, login, "MEMBER", body, at, at, f"https://x/issues/7#issuecomment-{cid}")


CLEAN = ExplicitCheck("claude-sonnet-5-5", False, "clean", [], "Nichts Drastisches.")
FLAGGED = ExplicitCheck(
    "claude-sonnet-5-5",
    True,
    "explicit",
    [Finding(2, "violence", "reißt <die> Beute", "Blutig geschildert.")],
    "Drastische Jagdszene.",
)


def review(rate: float = 1.0, explicit: ExplicitCheck = CLEAN) -> Review:
    return build_review("2026-11-19_rotfuchs", "Rotfuchs", TURNS, JUDGEMENTS, FACTS, rate, explicit)


# ---------------------------------------------------------------- sampling


def test_population_is_accepted_turns_with_fact_ids():
    assert review_population(TURNS, JUDGEMENTS) == [0, 2]


def test_draws_are_reproducible_from_the_seed():
    assert draw_values("2026-11-19_rotfuchs", 3) == draw_values("2026-11-19_rotfuchs", 3)
    assert draw_values("2026-11-19_rotfuchs", 3) != draw_values("2026-11-19_fuchs", 3)
    assert all(0 <= u < 1 for u in draw_values("x", 100))


def test_sample_draws_below_r():
    assert sample([0, 2, 4], [0.1, 0.5, 0.9], 0.5) == [
        Draw(0, 0.1, True),
        Draw(2, 0.5, False),
        Draw(4, 0.9, False),
    ]


def test_rate_one_draws_every_turn():
    assert all(d.sampled for d in sample(range(50), draw_values("s", 50), 1.0))


def test_build_review_records_draws_and_everything_the_reviewer_needs():
    r = review()
    assert r.population == [0, 2] and r.seed == "2026-11-19_rotfuchs" and r.rate == 1.0
    assert [d.u for d in r.draws] == draw_values("2026-11-19_rotfuchs", 2)
    assert r.status == "pending" and r.sampled() == [0, 2]
    turn = r.turns[1]
    assert turn.previous_text == "Echt? Erzähl mehr!"
    assert [(f.id, f.article, f.revid) for f in turn.facts] == [
        ("5-0001", "Rotfuchs", 5),
        ("7-0000", "Europa", 7),
    ]
    assert turn.facts[0].context == ["Der Fuchs."]
    assert turn.judgement.decided_by == "llm_judge"
    assert turn.judgement.reason == "Folgt aus 5-0001."
    assert turn.statements == [{"text": "Er lebt in Europa.", "fact_ids": ["5-0001", "7-0000"]}]
    assert r.turns[0].previous_text is None


def test_build_review_without_draws_still_waits_for_the_explicit_label():
    r = build_review("e", "T", TURNS, JUDGEMENTS, FACTS, 0.0, CLEAN)
    assert r.status == "pending" and r.turns == [] and len(r.draws) == 2


def test_review_without_draws_is_answered_by_the_explicit_label_and_clean():
    r = build_review("e", "T", TURNS, JUDGEMENTS, FACTS, 0.0, CLEAN)
    apply_labels(r, [comment(1, "explicit: korrekt")])
    assert r.status == "answered" and r.outcome == "clean"
    state = RateState(1.0)
    assert update_rate(state, r, REVIEW_CONFIG, NOW) and r.rate_applied == ["clean"]


def test_review_round_trips_through_json_with_stable_field_order():
    r = review()
    apply_labels(r, [comment(1, "0: belegt")])
    data = json.loads(json.dumps(r.to_dict()))
    assert Review.from_dict(data) == r
    assert list(data) == [
        "episode_id", "title", "rate", "seed", "population", "draws", "status", "outcome",
        "rate_applied", "issue", "issue_url", "explicit", "turns", "comments",
    ]  # fmt: skip
    assert list(data["explicit"]) == [
        "model", "predicted", "verdict", "findings", "reason", "label", "reviewer_flag", "note",
        "comment_id",
    ]  # fmt: skip
    assert list(data["turns"][0]) == [
        "turn_index", "speaker", "text", "previous_text", "fact_ids", "statements", "facts",
        "judgement", "label", "note", "comment_id",
    ]  # fmt: skip


# ---------------------------------------------------------------- comments


@pytest.mark.parametrize(
    ("login", "association", "expected"),
    [
        (REVIEWER, "OWNER", True),
        (REVIEWER, "MEMBER", True),
        (REVIEWER, "COLLABORATOR", True),
        (REVIEWER, "NONE", False),
        (REVIEWER, "CONTRIBUTOR", False),
        ("someone", "OWNER", False),
        ("ajfourteen", "MEMBER", False),
        ("github-actions[bot]", "NONE", False),
    ],
)
def test_only_the_reviewer_is_authorized(login, association, expected):
    c = Comment(1, login, association, "", "", "", "")
    assert is_authorized(c, REVIEWER, REVIEW_CONFIG["reviewer_associations"]) is expected


def test_parse_review_lines_with_notes_and_ignored_prose():
    body = (
        "Erste Runde.\n```\n0: belegt\n2:  Nicht   Belegt – Jahreszahl fehlt\n3:\n```\n"
        "> 2: belegt\nDanke!"
    )
    parsed = parse_comment(body, [0, 2, 3])
    assert parsed.errors == []
    assert [(p.turn_index, p.label, p.note) for p in parsed.lines] == [
        (0, "supported", None),
        (2, "not_supported", "Jahreszahl fehlt"),
    ]


@pytest.mark.parametrize("dash", ["-", "–", "—"])
def test_note_dashes(dash):
    (line,) = parse_comment(f"0: belegt {dash} passt", [0]).lines
    assert line.label == "supported" and line.note == "passt"


@pytest.mark.parametrize(
    ("body", "error"),
    [
        ("0: richtig", "unbekanntes Label"),
        ("0: belegtx", "unbekanntes Label"),
        ("5: belegt", "nicht in der Stichprobe"),
        ("0: belegt\n0: nicht belegt", "mehrfach"),
    ],
)
def test_parse_errors(body, error):
    assert error in parse_comment(body, [0, 2]).errors[0]


def test_later_comment_wins_and_a_comment_with_an_error_is_dropped_whole():
    r = review()
    comments = [
        comment(3, "0: nicht belegt", at="2026-11-19T12:00:00Z"),
        comment(1, "0: belegt\n2: belegt\nexplicit: korrekt", at="2026-11-19T11:00:00Z"),
        comment(4, "2: nicht belegt\n9: belegt", at="2026-11-19T13:00:00Z"),
        comment(5, "Nur ein Gedanke.", at="2026-11-19T14:00:00Z"),
    ]
    apply_labels(r, comments)
    assert [(t.label, t.comment_id) for t in r.turns] == [
        ("not_supported", 3),
        ("supported", 1),
    ]
    assert [(c.comment_id, c.applied) for c in r.comments] == [(1, True), (3, True), (4, False)]
    assert r.comments[2].errors == ["Redebeitrag 9 ist nicht in der Stichprobe."]
    assert r.status == "answered" and r.outcome == "error"


def test_deleted_comment_takes_its_labels_with_it():
    r = review()
    apply_labels(r, [comment(1, "0: belegt\n2: belegt\nexplicit: korrekt")])
    assert r.status == "answered" and r.outcome == "clean"
    apply_labels(r, [])
    assert r.status == "pending" and r.outcome is None and r.turns[0].label is None


def test_status_needs_every_turn_and_the_explicit_label():
    assert review_status([0, 2], [0], True, True) == "pending"
    assert review_status([0, 2], [2, 0], True, False) == "pending"  # explicit label missing
    assert review_status([0, 2], [2, 0], True, True) == "answered"
    assert review_status([], [], True, False) == "pending"  # only the explicit label to give
    assert review_status([], [], True, True) == "answered"


def test_status_of_reviews_from_before_the_explicit_check():
    assert review_status([], [], False, False) == "not_sampled"
    assert review_status([0, 2], [2, 0], False, False) == "answered"


def test_turn_labels_alone_do_not_answer_the_review():
    r = review()
    apply_labels(r, [comment(1, "0: belegt\n2: belegt")])
    assert r.status == "pending" and r.outcome is None
    assert update_rate(RateState(1.0), r, REVIEW_CONFIG, NOW) is False
    assert "Bitte noch das Explicit-Label ergänzen" in render_reply(r, None)


# ---------------------------------------------------------------- rate


def test_next_rate_follows_the_design():
    assert next_rate(1.0, "clean", REVIEW_CONFIG) == pytest.approx(2 / 3)
    assert next_rate(0.6, "error", REVIEW_CONFIG) == 1.0
    assert next_rate(0.25, "error", REVIEW_CONFIG) == 0.5
    assert next_rate(0.012, "clean", REVIEW_CONFIG) == 0.01


def test_rate_due_once_per_outcome_and_never_back_down():
    assert rate_due([], None) is None
    assert rate_due([], "clean") == "clean"
    assert rate_due([], "error") == "error"
    assert rate_due(["clean"], "clean") is None
    assert rate_due(["clean"], "error") == "error"
    assert rate_due(["error"], "clean") is None
    assert rate_due(["clean", "error"], "error") is None


def test_update_rate_records_the_step():
    r = review()
    r.issue = 7
    apply_labels(r, [comment(1, "0: belegt\n2: belegt\nexplicit: korrekt")])
    state = RateState(1.0)
    assert update_rate(state, r, REVIEW_CONFIG, NOW) is True
    assert state.r == pytest.approx(2 / 3) and r.rate_applied == ["clean"]
    (u,) = state.updates
    assert (u.episode_id, u.issue, u.outcome, u.at) == (
        "2026-11-19_rotfuchs",
        7,
        "clean",
        "2026-11-20T08:00:00Z",
    )
    assert update_rate(state, r, REVIEW_CONFIG, NOW) is False
    apply_labels(
        r, [comment(1, "0: belegt\n2: belegt\nexplicit: korrekt"), comment(2, "2: nicht belegt")]
    )
    assert update_rate(state, r, REVIEW_CONFIG, NOW) is True
    assert state.r == 1.0 and r.rate_applied == ["clean", "error"]


def test_fixed_rate_records_without_changing_r():
    r = review()
    apply_labels(r, [comment(1, "0: belegt\n2: belegt\nexplicit: korrekt")])
    state = RateState(1.0)
    assert update_rate(state, r, {**REVIEW_CONFIG, "adjust_rate": False}, NOW) is True
    assert state.r == 1.0 and state.updates[0].r_after == 1.0


def test_config_starts_at_100_percent_and_adjusts_after_phase_one():
    # Phase 1 exit criteria met on 2026-11-20; from then on r follows the rate rule.
    config = load_config()["review"]
    assert config["r_initial"] == 1.0 and config["adjust_rate"] is True


# ---------------------------------------------------------------- rendering


def test_escape_blocks_html_and_mentions():
    assert escape("an @someone <b>") == "an &#64;someone &lt;b&gt;"


def test_issue_shows_each_drawn_turn_and_a_template():
    links = {"Archiv": "https://x/archive"}
    title, messages = render_issue(review(), REVIEWER, links, 60000)
    assert title == "Review: Rotfuchs (2026-11-19_rotfuchs)"
    (body,) = messages
    marker = body.splitlines()[0]
    assert marker.startswith("<!-- gut-zu-wissen-review ") and marker.endswith(" -->")
    assert json.loads(marker[len("<!-- gut-zu-wissen-review ") : -4]) == {
        "episode_id": "2026-11-19_rotfuchs",
        "turns": [0, 2],
        "explicit": False,
    }
    assert "```\n0: \n2: \nexplicit: \n```" in body
    assert "[Archiv](https://x/archive)" in body
    assert "### Redebeitrag 0 – Lena" in body and "### Redebeitrag 2 – Lena" in body
    assert "Redebeitrag 1 " not in body and "Redebeitrag 3 " not in body
    assert "- `5-0001` (Rotfuchs, Revision 5): Er lebt in Europa." in body
    assert "  - Kontext: Der Fuchs." in body
    assert "- **Konfidenz (Jev p):** aussage_0 = 0.600" in body
    assert "- **Begründung:** Folgt aus 5-0001." in body
    assert "@someone" not in body and "<b>" not in body
    assert "r = 100 %" in body and "`ajFourteen`" in body and "@ajFourteen" not in body


def test_parsing_the_template_gives_no_labels_and_no_errors():
    _, (body,) = render_issue(review(), REVIEWER, {}, 60000)
    parsed = parse_comment(body, [0, 2])
    assert parsed.lines == [] and parsed.errors == []


def test_pack_splits_at_turn_boundaries_and_cuts_oversized_turns():
    messages = pack("H", ["a" * 60, "b" * 60, "c" * 500], 100, "F")
    assert all(len(m) <= 100 for m in messages)
    assert messages[0] == "H\n\n" + "a" * 60
    assert messages[1] == "F\n\n" + "b" * 60
    assert messages[2].startswith("F\n\nccc") and "gekürzt" in messages[2]


def test_long_issue_continues_in_comments():
    _, messages = render_issue(review(), REVIEWER, {}, 2500)
    assert len(messages) > 1 and all(len(m) <= 2500 for m in messages)
    assert messages[1].startswith("## Redebeiträge (Fortsetzung")


def test_reply_lists_missing_errors_and_rate():
    r = review()
    apply_labels(r, [comment(1, "0: nicht belegt"), comment(2, "2: falsch")])
    text = render_reply(r, None)
    assert "Labels übernommen: 1 von 2" in text and "Ohne Label: 2." in text
    assert "Nicht belegt: 0." in text and "unbekanntes Label" in text


# ---------------------------------------------------------------- archive and issues


class FakeIssues:
    def __init__(self, comments: list[Comment] | None = None, fail_comment: bool = False):
        self.comments = comments or []
        self.created: list[tuple[str, str]] = []
        self.posted: list[tuple[int, str]] = []
        self.closed: list[int] = []
        self.fail_comment = fail_comment

    def create_issue(self, title: str, body: str) -> tuple[int, str]:
        self.created.append((title, body))
        return 7, "https://x/issues/7"

    def create_comment(self, number: int, body: str) -> str:
        if self.fail_comment:
            raise RuntimeError("down")
        self.posted.append((number, body))
        return "https://x/c"

    def list_comments(self, number: int) -> list[Comment]:
        assert number == 7
        return self.comments

    def close_issue(self, number: int) -> None:
        self.closed.append(number)


def archived(tmp_path, rate: float = 1.0) -> Review:
    r = review(rate)
    save_review(tmp_path, r)
    return r


def test_open_issue_records_number_and_is_idempotent(tmp_path):
    archived(tmp_path)
    issues = FakeIssues()
    opened = open_issue(tmp_path, "2026-11-19_rotfuchs", issues, REVIEW_CONFIG, {})
    assert opened.issue == 7 and opened.issue_url == "https://x/issues/7"
    assert load_review(tmp_path, "2026-11-19_rotfuchs").issue == 7
    open_issue(tmp_path, "2026-11-19_rotfuchs", issues, REVIEW_CONFIG, {})
    assert len(issues.created) == 1


def test_open_issue_posts_overflow_as_comments(tmp_path):
    archived(tmp_path)
    issues = FakeIssues()
    open_issue(tmp_path, "2026-11-19_rotfuchs", issues, SMALL_ISSUES, {})
    assert issues.posted and all(n == 7 for n, _ in issues.posted)


def test_issue_number_is_kept_when_a_follow_up_comment_fails(tmp_path):
    archived(tmp_path)
    issues = FakeIssues(fail_comment=True)
    with pytest.raises(RuntimeError):
        open_issue(tmp_path, "2026-11-19_rotfuchs", issues, SMALL_ISSUES, {})
    assert load_review(tmp_path, "2026-11-19_rotfuchs").issue == 7


def test_issue_without_drawn_turns_is_for_the_explicit_label_only(tmp_path):
    archived(tmp_path, rate=0.0)
    issues = FakeIssues()
    assert open_issue(tmp_path, "2026-11-19_rotfuchs", issues, REVIEW_CONFIG, {}).issue == 7
    ((_, body),) = issues.created
    assert "dient nur dem Explicit-Label" in body and "## Explicit-Prüfung (D7)" in body
    assert "### Redebeitrag" not in body and "```\nexplicit: \n```" in body


def test_no_issue_for_a_review_from_before_the_explicit_check(tmp_path):
    r = review(rate=0.0)
    r.explicit, r.status = None, "not_sampled"
    save_review(tmp_path, r)
    issues = FakeIssues()
    assert open_issue(tmp_path, "2026-11-19_rotfuchs", issues, REVIEW_CONFIG, {}).issue is None
    assert issues.created == []


def test_apply_writes_labels_and_rate_from_the_reviewer_only(tmp_path):
    r = archived(tmp_path)
    r.issue = 7
    save_review(tmp_path, r)
    issues = FakeIssues(
        [
            comment(1, "0: nicht belegt\n2: nicht belegt", login="someone"),
            Comment(2, REVIEWER, "NONE", "0: nicht belegt", "2026-11-19T10:00:00Z", "", ""),
            comment(3, "0: belegt\n2: belegt – passt\nexplicit: korrekt"),
        ]
    )
    result = apply_issue(tmp_path, 7, issues, REVIEW_CONFIG, NOW)
    stored = load_review(tmp_path, "2026-11-19_rotfuchs")
    assert [(t.label, t.note) for t in stored.turns] == [
        ("supported", None),
        ("supported", "passt"),
    ]
    assert [c.comment_id for c in stored.comments] == [3]  # nobody else is recorded
    assert stored.status == "answered" and stored.rate_applied == ["clean"]
    state = load_rate_state(tmp_path, REVIEW_CONFIG)
    assert state.r == pytest.approx(2 / 3) and result.rate_update == state.updates[0]
    assert "Review vollständig: fehlerfrei." in result.reply
    path = tmp_path / "state" / "review.json"
    assert list(json.loads(path.read_text())) == ["r", "updates"]
    # A second run with the same comments changes nothing.
    before = path.read_text()
    apply_issue(tmp_path, 7, issues, REVIEW_CONFIG, NOW)
    assert path.read_text() == before


def test_closes_issue_only_on_the_change_to_answered():
    assert closes_issue("pending", "answered")
    assert not closes_issue("answered", "answered")  # later correction
    assert not closes_issue("pending", "pending")
    assert not closes_issue("not_sampled", "not_sampled")


def test_apply_closes_the_issue_once_when_turns_and_explicit_are_labelled(tmp_path):
    r = archived(tmp_path)
    r.issue = 7
    save_review(tmp_path, r)
    issues = FakeIssues([comment(1, "0: belegt")])
    first = apply_issue(tmp_path, 7, issues, REVIEW_CONFIG, NOW)
    assert first.review.status == "pending" and not first.close
    assert "geschlossen" not in first.reply
    issues.comments.append(comment(2, "2: belegt", at="2026-11-19T11:30:00Z"))
    second = apply_issue(tmp_path, 7, issues, REVIEW_CONFIG, NOW)
    assert second.review.status == "pending" and not second.close  # explicit label missing
    assert "Bitte noch das Explicit-Label ergänzen" in second.reply
    assert second.rate_update is None
    issues.comments.append(comment(3, "explicit: korrekt", at="2026-11-19T11:45:00Z"))
    third = apply_issue(tmp_path, 7, issues, REVIEW_CONFIG, NOW)
    assert third.review.status == "answered" and third.close
    assert "Das Issue wird geschlossen." in third.reply
    assert "Bitte noch das Explicit-Label" not in third.reply
    issues.comments.append(comment(4, "2: nicht belegt", at="2026-11-19T12:00:00Z"))
    fourth = apply_issue(tmp_path, 7, issues, REVIEW_CONFIG, NOW)
    assert fourth.review.outcome == "error" and not fourth.close
    assert issues.closed == []  # apply only decides; the reply command closes


def test_closing_reply_with_explicit_label_does_not_ask_for_it():
    r = review()
    apply_labels(r, [comment(1, "0: belegt\n2: belegt\nexplicit: korrekt")])
    text = render_reply(r, None, closing=True)
    assert "Das Issue wird geschlossen." in text and "Bitte noch das Explicit-Label" not in text


def test_apply_ignores_issues_that_are_not_in_the_archive(tmp_path):
    archived(tmp_path)
    result = apply_issue(tmp_path, 7, FakeIssues(), REVIEW_CONFIG, NOW)
    assert result.review is None and result.reply is None


def test_find_review_by_issue_number(tmp_path):
    r = archived(tmp_path)
    assert find_review(tmp_path, 7) is None
    r.issue = 7
    save_review(tmp_path, r)
    assert find_review(tmp_path, 7) == r


def test_rate_state_starts_at_r_initial(tmp_path):
    assert load_rate_state(tmp_path, REVIEW_CONFIG).r == REVIEW_CONFIG["r_initial"]


# ---------------------------------------------------------------- command line


def test_command_line_open_apply_reply(tmp_path, monkeypatch):
    import gut_zu_wissen.run
    from gut_zu_wissen import review as module

    archived(tmp_path)
    issues = FakeIssues(
        [comment(3, "0: belegt\n2: nicht belegt – falsches Jahr\nexplicit: korrekt")]
    )
    monkeypatch.setattr(gut_zu_wissen.run, "ARCHIVE_ROOT", tmp_path)
    monkeypatch.setattr(module, "_issues_from_env", lambda: issues)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")

    assert module.main(["open", "2026-11-19_rotfuchs"]) == 0
    ((_, body),) = issues.created
    assert "[Archiv](https://github.com/o/r/tree/main/archive/episodes/2026-11-19_rotfuchs)" in body
    assert "(https://github.com/o/r/actions/runs/42)" in body

    reply = tmp_path / "build" / "reply.md"
    close = tmp_path / "build" / "close"
    files = ["--reply-file", str(reply), "--close-file", str(close)]
    assert module.main(["apply", "7", *files]) == 0
    stored = load_review(tmp_path, "2026-11-19_rotfuchs")
    assert stored.outcome == "error" and stored.turns[1].note == "falsches Jahr"
    assert load_rate_state(tmp_path, REVIEW_CONFIG).r == 1.0  # phase 1: fixed
    assert close.exists()
    assert module.main(["reply", "7", str(reply), "--close-file", str(close)]) == 0
    ((number, text),) = issues.posted
    assert number == 7 and "Review vollständig: Fehler gefunden." in text
    assert issues.closed == [7]  # after the reply

    # A correction on the closed issue is applied; the issue is not closed (or reopened) again.
    issues.comments.append(comment(4, "2: belegt", at="2026-11-19T12:00:00Z"))
    assert module.main(["apply", "7", *files]) == 0
    assert load_review(tmp_path, "2026-11-19_rotfuchs").turns[1].label == "supported"
    assert not close.exists()
    assert module.main(["reply", "7", str(reply), "--close-file", str(close)]) == 0
    assert len(issues.posted) == 2 and issues.closed == [7]

    assert module.main(["apply", "8", *files]) == 0
    assert not reply.exists() and not close.exists()
    assert module.main(["reply", "8", str(reply), "--close-file", str(close)]) == 0
    assert len(issues.posted) == 2


def test_command_line_rejects_a_malformed_episode_id(monkeypatch):
    from gut_zu_wissen import review as module

    assert module.main(["open", "../../etc"]) == 2


# ---------------------------------------------------------------- explicit check (D7)


def test_build_review_keeps_the_explicit_result():
    e = review(explicit=FLAGGED).explicit
    assert (e.model, e.predicted, e.verdict, e.reason) == (
        "claude-sonnet-5-5",
        True,
        "explicit",
        "Drastische Jagdszene.",
    )
    assert e.findings == [
        {"turn_index": 2, "category": "violence", "quote": "reißt <die> Beute",
         "reason": "Blutig geschildert."}
    ]  # fmt: skip
    assert e.label is None and e.reviewer_flag is None


def test_issue_shows_the_explicit_result():
    _, (body,) = render_issue(review(explicit=FLAGGED), REVIEWER, {}, 60000)
    assert "## Explicit-Prüfung (D7)" in body
    assert "- **Flag:** explicit" in body
    assert "- **Urteil des Modells:** `explicit`" in body
    assert "- **Begründung:** Drastische Jagdszene." in body
    assert "  - Redebeitrag 2, Gewalt: „reißt &lt;die&gt; Beute“ – Blutig geschildert." in body
    _, (clean,) = render_issue(review(), REVIEWER, {}, 60000)
    assert "- **Flag:** nicht explicit" in clean and "- **Fundstellen:** keine" in clean


@pytest.mark.parametrize(
    ("value", "predicted", "label"),
    [
        ("korrekt", True, "correct"),
        ("Korrekt", False, "correct"),
        ("sollte explicit sein", False, "should_be_explicit"),
        ("sollte  nicht explicit sein", True, "should_not_be_explicit"),
    ],
)
def test_parse_explicit_label(value, predicted, label):
    parsed = parse_comment(f"0: belegt\nExplicit: {value} – Notiz", [0], predicted)
    assert parsed.errors == [] and parsed.explicit.label == label
    assert parsed.explicit.note == "Notiz" and len(parsed.lines) == 1


@pytest.mark.parametrize(
    ("body", "predicted", "error"),
    [
        ("explicit: ja", True, "unbekanntes Label"),
        ("explicit: sollte explicit sein", True, "korrekt"),
        ("explicit: sollte nicht explicit sein", False, "korrekt"),
        ("explicit: korrekt", None, "kein Ergebnis"),
        ("explicit: korrekt\nexplicit: korrekt", True, "mehrfach"),
    ],
)
def test_parse_explicit_errors(body, predicted, error):
    assert error in parse_comment(body, [0], predicted).errors[0]


def test_empty_explicit_template_line_is_skipped():
    parsed = parse_comment("explicit: \n0:", [0], True)
    assert parsed.explicit is None and parsed.errors == [] and parsed.count() == 0


@pytest.mark.parametrize(
    ("predicted", "label", "expected"),
    [
        (True, "correct", True),
        (False, "correct", False),
        (False, "should_be_explicit", True),
        (True, "should_not_be_explicit", False),
        (True, None, None),
    ],
)
def test_reviewer_flag(predicted, label, expected):
    assert reviewer_flag(predicted, label) is expected


def test_explicit_label_is_folded_like_turn_labels():
    r = review(explicit=FLAGGED)
    apply_labels(
        r,
        [
            comment(1, "explicit: korrekt", at="2026-11-19T11:00:00Z"),
            comment(
                2, "explicit: sollte nicht explicit sein – nur Jagd", at="2026-11-19T12:00:00Z"
            ),
            comment(3, "explicit: sollte explicit sein", at="2026-11-19T13:00:00Z"),  # error
        ],
    )
    e = r.explicit
    assert (e.label, e.reviewer_flag, e.note, e.comment_id) == (
        "should_not_be_explicit",
        False,
        "nur Jagd",
        2,
    )
    assert [(c.comment_id, c.applied) for c in r.comments] == [(1, True), (2, True), (3, False)]
    # The explicit label alone does not answer the review; the drawn turns need labels too.
    assert r.status == "pending" and r.outcome is None
    apply_labels(r, [])
    assert r.explicit.label is None and r.explicit.reviewer_flag is None


def test_reply_reports_the_explicit_label():
    r = review(explicit=FLAGGED)
    apply_labels(r, [comment(1, "explicit: sollte nicht explicit sein")])
    assert "Explicit: Prüfung explicit, Review nicht explicit." in render_reply(r, None)
    apply_labels(r, [])
    assert "Explicit: noch ohne Label (Prüfung: explicit)." in render_reply(r, None)


def test_explicit_confusion_counts_and_rates():
    pairs = [(True, True), (True, False), (False, False), (False, False), (False, True)]
    a = explicit_confusion(pairs)
    assert (a.episodes, a.true_positive, a.false_positive, a.false_negative, a.true_negative) == (
        5, 1, 1, 1, 2,
    )  # fmt: skip
    assert a.false_positive_rate == pytest.approx(1 / 3)
    assert a.false_negative_rate == pytest.approx(1 / 2)


def test_explicit_confusion_without_data_has_no_rates():
    a = explicit_confusion([])
    assert a.episodes == 0 and a.false_positive_rate is None and a.false_negative_rate is None
    only_clean = explicit_confusion([(False, False)])
    assert only_clean.false_positive_rate == 0.0 and only_clean.false_negative_rate is None


def test_explicit_pairs_skip_unlabelled_episodes():
    labelled = review(explicit=FLAGGED)
    apply_labels(labelled, [comment(1, "explicit: korrekt")])
    assert explicit_pairs([labelled, review()]) == [(True, True)]


def test_explicit_accuracy_over_the_archive(tmp_path):
    a = review(explicit=FLAGGED)
    apply_labels(a, [comment(1, "explicit: sollte nicht explicit sein")])
    save_review(tmp_path, a)
    b = build_review("2026-11-20_wolf", "Wolf", TURNS, JUDGEMENTS, FACTS, 1.0, CLEAN)
    apply_labels(b, [comment(2, "explicit: korrekt")])
    save_review(tmp_path, b)
    save_review(
        tmp_path, build_review("2026-11-21_luchs", "Luchs", TURNS, JUDGEMENTS, FACTS, 1.0, CLEAN)
    )
    accuracy = explicit_accuracy(tmp_path)
    assert accuracy.episodes == 2 and accuracy.false_positive == 1 and accuracy.true_negative == 1
    assert accuracy.false_positive_rate == 0.5 and accuracy.false_negative_rate is None


def test_apply_writes_the_explicit_label_from_the_reviewer_only(tmp_path):
    r = review(explicit=FLAGGED)
    r.issue = 7
    save_review(tmp_path, r)
    issues = FakeIssues(
        [
            comment(1, "explicit: sollte nicht explicit sein", login="someone"),
            comment(2, "explicit: korrekt"),
        ]
    )
    apply_issue(tmp_path, 7, issues, REVIEW_CONFIG, NOW)
    e = load_review(tmp_path, "2026-11-19_rotfuchs").explicit
    assert (e.label, e.reviewer_flag, e.comment_id) == ("correct", True, 2)
