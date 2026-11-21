import pytest

from gut_zu_wissen.dialog import (
    CONTINUE_PROMPT,
    DIALOG_SCHEMA,
    EXCLUDED_FACT,
    SYSTEM_PROMPT,
    TURN_PROMPT,
    TURN_SCHEMA,
    Statement,
    Turn,
    author_reason,
    check_turns,
    continue_dialog,
    format_offer,
    format_prefix,
    format_rejections,
    format_turn_context,
    offer_aliases,
    parse_dialog,
    regenerate_turn,
    repetition_problems,
    repetition_reason,
    resolve_turns,
    told_elsewhere,
    told_in,
    top_facts_gate,
    used_fact_ids,
    used_facts_gate,
    write_dialog,
)
from gut_zu_wissen.extraction import Fact
from gut_zu_wissen.ranking import RankedFact

OFFER = {"1-0000", "1-0001", "2-0000"}


def rf(sid: str, article: str = "Haupt", context: list[str] | None = None) -> RankedFact:
    return RankedFact(
        1, article, int(sid[0]), Fact(sid, "sentence", "", [], f"Fakt {sid}.", context or [])
    )


def turn(i: int, speaker="Lena", text="Hallo.", ids=(), statements=()) -> Turn:
    return Turn(i, speaker, text, list(ids), [Statement(t, list(f)) for t, f in statements])


def test_format_offer_marks_linked_facts_and_context():
    out = format_offer([rf("1-0000", context=["Davor."])], [rf("2-0000", "Nebenartikel")])
    assert out == (
        "[F1] Fakt 1-0000.\n  Kontext: Davor.\n[F2] (verlinkter Artikel: Nebenartikel) Fakt 2-0000."
    )


def test_format_offer_lists_einordnung_first_with_the_first_aliases():
    einordnung = [rf("1-0003"), rf("1-0004")]
    out = format_offer([rf("1-0000")], [rf("2-0000", "N")], einordnung)
    assert out.split("\n") == [
        "[F1] (Einordnung) Fakt 1-0003.",
        "[F2] (Einordnung) Fakt 1-0004.",
        "[F3] Fakt 1-0000.",
        "[F4] (verlinkter Artikel: N) Fakt 2-0000.",
    ]
    aliases = offer_aliases([rf("1-0000")], [rf("2-0000", "N")], einordnung)
    assert list(aliases.values()) == ["1-0003", "1-0004", "1-0000", "2-0000"]


def test_dialog_prompt_opens_with_a_short_einordnung():
    prompt = " ".join(SYSTEM_PROMPT.split())
    for rule in [
        "Fakten mit dem Vermerk „(Einordnung)“ stehen am Anfang des Angebots",
        "Das Gespräch beginnt mit einer kurzen Einordnung aus diesen Fakten, bevor die erste "
        "Überraschung kommt: was das Thema ist und, wenn die Fakten es sagen, wann und wo.",
        "Lena erzählt sie; Felix darf mit einer Einstiegsfrage beginnen.",
        "Halte die Einordnung kurz: ein oder zwei Redebeiträge.",
        "sie zählen als verwendete Fakten und dürfen in der Zusammenfassung wiederkommen.",
    ]:
        assert rule in prompt


def test_parse_keeps_order_and_tolerates_malformed_entries():
    answer = {
        "turns": [
            {
                "speaker": "Lena",
                "text": "A",
                "fact_ids": ["1-0000", 3],
                "statements": [{"text": "S", "fact_ids": ["1-0000"]}, "kaputt"],
            },
            "kaputt",
            {"speaker": "Felix", "text": "B"},
        ]
    }
    turns = parse_dialog(answer)
    assert [t.index for t in turns] == [0, 1]
    assert turns[0].fact_ids == ["1-0000"] and turns[0].statements == [Statement("S", ["1-0000"])]
    assert turns[1].fact_ids == [] and turns[1].statements == []


def test_valid_turns_pass_the_gate():
    turns = [turn(0, ids=["1-0000"], statements=[("S", ["1-0000"])]), turn(1, "Felix", "Echt?")]
    assert check_turns(turns, OFFER) == []


def test_gate_reports_unknown_ids_in_turn_and_statement():
    turns = [turn(0, ids=["9-9"], statements=[("S", ["8-8"])])]
    (problem,) = check_turns(turns, OFFER)
    assert problem.turn_index == 0 and "8-8" in problem.reason and "9-9" in problem.reason


def test_archived_unknown_id_problem_names_the_token_as_written():
    (problem,) = check_turns([turn(0, ids=["F53"])], OFFER)
    assert problem.reason == (
        "Unknown fact IDs: F53 (cited by the author; no offered fact has this identifier)."
    )


def test_gate_reports_statement_without_ids_unknown_speaker_and_empty_text():
    turns = [turn(0, "Hans", " ", statements=[("S", [])])]
    reasons = [p.reason for p in check_turns(turns, OFFER)]
    assert len(reasons) == 3


def test_used_facts_counts_turn_and_statement_ids_once_ignoring_unknown():
    turns = [
        turn(0, ids=["1-0000", "9-9"], statements=[("S", ["1-0001"])]),
        turn(1, ids=["1-0000", "2-0000"]),
    ]
    assert used_fact_ids(turns, OFFER) == ["1-0000", "1-0001", "2-0000"]


def test_top_facts_gate_requires_unexcluded_top_facts():
    assert top_facts_gate({"a", "b"}, ["a", "b", "c"]) is None
    assert top_facts_gate({"a", "b"}, ["a"], exempt={"b"}) is None
    message = top_facts_gate({"a", "b"}, ["a"])
    assert message is not None and "D8" in message and "b" in message
    assert top_facts_gate(set(), []) is None


def test_format_offer_marks_must_use_facts_in_the_main_list_only():
    main = [rf("1-0001"), rf("1-0002")]
    text = format_offer(main, [], [], {"1-0002", "1-0009"})
    lines = [line for line in text.split("\n") if line.startswith("[")]
    assert lines[0].startswith("[F1] ") and "(Pflicht)" not in lines[0]
    assert lines[1].startswith("[F2] (Pflicht) ")


def test_used_facts_gate_at_the_boundary():
    assert used_facts_gate(10, 10) is None
    assert "D5" in used_facts_gate(9, 10)


class FakeLlm:
    def __init__(self, answer: dict) -> None:
        self.answer, self.calls = answer, []
        self.caches: list[dict] = []

    def complete_json(self, system: str, user: str, schema: dict, **cache) -> dict:
        self.calls.append((system, user, schema))
        self.caches.append(cache)
        return self.answer


def test_write_dialog_sends_offer_and_records_problems_and_use():
    answer = {
        "turns": [
            {
                "speaker": "Lena",
                "text": "A",
                "fact_ids": ["1-0000"],
                "statements": [{"text": "S", "fact_ids": ["1-0000"]}],
            },
            {"speaker": "Felix", "text": "B", "fact_ids": ["F7"], "statements": []},
        ]
    }
    llm = FakeLlm(answer)
    dialog = write_dialog(llm, [rf("1-0000")], [rf("2-0000", "N")], 10, 2)
    system, user, schema = llm.calls[0]
    assert "mindestens 10" in system and "[F2]" in user and schema is DIALOG_SCHEMA
    assert "[1-0000]" not in user and "[2-0000]" not in user
    assert dialog.used_fact_ids == ["1-0000"]
    assert [p.turn_index for p in dialog.problems] == [1]


def test_write_dialog_maps_aliases_back_to_segment_ids():
    answer = {
        "turns": [
            {
                "speaker": "Lena",
                "text": "A",
                "fact_ids": ["F1", " [f2] "],
                "statements": [
                    {"text": "S", "fact_ids": ["f1"]},
                    {"text": "T", "fact_ids": ["F2"]},
                ],
            },
        ]
    }
    dialog = write_dialog(FakeLlm(answer), [rf("1-0000")], [rf("2-0000", "N")], 1, 2)
    (t,) = dialog.turns
    assert t.fact_ids == ["1-0000", "2-0000"]
    assert [s.fact_ids for s in t.statements] == [["1-0000"], ["2-0000"]]
    assert dialog.problems == [] and dialog.used_fact_ids == ["1-0000", "2-0000"]


ALIASES = offer_aliases([rf("1-0000"), rf("1-0001")], [rf("2-0000", "N")])


def test_offer_aliases_main_first_then_linked():
    assert ALIASES == {"F1": "1-0000", "F2": "1-0001", "F3": "2-0000"}


def test_resolve_turns_maps_turn_and_statement_ids_and_keeps_unknown_tokens():
    (t,) = resolve_turns([turn(0, ids=["F3", "1-0001", "F9"], statements=[("S", ["f1"])])], ALIASES)
    assert t.fact_ids == ["2-0000", "1-0001", "F9"]
    assert t.statements == [Statement("S", ["1-0000"])]
    (problem,) = check_turns([t], OFFER)
    assert "F9" in problem.reason


def test_author_reason_for_unknown_ids_is_german_and_names_the_offer_range():
    old = turn(0, ids=["F53", "1-0000"])
    archived = " ".join(p.reason for p in check_turns([old], ALIASES.values()))
    assert author_reason(old, archived, ALIASES) == (
        "Die Kennung F53 gibt es im Angebot nicht. Verwende nur Kennungen aus dem Angebot "
        "(F1–F3); was du mit keiner angebotenen Kennung belegen kannst, lässt du weg."
    )


def test_author_reason_shows_offered_ids_as_aliases_and_keeps_other_reasons():
    old = turn(0, ids=["1-0001"])
    assert author_reason(old, "Fakt 1-0001 trägt das nicht.", ALIASES) == "Fakt F2 trägt das nicht."
    assert author_reason(old, None, ALIASES) is None


def test_turn_context_has_neighbours_old_turn_and_reason():
    turns = [turn(0, text="Eins."), turn(1, "Felix", "Zwei."), turn(2, text="Drei.")]
    out = format_turn_context(turns, 1, "Zahl falsch.", ALIASES)
    assert out.index("Eins.") < out.index("Zwei.") < out.index("Zahl falsch.") < out.index("Drei.")
    assert "(Felix)" in out and "Kennungen" not in out  # no cited facts, no line
    assert "Vorheriger" not in format_turn_context(turns, 0, None, ALIASES)
    assert "Nächster" not in format_turn_context(turns, 2, None, ALIASES)
    assert "neu geschrieben" in format_turn_context(turns, 2, None, ALIASES)


def test_turn_context_renders_neighbours_and_old_turn_with_aliases():
    turns = [
        turn(0, text="Eins.", ids=["1-0000"], statements=[("S", ["1-0001"])]),
        turn(1, "Felix", "Zwei.", ids=["2-0000", "F53"]),
        turn(2, text="Drei.", ids=["1-0001"]),
    ]
    out = format_turn_context(turns, 1, "Grund.", ALIASES)
    assert out.split("\n") == [
        "Vorheriger Redebeitrag (Lena):",
        "Eins.",
        "Kennungen: F1, F2",
        "",
        "Zu ersetzender Redebeitrag (Felix):",
        "Zwei.",
        "Kennungen: F3",  # the unknown token is named in the reason, not repeated here
        "",
        "Grund für den Ersatz:",
        "Grund.",
        "",
        "Nächster Redebeitrag (Lena):",
        "Drei.",
        "Kennungen: F2",
        "",
        "Schon in anderen Redebeiträgen erzählt (nicht noch einmal erzählen; nur eine kurze "
        "Nachfrage direkt nach der Erzählung darf sich darauf beziehen):",
        "- F2: Redebeitrag 0",
    ]
    assert not any(sid in out for sid in ALIASES.values())


def test_told_elsewhere_skips_the_replaced_turn_and_the_closing_summary():
    # User decision 2026-11-20: the author sees which facts are already told when rewriting.
    turns = [
        turn(0, statements=[("A", ["1-0000"])]),
        turn(1, statements=[("B", ["1-0000", "1-0001"])]),
        turn(2, statements=[("C", ["1-0001"]), ("C2", ["1-0001"])]),
        turn(3, statements=[("Summary", ["1-0000", "2-0000"])]),
    ]
    assert told_elsewhere(turns, 1) == {"1-0000": [0], "1-0001": [2]}
    assert told_elsewhere(turns, 0) == {"1-0000": [1], "1-0001": [1, 2]}
    assert "Schon in anderen" not in format_turn_context(turns[:1], 0, None, ALIASES)


def test_regenerate_turn_keeps_index_and_speaker_and_uses_turn_schema():
    answer = {"speaker": "Lena", "text": "Neu.", "fact_ids": ["[F1]"], "statements": []}
    llm = FakeLlm(answer)
    turns = [turn(0), turn(1, "Felix", "Alt.")]
    new = regenerate_turn(llm, "[F1] Fakt.", ALIASES, turns, 1, "Grund.")
    assert (new.index, new.speaker, new.text, new.fact_ids) == (1, "Felix", "Neu.", ["1-0000"])
    _, user, schema = llm.calls[0]
    assert schema is TURN_SCHEMA and "[F1] Fakt." in user and "Grund." in user


def test_regenerate_turn_caches_the_offer_at_the_start_of_the_user_message():
    answer = {"speaker": "Lena", "text": "Neu.", "fact_ids": [], "statements": []}
    llm = FakeLlm(answer)
    turns = [turn(0), turn(1, "Felix", "Alt.")]
    regenerate_turn(llm, "[F1] Fakt.", ALIASES, turns, 1, "Grund.")
    regenerate_turn(llm, "[F1] Fakt.", ALIASES, turns, 0, None)  # a cascade step: same prefix
    prefix = "Angebot:\n[F1] Fakt.\n\n"
    assert llm.caches == [{"cache_user_prefix": prefix}] * 2
    assert all(user.startswith(prefix) for _, user, _ in llm.calls)
    assert llm.calls[0][1] != llm.calls[1][1]  # only the part after the offer varies


def test_write_and_continuation_mark_no_cache_prefix():
    # Neither prefix is sent again (own schema and system prompt; a continuation's offer has
    # new exclusions), so a breakpoint would only pay the write price.
    llm = FakeLlm({"turns": []})
    write_dialog(llm, [], [], 10, 2)
    continue_dialog(llm, "[F1] Fakt.", ALIASES, PREFIX, 10)
    assert llm.caches == [{}, {}]
    assert llm.calls[1][1].startswith("Angebot:\n[F1] Fakt.\n\n")  # offer still first


def test_regenerate_turn_shows_the_author_the_german_unknown_id_problem():
    turns = [turn(0, ids=["F53"])]
    archived = " ".join(p.reason for p in check_turns(turns, ALIASES.values()))
    llm = FakeLlm({"speaker": "Lena", "text": "Neu.", "fact_ids": [], "statements": []})
    regenerate_turn(llm, "[F1] Fakt.", ALIASES, turns, 0, archived)
    _, user, _ = llm.calls[0]
    assert "Die Kennung F53 gibt es im Angebot nicht." in user and "(F1–F3)" in user
    assert "Unknown fact IDs" not in user


def told(i: int, *ids: str, speaker: str = "Lena") -> Turn:
    """A turn whose one statement cites `ids`."""
    return turn(i, speaker, f"Text {i}.", ids, [(f"S{i}", ids)])


def test_repetition_gate_rejects_the_third_turn_citing_a_fact_with_the_real_id():
    turns = [told(0, "1-0000"), told(1, "1-0000"), told(2, "1-0000"), told(3, "1-0001")]
    (problem,) = repetition_problems(turns, OFFER, 2)
    assert problem.turn_index == 2
    assert problem.reason == (
        "Der Fakt 1-0000 wurde schon in Redebeitrag 0 und 1 erzählt. Erzähle ihn nicht noch "
        "einmal; gehe zum nächsten Fakt über oder reagiere kurz ohne seine Details."
    )


def test_repetition_gate_rejects_every_further_turn_and_names_all_earlier_ones():
    turns = [told(i, "1-0000") for i in range(5)] + [told(5, "1-0001")]
    problems = repetition_problems(turns, OFFER, 2)
    assert [p.turn_index for p in problems] == [2, 3, 4]
    assert "in Redebeitrag 0, 1, 2 und 3 erzählt" in problems[-1].reason


def test_repetition_gate_does_not_count_or_check_the_last_turn():
    turns = [told(0, "1-0000"), told(1, "1-0000"), told(2, "1-0000")]
    assert repetition_problems(turns, OFFER, 2) == []
    turns = [told(0, "1-0000"), told(1, "1-0000"), told(2, "1-0001"), told(3, "1-0000")]
    assert repetition_problems(turns, OFFER, 2) == []


def test_repetition_gate_counts_turns_not_statements_and_only_statement_ids():
    twice = turn(1, ids=["1-0000"], statements=[("A", ["1-0000"]), ("B", ["1-0000", "1-0001"])])
    turn_ids_only = turn(2, ids=["1-0000"])  # reaction with turn-level IDs but no statement
    turns = [told(0, "1-0000"), twice, turn_ids_only, told(3, "1-0001"), told(4, "2-0000")]
    assert repetition_problems(turns, OFFER, 2) == []


def test_repetition_gate_ignores_unknown_ids_and_follows_the_limit():
    turns = [told(0, "9-9"), told(1, "9-9"), told(2, "9-9"), told(3, "1-0000")]
    assert repetition_problems(turns, OFFER, 2) == []
    turns = [told(0, "1-0000", "1-0001"), told(1, "1-0001"), told(2, "2-0000")]
    (problem,) = repetition_problems(turns, OFFER, 1)
    assert (problem.turn_index, "1-0001" in problem.reason) == (1, True)


def test_repetition_reason_lists_one_earlier_turn():
    assert repetition_reason("1-0000", [3]).startswith(
        "Der Fakt 1-0000 wurde schon in Redebeitrag 3 erzählt."
    )


def test_author_sees_the_repetition_problem_with_the_alias():
    reason = repetition_reason("1-0001", [3, 4])
    assert author_reason(told(5, "1-0001"), reason, ALIASES).startswith(
        "Der Fakt F2 wurde schon in Redebeitrag 3 und 4 erzählt."
    )


def test_write_dialog_records_repetition_problems_in_turn_order():
    def t(fid: str) -> dict:
        return {
            "speaker": "Lena",
            "text": "A",
            "fact_ids": [fid],
            "statements": [{"text": "S", "fact_ids": [fid]}],
        }

    answer = {"turns": [t("F1"), t("F1"), t("F1"), t("F9"), t("F2")]}
    dialog = write_dialog(FakeLlm(answer), [rf("1-0000")], [rf("2-0000", "N")], 1, 2)
    assert [p.turn_index for p in dialog.problems] == [2, 3]
    assert "Der Fakt 1-0000" in dialog.problems[0].reason
    assert dialog.problems[1].reason.startswith("Unknown fact IDs")


@pytest.mark.parametrize("prompt", [SYSTEM_PROMPT, TURN_PROMPT, CONTINUE_PROMPT])
@pytest.mark.parametrize(
    "rule",
    [
        # (a) only the cited fact sentence, never the context
        "Sage nur, was der zitierte Fakt-Satz selbst sagt.",
        "was nur im Kontext steht, sprichst du nie aus",
        # (b) question premises referring back carry fact IDs
        "Jede Frage, deren Prämisse sich auf zuvor Gesagtes bezieht",
        "„Wofür stand der Axolotl in diesem Film?“",
        # (c) reactions without fact IDs repeat no numbers, names or facts
        "Eine Reaktion ohne Kennungen wiederholt keine Zahlen, Namen oder Fakten",
        "„Vier Monate Suche und kein einziger Fund.“ ohne Kennungen",
        # (d) no generalisations beyond the fact sentence
        "Lena verallgemeinert nicht über den Fakt-Satz hinaus.",
        "„Es gibt keine Zellen, die einfach alles werden“",
        "„Auf diese Eigenschaft wurde mit dem Namen angespielt“",
        # (e) short evaluations with content need fact IDs or are left out (user, 2026-11-20)
        "Kurze Wertungen, die etwas über die Sache behaupten",
        "Frei bleiben reine Gefühlsausrufe",
        # (f) only one closing summary (user, 2026-11-20)
        "Am Ende fasst genau eine der beiden Personen das Gespräch einmal kurz zusammen.",
        "Keine zweite Zusammenfassung, kein abwechselndes Zusammenfassen.",
        "Auch die Zusammenfassung braucht Statements und Kennungen.",
        # (g) each fact is told once (user, 2026-11-20)
        "Lena erzählt jeden Fakt nur einmal.",
        "mit einer Nachfrage oder Reaktion direkt im Anschluss, ohne seine Details (Zahlen, Namen) "
        "zu wiederholen",
        "Erst in der Zusammenfassung, dem letzten Redebeitrag, darf er wiederkommen.",
        "Jeder Fakt ist ein neuer Schritt in der Geschichte",
        "Ein späterer Redebeitrag, der einen schon erzählten Fakt noch einmal erzählt oder "
        "zusammenfasst, ist nicht erlaubt.",
        # (i) follow-up questions do not repeat the fact (user, 2026-11-21)
        "Nachfragen wiederholen den Fakt nicht",
        "Eine Frage, die einen schon erzählten Fakt als Prämisse nennt, zählt als weiterer "
        "Redebeitrag für diesen Fakt.",
        "Überleitungsfragen zum nächsten Thema bauen nicht auf Fakten auf",
        # (j) no inferences or contrasts the fact sentence lacks (2026-11-21)
        "Schlussfolgerungen sind keine Reaktion",
        "Lena stellt keine Vergleiche oder Gegensätze her",
        "Das Statement eines Redebeitrags gibt den Fakt-Satz wieder",
        # (k) references, answers and open questions (issues #8, #15)
        "Nennt ein Fakt-Satz seinen Gegenstand oder eine beteiligte Person nur mit einem "
        "Bezugswort",
        "Dieser Name ist die einzige Ausnahme von der Regel davor",
        "Beantwortet Lena eine Frage, handelt der zitierte Fakt-Satz vom Gegenstand der Frage.",
        "Seine Fragen legen keine Antwort nahe",
        "„War die Kritik vielleicht sogar ziemlich scharf?“",
        # (h) Felix may use the definition sentence as a premise more often (user, 2026-11-20)
        "Den ersten Einordnungssatz darf Felix öfter als Prämisse einer Frage verwenden; erzählt "
        "wird er nur einmal.",
    ],
)
def test_author_prompts_contain_the_fidelity_rules(prompt, rule):
    assert rule in prompt


# ---------------------------------------------------------------- continuation


PREFIX = [
    turn(0, "Felix", "Was ist das?", (), ()),
    turn(1, "Lena", "Das ist ein Fakt.", ["1-0000"], [("Ein Fakt.", ["1-0000"])]),
    turn(2, "Felix", "Und dann?", ["1-0001"], [("Davor.", ["1-0001"])]),
]


def test_told_in_lists_every_statement_fact_of_the_prefix():
    assert told_in(PREFIX) == {"1-0000": [1], "1-0001": [2]}
    assert told_in(PREFIX, (2,)) == {"1-0000": [1]}


def test_prefix_is_shown_with_aliases_told_facts_counts_and_the_start():
    text = format_prefix(PREFIX, ALIASES, ALIASES.values())
    assert text.startswith("Bisheriges Gespräch (Redebeitrag 0 bis 2, bleibt unverändert):")
    assert "Redebeitrag 1 (Lena):\nDas ist ein Fakt.\nKennungen: F1" in text
    assert "- F1: Redebeitrag 1\n- F2: Redebeitrag 2" in text
    assert "Das bisherige Gespräch hat 9 Wörter und verwendet 2 Fakten." in text
    assert "ab Redebeitrag 3 bis zum Ende" in text and "an Redebeitrag 2 (Felix) an" in text
    assert "1-000" not in text


def test_continue_dialog_uses_the_dialog_schema_and_reindexes_from_k():
    answer = {
        "turns": [
            {"speaker": "Lena", "text": "Weiter.", "fact_ids": ["F3"], "statements": []},
            {"speaker": "Felix", "text": "Schluss.", "fact_ids": ["F1"], "statements": []},
        ]
    }
    llm = FakeLlm(answer)
    turns = continue_dialog(llm, "[F1] Fakt.", ALIASES, PREFIX, 10)
    assert [(t.index, t.speaker, t.fact_ids) for t in turns] == [
        (3, "Lena", ["2-0000"]),
        (4, "Felix", ["1-0000"]),
    ]
    system, user, schema = llm.calls[0]
    assert schema is DIALOG_SCHEMA
    assert system == CONTINUE_PROMPT.format(min_used=10)
    assert user.startswith("Angebot:\n[F1] Fakt.\n\nBisheriges Gespräch")


def test_continue_prompt_is_german_and_asks_for_a_seamless_continuation():
    for phrase in (
        "nahtlos an den letzten bisherigen Redebeitrag an",
        "keine Begrüßung",
        "nichts wiederholen",
        "erzählst du nicht noch einmal",
        "mindestens 10 Fakten",
        "endet mit der einen Zusammenfassung",
    ):
        assert phrase in CONTINUE_PROMPT.format(min_used=10)


def test_author_reason_against_an_earlier_offer_marks_excluded_facts():
    # the continuation's offer lacks 1-0001 (excluded by D6), its aliases are rebuilt
    aliases = {"F1": "1-0000", "F2": "2-0000"}
    old = turn(0, ids=["1-0001"])
    reason = "Fakt 1-0001 trägt das nicht, 1-0000 schon."
    assert author_reason(old, reason, aliases, OFFER) == (
        f"Fakt {EXCLUDED_FACT} trägt das nicht, F1 schon."
    )


def test_author_reason_against_an_earlier_offer_keeps_the_unknown_id_problem_german():
    aliases = {"F1": "1-0000", "F2": "2-0000"}
    old = turn(0, ids=["1-0001", "9-9999"])
    archived = " ".join(p.reason for p in check_turns([old], OFFER))
    out = author_reason(old, archived, aliases, OFFER)
    assert out.startswith("Die Kennung 9-9999 gibt es im Angebot nicht.")
    assert "(F1–F2)" in out and "Unknown" not in out


def test_rejections_name_the_last_turn_list_earlier_ones_and_ask_to_avoid_them():
    assert format_rejections([]) == ""
    assert format_rejections([(2, "Zahl falsch.")]).split("\n") == [
        "Grund, warum der bisherige Redebeitrag 2 verworfen wurde:",
        "Zahl falsch.",
        "",
        "Vermeide diesen Fehler in der Fortsetzung.",
    ]
    out = format_rejections([(2, "Erst."), (2, "Erst."), (3, "Dann."), (3, "Dann.")])
    assert out.split("\n") == [
        "Grund, warum der bisherige Redebeitrag 3 verworfen wurde:",
        "Dann.",
        "",
        "Davor in derselben Kette verworfen:",
        "- Redebeitrag 2: Erst.",
        "",
        "Vermeide diese Fehler in der Fortsetzung.",
    ]


def test_continue_dialog_gives_the_author_the_rejection_reason_after_the_prefix():
    llm = FakeLlm({"turns": []})
    continue_dialog(llm, "[F1] Fakt.", ALIASES, PREFIX, 10, [(3, "Der Fakt F2 fehlt.")])
    _, user, _ = llm.calls[0]
    assert user.endswith(
        "Grund, warum der bisherige Redebeitrag 3 verworfen wurde:\nDer Fakt F2 fehlt.\n\n"
        "Vermeide diesen Fehler in der Fortsetzung."
    )
    continue_dialog(llm, "[F1] Fakt.", ALIASES, PREFIX, 10)
    assert "verworfen" not in llm.calls[1][1]


# ---------------------------------------------------------------- premise exemption

PINNED = frozenset({"1-0000"})


def test_felix_citing_the_pinned_fact_in_many_turns_is_no_repetition():
    turns = [
        told(0, "1-0000"),
        *(told(i, "1-0000", speaker="Felix") for i in range(1, 6)),
        told(6, "1-0001"),
    ]
    assert repetition_problems(turns, OFFER, 2, PINNED) == []
    assert len(repetition_problems(turns, OFFER, 2)) == 4  # without the exemption


def test_lena_retelling_the_pinned_fact_a_third_time_is_still_a_repetition():
    turns = [
        told(0, "1-0000"),
        told(1, "1-0000", speaker="Felix"),
        told(2, "1-0000"),
        told(3, "1-0000", speaker="Felix"),
        told(4, "1-0000"),
        told(5, "1-0001"),
    ]
    (problem,) = repetition_problems(turns, OFFER, 2, PINNED)
    assert problem.turn_index == 4
    assert problem.reason == repetition_reason("1-0000", [0, 2])


def test_non_pinned_fact_cited_by_felix_counts_as_before():
    turns = [told(0, "1-0001"), *(told(i, "1-0001", speaker="Felix") for i in (1, 2)), told(3)]
    (problem,) = repetition_problems(turns, OFFER, 2, PINNED)
    assert problem.turn_index == 2


def test_write_dialog_exempts_felix_premises_on_the_pinned_fact():
    def t(speaker: str, fid: str) -> dict:
        return {
            "speaker": speaker,
            "text": "A",
            "fact_ids": [fid],
            "statements": [{"text": "S", "fact_ids": [fid]}],
        }

    answer = {"turns": [t("Lena", "F1"), t("Felix", "F1"), t("Felix", "F1"), t("Lena", "F2")]}
    main, linked = [rf("1-0000")], [rf("2-0000", "N")]
    assert write_dialog(FakeLlm(answer), main, linked, 1, 2, (), PINNED).problems == []
    assert [p.turn_index for p in write_dialog(FakeLlm(answer), main, linked, 1, 2).problems] == [2]


def test_told_listing_marks_the_pinned_fact_and_leaves_out_felix_premises():
    turns = [
        told(0, "1-0000"),
        told(1, "1-0000", speaker="Felix"),
        told(2, "1-0001", speaker="Felix"),
        turn(3, "Felix", "Neu."),
        told(4, "1-0000"),
    ]
    assert told_elsewhere(turns, 3, PINNED) == {"1-0000": [0], "1-0001": [2]}
    assert told_in(turns, premise_exempt_ids=PINNED) == {"1-0000": [0, 4], "1-0001": [2]}
    out = format_turn_context(turns, 3, "Grund.", ALIASES, PINNED)
    assert (
        "- F1: Redebeitrag 0 (Felix darf ihn erneut als Prämisse einer Frage verwenden)\n"
        "- F2: Redebeitrag 2"
    ) in out
    assert "Prämisse" not in format_turn_context(turns, 3, "Grund.", ALIASES)


def test_continuation_prefix_marks_the_pinned_fact():
    text = format_prefix(PREFIX, ALIASES, ALIASES.values(), PINNED)
    assert "- F1: Redebeitrag 1 (Felix darf ihn erneut als Prämisse einer Frage verwenden)" in text
    llm = FakeLlm({"turns": []})
    continue_dialog(llm, "[F1] Fakt.", ALIASES, PREFIX, 10, (), PINNED)
    assert "(Felix darf ihn erneut als Prämisse" in llm.calls[0][1]
