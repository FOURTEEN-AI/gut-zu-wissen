"""Workflow files against the config: the review workflow's gate must name the configured
reviewer, and no untrusted event text may reach an expression."""

import json
import re
import subprocess

from gut_zu_wissen.resume import EPISODE_ID_PATTERN, valid_episode_id
from gut_zu_wissen.run import REPO_ROOT, load_config

WORKFLOWS = REPO_ROOT / ".github" / "workflows"
REVIEW = (WORKFLOWS / "review.yml").read_text(encoding="utf-8")
EPISODE = (WORKFLOWS / "episode.yml").read_text(encoding="utf-8")
CONFIG = load_config()["review"]


def test_review_gate_names_the_configured_reviewer_only():
    logins = set(re.findall(r"login == '([^']+)'", REVIEW))
    logins |= set(re.findall(r"github\.actor == '([^']+)'", REVIEW))
    assert logins == {CONFIG["reviewer"]}


def test_review_gate_lists_the_configured_associations():
    (listed,) = re.findall(r"fromJSON\('(\[[^']*\])'\)", REVIEW)
    assert json.loads(listed) == CONFIG["reviewer_associations"]


def test_no_comment_or_issue_text_in_expressions():
    for text in (REVIEW, EPISODE, RESUME):
        expressions = re.findall(r"\$\{\{(.*?)\}\}", text)
        for expression in expressions:
            assert "comment.body" not in expression and "issue.title" not in expression
            assert "issue.body" not in expression


def test_review_workflow_permissions_are_minimal():
    assert "permissions:\n  contents: write\n  issues: write\n\n" in REVIEW
    assert "issues: write" in EPISODE


def _step(text: str, name: str) -> str:
    start = text.index(f"- name: {name}\n")
    end = text.find("\n      - ", start + 1)
    return text[start : end if end != -1 else len(text)]


def test_episode_archive_is_committed_on_failure_and_review_issue_only_on_success():
    assert "if: always()" in _step(EPISODE, "Commit archive")
    # the run step sets episode_id only after a successful run
    assert "if: steps.run.outputs.episode_id != ''" in _step(EPISODE, "Open review issue")


def test_episode_job_has_a_timeout():
    assert "  episode:\n    runs-on: ubuntu-latest\n    timeout-minutes: 90\n" in EPISODE


# ---------------------------------------------------------------- resume-audio.yml

RESUME = (WORKFLOWS / "resume-audio.yml").read_text(encoding="utf-8")
STEP_INDENT = "\n      - "
EPISODE_ONLY = ("Run pipeline",)
RESUME_ONLY = ("Check episode ID", "Resume audio step")


def _steps(text: str) -> list[str]:
    """Every step of the single job, comment lines dropped (comments may differ)."""
    body = text[text.index("    steps:\n") :]
    steps = body.split(STEP_INDENT)[1:]
    return [
        "\n".join(line for line in step.splitlines() if not line.lstrip().startswith("#")).strip()
        for step in steps
    ]


def _named(steps: list[str], names: tuple[str, ...]) -> list[str]:
    return [s for s in steps if not any(s.startswith(f"name: {n}\n") for n in names)]


def _block(text: str, key: str) -> str:
    start = text.index(f"\n{key}:\n")
    return text[start : text.index("\n\n", start + 1)]


def test_resume_shares_every_other_step_with_the_episode_workflow():
    shared = _named(_steps(EPISODE), EPISODE_ONLY)
    assert _named(_steps(RESUME), RESUME_ONLY) == shared
    names = [s.splitlines()[0] for s in shared]
    for name in ("System packages", "Restore Whisper model cache", "Upload audio",
                 "Open review issue", "Commit archive"):  # fmt: skip
        assert f"name: {name}" in names


def test_resume_has_the_episode_concurrency_permissions_and_timeout():
    for key in ("concurrency", "permissions"):
        assert _block(RESUME, key) == _block(EPISODE, key)
    assert "group: episode\n  cancel-in-progress: false" in RESUME
    assert "  resume:\n    runs-on: ubuntu-latest\n    timeout-minutes: 90\n" in RESUME


def test_resume_step_gets_the_secrets_of_the_episode_run():
    def secrets(text: str) -> list[str]:
        return re.findall(r"\$\{\{ secrets\.(\w+) \}\}", text)

    assert secrets(_step(RESUME, "Resume audio step")) == secrets(_step(EPISODE, "Run pipeline"))
    assert 'uv run python -m gut_zu_wissen.resume "$EPISODE_ID"' in RESUME


def test_resume_input_reaches_the_shell_only_through_the_environment():
    uses = [m.start() for m in re.finditer(r"inputs\.episode_id", RESUME)]
    assert len(uses) == len(re.findall(r"EPISODE_ID: \$\{\{ inputs\.episode_id \}\}", RESUME)) == 2
    for expression in re.findall(r"\$\{\{(.*?)\}\}", RESUME):
        assert "inputs." not in expression or expression.strip() == "inputs.episode_id"
    run_lines = [line for line in RESUME.splitlines() if "${{" in line and "run:" in line]
    assert run_lines == []


def test_resume_workflow_checks_the_input_against_the_episode_id_pattern():
    (pattern,) = re.findall(r"pattern='([^']+)'", RESUME)
    assert pattern == EPISODE_ID_PATTERN
    assert RESUME.index("name: Check episode ID") < RESUME.index("name: System packages")
    script = _step(RESUME, "Check episode ID").split("run: |\n", 1)[1]
    script = "\n".join(line.strip() for line in script.splitlines())
    for value, ok in [
        ("2026-11-20_tunguska-ereignis", True),
        ("2026-11-20_tunguska-ereignis-2", True),
        ("2026-11-20_x; rm -rf /", False),
        ("$(id)", False),
        ("../state", False),
    ]:
        result = subprocess.run(
            ["bash", "-c", script], env={"EPISODE_ID": value}, capture_output=True, check=False
        )
        assert (result.returncode == 0) is ok, value
        assert valid_episode_id(value) is ok
