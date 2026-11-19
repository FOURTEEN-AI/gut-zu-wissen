"""Workflow files against the config: the review workflow's gate must name the configured
reviewer, and no untrusted event text may reach an expression."""

import json
import re

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
    for text in (REVIEW, EPISODE):
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
