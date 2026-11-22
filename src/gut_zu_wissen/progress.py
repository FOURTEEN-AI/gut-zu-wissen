"""Live progress of a running pipeline for the status page (user decision 2026-11-22).

The archive is committed only when a run ends; the status page ("Was gerade entsteht",
`status.py`) learns the current step from `live.json` on its own branch (`[status] branch`). At
every step the run replaces that branch with a single commit holding only the live file (no
history, nothing on `main`) and starts the Pages build: a push with the workflow token starts no
other workflow, a dispatch does.

Progress is best effort: a failed push or dispatch is logged as a warning and the run goes on.
The live file holds the episode ID, the article title, the step and two times; nothing else.
`GitProgress` is used only when the workflow sets `LIVE_STATUS=true`; tests use fakes.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Protocol

from gut_zu_wissen.archive import EpisodeRecord, format_timestamp
from gut_zu_wissen.status import Live

log = logging.getLogger(__name__)

LIVE_ENV = "LIVE_STATUS"
BOT_NAME = "github-actions[bot]"
BOT_EMAIL = "41898282+github-actions[bot]@users.noreply.github.com"

Runner = Callable[[Sequence[str], str | None], str]


class Progress(Protocol):
    def step(self, record: EpisodeRecord, step: str, now: datetime) -> None:
        """The run of `record` enters `step` (`archive.STEPS`) at `now`."""
        ...


class NoProgress:
    def step(self, record: EpisodeRecord, step: str, now: datetime) -> None:
        return None


def live_for(record: EpisodeRecord, step: str, now: datetime) -> Live:
    """The live file's content (pure)."""
    return Live(record.episode_id, record.title, step, record.started_at, format_timestamp(now))


def live_json(live: Live) -> str:
    return json.dumps(live.to_dict(), ensure_ascii=False, indent=2) + "\n"


def git_runner(cwd: Path) -> Runner:
    """Runs git in `cwd` with the bot as author and committer; returns stdout, stripped."""
    env_identity = {
        "GIT_AUTHOR_NAME": BOT_NAME,
        "GIT_AUTHOR_EMAIL": BOT_EMAIL,
        "GIT_COMMITTER_NAME": BOT_NAME,
        "GIT_COMMITTER_EMAIL": BOT_EMAIL,
    }

    def run(args: Sequence[str], stdin: str | None) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            input=stdin,
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, **env_identity},
        )
        return result.stdout.strip()

    return run


class GitProgress:
    """Replaces `branch` with one commit holding `live_file`, then calls `rebuild`."""

    def __init__(
        self, git: Runner, branch: str, live_file: str, rebuild: Callable[[], None]
    ) -> None:
        self._git, self._branch, self._file, self._rebuild = git, branch, live_file, rebuild

    def step(self, record: EpisodeRecord, step: str, now: datetime) -> None:
        live = live_for(record, step, now)
        try:
            blob = self._git(["hash-object", "-w", "--stdin"], live_json(live))
            tree = self._git(["mktree"], f"100644 blob {blob}\t{self._file}\n")
            message = f"Live status: {record.episode_id} {step}"
            commit = self._git(["commit-tree", tree, "-m", message], None)
            self._git(["push", "--force", "origin", f"{commit}:refs/heads/{self._branch}"], None)
            self._rebuild()
        except Exception as exc:  # best effort: the run must not fail on its status page
            detail = getattr(exc, "stderr", "") or exc
            log.warning("Live status for step %s not published: %s", step, detail)
        else:
            log.info("Live status published: step %s", step)


def from_env(status_config: dict, repo_root: Path) -> Progress:
    """`GitProgress` in a workflow that sets `LIVE_STATUS=true` (with `GITHUB_REPOSITORY` and
    `GITHUB_TOKEN`), otherwise `NoProgress`."""
    from gut_zu_wissen.github import HttpWorkflows

    repository, token = os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_TOKEN")
    if os.environ.get(LIVE_ENV) != "true" or not repository or not token:
        return NoProgress()
    workflows = HttpWorkflows(repository, token)
    return GitProgress(
        git_runner(repo_root),
        status_config["branch"],
        status_config["live_file"],
        lambda: workflows.dispatch(status_config["site_workflow"], status_config["site_ref"]),
    )
