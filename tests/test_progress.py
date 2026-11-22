"""Live progress: the status branch holds one commit with the live file; failures never stop
the run. Git runs against a local bare repository, the dispatch against a mock transport."""

import json
import logging
import subprocess
from datetime import UTC, datetime

import httpx

from gut_zu_wissen.archive import EpisodeRecord
from gut_zu_wissen.github import HttpWorkflows
from gut_zu_wissen.progress import GitProgress, NoProgress, from_env, git_runner, live_for
from gut_zu_wissen.run import load_config
from gut_zu_wissen.status import Live

STATUS = load_config()["status"]
NOW = datetime(2026, 11, 22, 11, 20, tzinfo=UTC)
RECORD = EpisodeRecord("2026-11-22_koelner-dom", "Kölner Dom", "running", "2026-11-22T11:00:00Z")


def git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def clone_with_remote(tmp_path):
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "-q", "--bare", str(remote))
    work = tmp_path / "work"
    git(tmp_path, "init", "-q", str(work))
    git(work, "remote", "add", "origin", str(remote))
    return work, remote


def test_live_file_content():
    live = live_for(RECORD, "dialog", NOW)
    assert live == Live(
        "2026-11-22_koelner-dom",
        "Kölner Dom",
        "dialog",
        "2026-11-22T11:00:00Z",
        "2026-11-22T11:20:00Z",
    )
    assert list(live.to_dict()) == [
        "episode_id",
        "title",
        "step",
        "started_at",
        "step_started_at",
    ]


def test_each_step_replaces_the_branch_with_one_commit_and_rebuilds(tmp_path):
    work, remote = clone_with_remote(tmp_path)
    rebuilds = []
    progress = GitProgress(
        git_runner(work), STATUS["branch"], STATUS["live_file"], lambda: rebuilds.append(1)
    )
    progress.step(RECORD, "extraction", NOW)
    progress.step(RECORD, "dialog", NOW)
    branch = STATUS["branch"]
    assert git(remote, "rev-list", "--count", branch) == "1"
    assert git(remote, "ls-tree", "--name-only", branch) == STATUS["live_file"]
    live = json.loads(git(remote, "show", f"{branch}:{STATUS['live_file']}"))
    assert live["step"] == "dialog" and live["title"] == "Kölner Dom"
    assert git(remote, "log", "-1", "--format=%an", branch) == "github-actions[bot]"
    assert rebuilds == [1, 1]


def test_a_failed_push_is_a_warning_and_skips_the_rebuild(tmp_path, caplog):
    work = tmp_path / "work"
    git(tmp_path, "init", "-q", str(work))  # no remote
    rebuilds = []
    progress = GitProgress(git_runner(work), "status", "live.json", lambda: rebuilds.append(1))
    with caplog.at_level(logging.WARNING):
        progress.step(RECORD, "dialog", NOW)
    assert rebuilds == []
    assert "Live status for step dialog not published" in caplog.text


def test_a_failed_rebuild_is_a_warning(tmp_path, caplog):
    work, _ = clone_with_remote(tmp_path)

    def broken():
        raise RuntimeError("dispatch refused")

    with caplog.at_level(logging.WARNING):
        GitProgress(git_runner(work), "status", "live.json", broken).step(RECORD, "audio", NOW)
    assert "dispatch refused" in caplog.text


def test_from_env_needs_the_switch_and_the_token(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    monkeypatch.delenv("LIVE_STATUS", raising=False)
    assert isinstance(from_env(STATUS, tmp_path), NoProgress)
    monkeypatch.setenv("LIVE_STATUS", "true")
    assert isinstance(from_env(STATUS, tmp_path), GitProgress)
    monkeypatch.delenv("GITHUB_TOKEN")
    assert isinstance(from_env(STATUS, tmp_path), NoProgress)


def test_dispatch_starts_the_workflow_on_the_ref():
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(204)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    HttpWorkflows("o/r", "secret", "https://api.example", client).dispatch("pages.yml", "main")
    (request,) = sent
    assert request.method == "POST"
    assert (
        str(request.url) == "https://api.example/repos/o/r/actions/workflows/pages.yml/dispatches"
    )
    assert json.loads(request.content) == {"ref": "main"}
    assert request.headers["Authorization"] == "Bearer secret"
