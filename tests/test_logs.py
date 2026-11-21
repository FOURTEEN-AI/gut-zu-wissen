import logging

import pytest

from gut_zu_wissen.logs import PACKAGE, setup_logging, timed


@pytest.fixture(autouse=True)
def _clean():
    logger = logging.getLogger(PACKAGE)
    yield
    logger.handlers.clear()
    logger.setLevel(logging.NOTSET)
    logger.propagate = True


def test_setup_is_idempotent_and_reads_env(monkeypatch, capsys):
    monkeypatch.setenv("LOG_LEVEL", "debug")
    setup_logging()
    setup_logging()
    logger = logging.getLogger(PACKAGE)
    assert len(logger.handlers) == 1
    logging.getLogger(f"{PACKAGE}.x").debug("hello")
    assert "hello" in capsys.readouterr().out


def test_unknown_level_falls_back_to_info(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "nonsense")
    setup_logging()
    assert logging.getLogger(PACKAGE).level == logging.INFO


def test_timed_logs_start_end_and_failure(caplog):
    log = logging.getLogger(f"{PACKAGE}.t")
    with caplog.at_level(logging.INFO, logger=PACKAGE), timed(log, "work"):
        pass
    assert [r.getMessage()[:8] for r in caplog.records] == ["work …", "work don"]
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=PACKAGE), pytest.raises(ValueError), timed(log, "w"):
        raise ValueError
    assert "failed" in caplog.records[-1].getMessage()


def test_modules_started_with_dash_m_name_their_logger_explicitly():
    """Run as `python -m`, `__name__` is "__main__" and the package handler would miss it."""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / PACKAGE
    offenders = [
        path.name
        for path in src.glob("*.py")
        if '__name__ == "__main__"' in path.read_text()
        and "getLogger(__name__)" in path.read_text()
        and path.name != "logs.py"
    ]
    assert offenders == []
    for path in src.glob("*.py"):  # explicit names stay inside the package
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and ast.unparse(node.func) == "logging.getLogger"
                and isinstance(node.args[0], ast.Constant)
                and path.name != "logs.py"
            ):
                assert node.args[0].value.startswith(f"{PACKAGE}."), path.name
