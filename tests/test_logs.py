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
