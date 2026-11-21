"""Progress logging of a pipeline run (stdlib `logging`).

Every module logs through `logging.getLogger(__name__)`, except the ones started with
`python -m` (their `__name__` is "__main__", outside the package logger): those name their
logger `gut_zu_wissen.<module>`. The entry points call `setup_logging()` once.

Output goes to stdout (same stream as the `print` lines, so the order in the Actions log holds),
one line per event, flushed at once. The level comes from `LOG_LEVEL` (default INFO;
DEBUG adds every LLM call). Only the package logger is configured, so SDK loggers stay quiet.

The log of a run is public (CLAUDE.md): log IDs, counts, durations and reasons, never secrets,
prompt text or listener data.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

PACKAGE = "gut_zu_wissen"
LEVEL_ENV = "LOG_LEVEL"
DEFAULT_LEVEL = "INFO"
FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: str | None = None) -> None:
    """Attach one stdout handler to the package logger; safe to call more than once."""
    name = (level or os.environ.get(LEVEL_ENV) or DEFAULT_LEVEL).upper()
    logger = logging.getLogger(PACKAGE)
    logger.setLevel(logging.getLevelName(name) if name in logging.getLevelNamesMapping() else 20)
    if any(getattr(h, "_gzw", False) for h in logger.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(FORMAT, DATE_FORMAT))
    handler._gzw = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.propagate = False


@contextmanager
def timed(log: logging.Logger, what: str) -> Iterator[None]:
    """Log `what` at start and, with the duration, at its end (a failure is logged and raised)."""
    log.info("%s …", what)
    start = time.monotonic()
    try:
        yield
    except BaseException:
        log.warning("%s failed after %.1f s", what, time.monotonic() - start)
        raise
    log.info("%s done in %.1f s", what, time.monotonic() - start)
