"""Loguru-based logging with file and optional console sinks."""

import os
from datetime import datetime
from loguru import logger

logger.remove()

timestamp = datetime.now().strftime("%Y%m%d_%H")
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "log", timestamp)
os.makedirs(log_dir, exist_ok=True)

logger.add(
    os.path.join(log_dir, "all.log"),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="500 MB",
    retention="30 days",
    encoding="utf-8",
)

logger.add(
    os.path.join(log_dir, "error.log"),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="ERROR",
    rotation="500 MB",
    retention="30 days",
    encoding="utf-8",
)

# NOTE: We intentionally do NOT add a console/terminal sink here.
# Log output competes with rich Live/Panel rendering and causes ghosting
# when the terminal is resized or scrolled.  All logs go to files only;
# runtime progress and generated text are shown via rich in engine/progress.py.

__all__ = ["logger"]
