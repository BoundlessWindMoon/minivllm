import os
import sys
from datetime import datetime
from loguru import logger
from utils.progress import CONSOLE

logger.remove()

timestamp = datetime.now().strftime("%Y%m%d_%H")
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "log", timestamp)
os.makedirs(log_dir, exist_ok=True)


def rich_sink(message):

    CONSOLE.print(message, end="")


# logger.add(
#     rich_sink,
#     format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
#     "<level>{level: <8}</level> | "
#     "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
#     "<level>{message}</level>",
#     level="DEBUG",
#     colorize=True,
# )


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

__all__ = ["logger"]
