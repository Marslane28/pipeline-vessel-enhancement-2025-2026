import sys
import logging
import colorlog
from logging import Logger
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logger(
    name: str = "default",
    log_dir: str | Path = "logs",
    debug_mode: bool = False,
    max_bytes: int = 1_000_000,
    backup_count: int = 3,
) -> Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False  

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=log_path / f"{name}.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    console_handler.setFormatter(colorlog.ColoredFormatter(
        fmt="%(log_color)s%(asctime)s [%(levelname)-8s]%(reset)s %(message)s",
        datefmt="%H:%M:%S",
        log_colors={
            "DEBUG":    "cyan",
            "INFO":     "green",
            "WARNING":  "yellow",
            "ERROR":    "red",
            "CRITICAL": "bold_red",
        },
    ))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def close_logger(logger: Logger) -> None:
    """Ferme proprement tous les handlers du logger."""
    for handler in logger.handlers[:]:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)