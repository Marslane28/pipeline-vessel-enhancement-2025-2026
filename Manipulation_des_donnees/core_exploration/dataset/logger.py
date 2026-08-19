import sys
from logging import Logger
import colorlog
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from configs.args import LOG_DIR


def setup_logger(log_file: str = "default", debug_mode: bool = False) -> Logger:
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    log_path = Path(LOG_DIR) / f"{log_file}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True) 

    # File Handler
    file_handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - [%(levelname)s]: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_level = logging.DEBUG if debug_mode else logging.WARNING
    console_handler.setLevel(console_level)
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - [%(levelname)s]:%(reset)s %(message)s',
        datefmt='%H:%M:%S',
        log_colors={
                'DEBUG':    'cyan',
                'INFO':     'green',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'bold_red',
            }
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger