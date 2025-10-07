# logger_factory.py
from __future__ import annotations

import logging
from pathlib import Path
from rich.logging import RichHandler


def get_logger(
    name: str,
    log_dir: Path,
    level: int = logging.INFO,
    to_console: bool = True,
    to_file: bool = True,
) -> logging.Logger:
    """
    Create and configure a rich, multi-handler logger.

    Parameters
    ----------
    name : str
        Logger name (usually __name__ or app module name)
    log_dir : Path
        Directory for log files (must be writable)
    level : int
        Logging level (default: logging.INFO)
    to_console : bool
        Whether to output logs to console using RichHandler
    to_file : bool
        Whether to output logs to a rotating file handler

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # Prevent duplicate handlers if reused

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if to_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if to_console:
        console_handler = RichHandler(rich_tracebacks=True, show_path=False)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.propagate = False
    logger.info(f"Logger initialized → {log_file if to_file else 'console only'}")

    return logger
