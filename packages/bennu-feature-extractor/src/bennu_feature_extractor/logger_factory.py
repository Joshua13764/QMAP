from __future__ import annotations
import logging
import re
from pathlib import Path
from datetime import date
from rich.console import Console
from rich.logging import RichHandler


class FlushFileHandler(logging.FileHandler):
    """FileHandler that flushes after every write (important for parallel jobs)."""

    def emit(self, record):
        super().emit(record)
        self.flush()


def _next_log_file(log_dir: Path, name: str) -> Path:
    """
    Return a Path for a new log file with format: {name}-{YYYYMMDD}-{index}.log
    Index starts at 0 and increments based on existing files for the same date.
    """
    today_str = date.today().strftime("%Y%m%d")
    prefix = f"{name}-{today_str}-"
    pattern = re.compile(rf"^{re.escape(name)}-{today_str}-(\d+)\.log$")
    max_index = -1
    for p in log_dir.iterdir():
        if not p.is_file():
            continue
        m = pattern.match(p.name)
        if m:
            try:
                idx = int(m.group(1))
            except ValueError:
                continue
            if idx > max_index:
                max_index = idx
    next_index = max_index + 1
    filename = f"{name}-{today_str}-{next_index}.log"
    return log_dir / filename


def get_logger(
    name: str,
    log_dir: Path,
    level: int = logging.INFO,
    to_console: bool = True,
    to_file: bool = True,
) -> logging.Logger:
    """
    Create and configure a rich, colorized, multi-handler logger.
    Removes redundant [LEVEL] markup in console output.

    Writes logs to a new file each run named: {name}-{YYYYMMDD}-{index}.log
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # --- File Handler ---
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(processName)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_file = None
    if to_file:
        log_file = _next_log_file(log_dir, name)
        file_handler = FlushFileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # --- Console Handler ---
    if to_console:
        console = Console(force_terminal=True)
        # Let RichHandler handle color and formatting (no duplicate tags)
        console_handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            show_time=True,
            show_level=True,
            show_path=False,
            markup=True,
            log_time_format="%Y-%m-%d %H:%M:%S",
        )
        # Simpler formatter: RichHandler will handle the rest
        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    logger.propagate = False
    logger.info(
        f"[cyan]Logger initialized → {log_file if to_file else 'console only'}[/]"
    )
    return logger
