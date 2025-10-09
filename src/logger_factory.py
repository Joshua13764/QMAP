from __future__ import annotations
import logging
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler


class FlushFileHandler(logging.FileHandler):
    """FileHandler that flushes after every write (important for parallel jobs)."""
    def emit(self, record):
        super().emit(record)
        self.flush()


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
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # --- File Handler ---
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(processName)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if to_file:
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
    logger.info(f"[cyan]Logger initialized → {log_file if to_file else 'console only'}[/]")
    return logger
