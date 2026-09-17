"""Root logging configuration.

Modules obtain their own logger with ``logging.getLogger(__name__)`` and
never install handlers; :func:`configure_logging` is called once from
``main.py``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-32s | %(message)s"
DATE_FORMAT = "%H:%M:%S"
NOISY_LOGGERS = ("matplotlib", "PIL", "tensorflow", "absl", "h5py", "yfinance")


def configure_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> None:
    """Configure the root logger for the whole application.

    Repeated calls replace the installed handlers, so re-running from a
    notebook does not duplicate lines.

    Args:
        level: Minimum severity emitted.
        log_file: Optional extra plain-text log file; parents are created.

    Raises:
        OSError: If ``log_file`` cannot be opened for writing.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            handlers.append(
                logging.FileHandler(log_file, mode="w", encoding="utf-8")
            )
        except OSError as exc:
            raise OSError(f"Cannot open log file {log_file}: {exc}") from exc

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(level)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
