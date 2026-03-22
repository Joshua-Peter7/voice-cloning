"""
core/logger.py
──────────────
Structured logging with loguru.
Outputs JSON in production, colored text in development.
"""

from __future__ import annotations

import sys
from typing import Optional

from loguru import logger as _logger


def setup_logger(
    level: str = "INFO",
    env: str = "production",
    log_dir: Optional[str] = None,
) -> None:
    """
    Configure loguru for the application.

    Production  → JSON to stdout + rotating file
    Development → colored text to stdout
    """
    _logger.remove()

    if env == "development":
        _logger.add(
            sys.stdout,
            level=level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            colorize=True,
        )
    else:
        # JSON structured logging for production log aggregators
        _logger.add(
            sys.stdout,
            level=level,
            serialize=True,   # JSON output
        )

    # Rotating file log regardless of environment
    if log_dir:
        _logger.add(
            f"{log_dir}/ivc_{{time}}.log",
            level=level,
            rotation="100 MB",
            retention="7 days",
            compression="gz",
            serialize=True,
        )


def get_logger(name: str):
    """Return a logger bound with a component name."""
    return _logger.bind(component=name)
