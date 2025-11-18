"""
Logging configuration and utilities.
"""
import logging
import sys
from typing import Optional
from pathlib import Path
from datetime import datetime
import json


class JSONFormatter(logging.Formatter):
    """
    Format log records as JSON for structured logging.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as JSON.

        Args:
            record: Log record to format

        Returns:
            str: JSON formatted log message
        """
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, 'extra'):
            log_data['extra'] = record.extra

        return json.dumps(log_data)


def setup_logger(
    name: str,
    level: str = 'INFO',
    log_file: Optional[str] = None,
    json_format: bool = False
) -> logging.Logger:
    """
    Set up a logger with consistent configuration.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        json_format: Whether to use JSON formatting

    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    if json_format:
        console_formatter = JSONFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))

        if json_format:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the given name.

    Args:
        name: Logger name

    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)


class LoggerContext:
    """
    Context manager for temporary logger configuration.

    Example:
        with LoggerContext('my_module', level='DEBUG'):
            # Code with DEBUG logging
            pass
    """

    def __init__(self, logger_name: str, level: str = 'DEBUG'):
        """
        Initialize logger context.

        Args:
            logger_name: Name of logger to modify
            level: Temporary logging level
        """
        self.logger = logging.getLogger(logger_name)
        self.original_level = self.logger.level
        self.new_level = getattr(logging, level.upper())

    def __enter__(self):
        """Enter context and set new level."""
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore original level."""
        self.logger.setLevel(self.original_level)


class ProgressLogger:
    """
    Logger for tracking progress of long-running operations.
    """

    def __init__(self, logger: logging.Logger, total: int, description: str = ''):
        """
        Initialize progress logger.

        Args:
            logger: Logger to use
            total: Total number of items
            description: Description of the operation
        """
        self.logger = logger
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = datetime.now()

    def update(self, increment: int = 1):
        """
        Update progress.

        Args:
            increment: Number of items completed
        """
        self.current += increment
        percentage = (self.current / self.total) * 100

        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / rate if rate > 0 else 0

        self.logger.info(
            f"{self.description}: {self.current}/{self.total} "
            f"({percentage:.1f}%) - {rate:.1f} items/sec - "
            f"ETA: {eta:.0f}s"
        )

    def complete(self):
        """Mark operation as complete."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(
            f"{self.description}: Completed {self.total} items "
            f"in {elapsed:.1f}s"
        )


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls with arguments and results.

    Args:
        logger: Logger to use

    Example:
        @log_function_call(logger)
        def my_function(arg1, arg2):
            return result
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(
                f"Calling {func.__name__} with args={args}, kwargs={kwargs}"
            )
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} returned: {result}")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} raised exception: {e}")
                raise
        return wrapper
    return decorator


def log_performance(logger: logging.Logger, threshold_seconds: float = 1.0):
    """
    Decorator to log slow function calls.

    Args:
        logger: Logger to use
        threshold_seconds: Log if execution takes longer than this

    Example:
        @log_performance(logger, threshold_seconds=0.5)
        def slow_function():
            # Long-running code
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            result = func(*args, **kwargs)
            elapsed = (datetime.now() - start_time).total_seconds()

            if elapsed > threshold_seconds:
                logger.warning(
                    f"{func.__name__} took {elapsed:.2f}s "
                    f"(threshold: {threshold_seconds}s)"
                )

            return result
        return wrapper
    return decorator


# Default logger for the package
default_logger = setup_logger('bio_match_platform', level='INFO')
