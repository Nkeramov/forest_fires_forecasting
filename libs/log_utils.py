import logging
import logging.handlers
from typing import Any
from pathlib import Path
from colorama import Fore, Style


from .cls_utils import Singleton


class CustomColoredFormatter(logging.Formatter):
    """Custom logging colored formatter with configurable colors"""
    LEVEL_COLORS = {
        logging.DEBUG: Fore.LIGHTBLUE_EX,
        logging.INFO: Fore.LIGHTGREEN_EX,
        logging.WARNING: Fore.LIGHTYELLOW_EX,
        logging.ERROR: Fore.LIGHTRED_EX,
        logging.CRITICAL: Fore.LIGHTRED_EX + Style.BRIGHT,
    }

    def __init__(
            self,
            fmt: str | None = None,
            datefmt: str | None = None,
            colors: dict[str, str] | None = None
    ):
        """
        Initialize formatter with optional custom formats and colors

        Args:
            fmt: Log message format
            datefmt: Date format
            colors: Optional dict mapping log levels to color names
        """
        self.fmt = fmt
        self.datefmt = datefmt
        super().__init__(fmt=fmt, datefmt=datefmt)

        # Update level colors if custom colors provided
        if colors:
            self.LEVEL_COLORS.update(
                {getattr(logging, k.upper()): v for k, v in colors.items()
                 if k.upper() in ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')}
            )

    def format(self, record: logging.LogRecord) -> str:
        """Format the specified record with color"""
        log_fmt = self.fmt
        if  record.levelno in self.LEVEL_COLORS:
            log_fmt = f"{self.LEVEL_COLORS[record.levelno]}{self.fmt}{Style.RESET_ALL}"
        formatter = logging.Formatter(fmt=log_fmt, datefmt=self.datefmt)
        return formatter.format(record)

class LoggerSingleton(metaclass=Singleton):
    """Thread-safe singleton logger with file and stream handlers"""
    __logger: logging.Logger = logging.getLogger('SuperLogger')
    __allow_reinitialization: bool = False

    DEFAULT_FORMAT = '%(asctime)s | %(levelname)s | %(module)s | %(funcName)s | %(lineno)s | %(message)s'
    DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

    def __init__(self,
                 log_dir: str | None = None, log_file: str | None = None, level: str | None = None,
                 msg_format: str | None = None, date_format: str | None = None,
                 colored: bool = False, max_size_mb: int = 10, keep: int = 10, **kwargs: Any):
        if not hasattr(self, '_initialized') or self.__allow_reinitialization:
            self._initialize_logger(log_dir=log_dir, log_file=log_file, level=level or "INFO",
                                    msg_format=msg_format or self.DEFAULT_FORMAT, 
                                    date_format = date_format or self.DEFAULT_DATE_FORMAT,
                                    colored=colored, max_size_mb=max_size_mb, keep=keep, **kwargs)
            self._initialized = True

    def _initialize_logger(self, log_dir: str | None, log_file: str | None, level: str, msg_format: str,
                           date_format: str, colored: bool, max_size_mb: int, keep: int, **kwargs: Any) -> None:
        """Initialize logger with configured handlers"""
        self.__class__.__logger.setLevel(level)
        # Clear existing handlers to avoid duplicates
        self.__class__.__logger.handlers.clear()
        # Add stream handler
        self._add_stream_handler(level, msg_format, date_format, colored, **kwargs)
        # Add file handler if configured
        if log_dir and log_file:
            self._add_file_handler(
                log_dir, log_file, level, msg_format, date_format, max_size_mb, keep
            )

    def _add_stream_handler(self, level: str, msg_format: str, date_format: str, colored: bool, **kwargs: Any) -> None:
        """Add and configure stream handler"""
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        formatter = (
            CustomColoredFormatter(fmt=msg_format, datefmt=date_format, **kwargs)
            if colored
            else logging.Formatter(fmt=msg_format, datefmt=date_format)
        )
        stream_handler.setFormatter(formatter)
        self.__class__.__logger.addHandler(stream_handler)

    def _add_file_handler(self, log_dir: str, log_file: str, level: str, msg_format: str, date_format: str,
                          max_size_mb: int, keep: int) -> None:
        """Add and configure file handler with rotation"""
        try:
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)
            file_path = log_dir_path / log_file
            file_handler = (logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=max_size_mb * 1024 * 1024,
                backupCount=keep,
                encoding="utf-8"
            ))
            file_handler.setLevel(level)
            file_handler.setFormatter(
                logging.Formatter(fmt=msg_format, datefmt=date_format)
            )
            LoggerSingleton.__logger.addHandler(file_handler)
        except (OSError, IOError) as e:
            self.__logger.error(f"Failed to initialize file handler: {e}", exc_info=True)
            raise

    @classmethod
    def get_logger(cls) -> logging.Logger:
        """Get the logger instance, initializing with defaults if not already initialized"""
        if cls.__logger is None:
            cls()
        if not isinstance(cls.__logger, logging.Logger):
            raise RuntimeError("Logger was not properly initialized")
        return cls.__logger

    @classmethod
    def update_config(cls, **kwargs: Any) -> None:
        """Update logger configuration"""
        with cls._lock:
            if cls.__logger is not None:
                cls()._initialize_logger(**kwargs)
