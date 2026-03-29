import logging
import os
import sys
from logging.handlers import RotatingFileHandler


def setup_logging(log_dir: str = "logs", log_name: str = "pipeline.log", level: int = logging.INFO) -> None:
    """Configures the root logger with console and file handlers.

    Args:
        log_dir: Directory to save log files.
        log_name: Name of the log file.
        level: Logging threshold (INFO, DEBUG, WARNING).
    """
    # 1. Create Log Directory
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)

    # 2. Define Formatters
    # File gets detailed info: Time, Module, Level, Message
    file_formatter = logging.Formatter("%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
    # Console gets simple info: Level, Message
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")

    # 3. Get Root Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to prevent duplicate logs if function called twice
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 4. File Handler (Rotating)
    # Keeps 5 files of 10MB each. Prevents logs from eating the hard drive.
    file_handler = RotatingFileHandler(log_path, maxBytes=10 * 1024 * 1024, backupCount=5)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # 5. Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # 6. Silence Noisy Third-Party Libraries
    # These libraries spam DEBUG logs that we don't care about
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("fiona").setLevel(logging.WARNING)
    logging.getLogger("pyproj").setLevel(logging.WARNING)

    logging.info(f"Logging initialized. Writing to {log_path}")
