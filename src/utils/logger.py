import logging
import os
from logging.handlers import RotatingFileHandler

_file_handler: logging.Handler | None = None

try:
    import colorama
    colorama.init(autoreset=True)
    _COLORS = {
        logging.DEBUG:    colorama.Fore.CYAN,
        logging.INFO:     colorama.Fore.WHITE,
        logging.WARNING:  colorama.Fore.YELLOW,
        logging.ERROR:    colorama.Fore.RED,
        logging.CRITICAL: colorama.Fore.RED + colorama.Style.BRIGHT,
    }
    _RESET = colorama.Style.RESET_ALL
except ImportError:
    _COLORS = {}
    _RESET = ""


class _ColoredConsoleHandler(logging.StreamHandler):
    def emit(self, record):
        color = _COLORS.get(record.levelno, "")
        record.msg = f"{color}{record.msg}{_RESET}"
        super().emit(record)


def get_logger(name: str) -> logging.Logger:
    global _file_handler

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Shared file handler — only one instance writes to vox.log
    if _file_handler is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        logs_dir = os.path.join(project_root, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        _file_handler = RotatingFileHandler(
            os.path.join(logs_dir, "vox.log"),
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
            delay=True,
        )
        _file_handler.setLevel(logging.DEBUG)
        _file_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))

    ch = _ColoredConsoleHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter("[%(name)s] %(message)s"))

    logger.addHandler(_file_handler)
    logger.addHandler(ch)
    return logger
