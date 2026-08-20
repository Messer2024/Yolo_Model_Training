"""
结构化日志与 Qt Signal 日志重定向工具 (Logger)
"""
import logging
import sys
from typing import Optional


class SafeStreamHandler(logging.StreamHandler):
    """安全控制台输出 Handler，防止 Windows cp1252 等终端编码崩溃"""
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            encoding = getattr(stream, "encoding", None) or "utf-8"
            try:
                stream.write(msg + self.terminator)
            except (UnicodeEncodeError, UnicodeError):
                # 安全字符转义替换
                safe_msg = msg.encode(encoding, errors="backslashreplace").decode(encoding, errors="replace")
                stream.write(safe_msg + self.terminator)
            self.flush()
        except Exception:
            pass


class QtLogHandler(logging.Handler):
    """用于将 Python logging 消息通过 Qt 信号发送给 UI 终端的 Handler"""
    def __init__(self, signal_emitter=None):
        super().__init__()
        self.signal_emitter = signal_emitter

    def emit(self, record):
        try:
            msg = self.format(record)
            if self.signal_emitter and hasattr(self.signal_emitter, "emit"):
                self.signal_emitter.emit(msg)
        except Exception:
            pass


def setup_logger(name: str = "YOLO_Studio", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        console_handler = SafeStreamHandler(sys.stdout)
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


logger = setup_logger()
