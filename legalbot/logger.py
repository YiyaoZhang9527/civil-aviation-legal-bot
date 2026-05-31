"""终端结构化日志。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import sys


COLORS = {
    "DEBUG": "\033[2m",
    "INFO": "\033[36m",
    "SUCCESS": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "RESET": "\033[0m",
}


@dataclass
class LogEntry:
    stage: str
    event: str
    level: str
    message: str
    timestamp: str


class TerminalLogger:
    def __init__(self, enabled: bool = True, stream=None) -> None:
        self.enabled = enabled
        self.stream = stream or sys.stderr
        self.entries: list[LogEntry] = []

    def log(self, stage: str, event: str, message: str, level: str = "INFO") -> None:
        entry = LogEntry(
            stage=stage,
            event=event,
            level=level,
            message=message,
            timestamp=datetime.now().strftime("%H:%M:%S"),
        )
        self.entries.append(entry)
        # 防止长会话内存泄漏，保留最近 500 条
        if len(self.entries) > 500:
            self.entries = self.entries[-500:]
        if not self.enabled:
            return
        color = COLORS.get(level, "")
        reset = COLORS["RESET"] if color else ""
        print(f"{color}[{entry.timestamp}] [{stage}] {event}: {message}{reset}", file=self.stream)

    def debug(self, stage: str, event: str, message: str) -> None:
        self.log(stage, event, message, "DEBUG")

    def info(self, stage: str, event: str, message: str) -> None:
        self.log(stage, event, message, "INFO")

    def success(self, stage: str, event: str, message: str) -> None:
        self.log(stage, event, message, "SUCCESS")

    def warning(self, stage: str, event: str, message: str) -> None:
        self.log(stage, event, message, "WARNING")

    def error(self, stage: str, event: str, message: str) -> None:
        self.log(stage, event, message, "ERROR")

