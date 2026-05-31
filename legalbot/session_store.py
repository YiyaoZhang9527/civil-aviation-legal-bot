"""Markdown session history and pending clarification storage."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from .config import DATA_DIR


class SessionStore:
    def __init__(self, sessions_dir: Path | None = None) -> None:
        self.sessions_dir = sessions_dir or (DATA_DIR / "sessions")
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def load_history(self, session_id: str) -> list[dict]:
        path = self._history_path(session_id)
        if not path.exists():
            return []
        text = path.read_text(encoding="utf-8")
        parts = re.split(r"^## (用户|助手)\s*$", text, flags=re.MULTILINE)
        history: list[dict] = []
        for i in range(1, len(parts), 2):
            role = "user" if parts[i] == "用户" else "assistant"
            content = _strip_frontmatter(parts[i + 1]).strip()
            if content:
                history.append({"role": role, "content": content})
        return history

    def save_history(self, session_id: str, history: list[dict]) -> None:
        path = self._history_path(session_id)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rounds = sum(1 for item in history if item.get("role") == "user")
        lines = [
            "---",
            f"session_id: {session_id}",
            f'updated: "{now}"',
            f"rounds: {rounds}",
            "---",
            "",
        ]
        for item in history:
            role = "用户" if item.get("role") == "user" else "助手"
            lines.append(f"## {role}")
            lines.append(str(item.get("content", "")).strip())
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")

    def load_pending(self, session_id: str) -> dict | None:
        path = self._pending_path(session_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def save_pending(self, session_id: str, pending: dict) -> None:
        path = self._pending_path(session_id)
        payload = dict(pending)
        payload["updated_at"] = datetime.now().isoformat()
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def clear_pending(self, session_id: str) -> None:
        path = self._pending_path(session_id)
        if path.exists():
            path.unlink()

    def clear(self, session_id: str) -> None:
        for path in (self._history_path(session_id), self._pending_path(session_id)):
            if path.exists():
                path.unlink()

    def _history_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{_safe_session_id(session_id)}.md"

    def _pending_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{_safe_session_id(session_id)}.pending.json"


def _safe_session_id(session_id: str) -> str:
    safe = re.sub(r"[^\w\-.]", "_", session_id.strip())
    return safe or "default"


def _strip_frontmatter(text: str) -> str:
    if text.lstrip().startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            return parts[2]
    return text
