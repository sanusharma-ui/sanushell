from __future__ import annotations

import json
from dataclasses import dataclass, field


def _escaped_line_breaks_outside_strings(text: str) -> int:
    """Count slash-n/slash-r sequences that appear outside source strings."""
    count = 0
    quote: str | None = None
    index = 0
    while index < len(text):
        char = text[index]
        if quote is not None:
            if char == "\\" and index + 1 < len(text):
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue

        if char == "\\" and index + 1 < len(text):
            if text[index + 1] in {"n", "r"}:
                count += 1
            index += 2
            continue
        if char in {"'", '"', "`"}:
            quote = char
        index += 1
    return count


def _restore_overescaped_file_content(content: str) -> str:
    """Undo one accidental JSON-escape layer around a complete source file.

    Some providers occasionally return the `content` field as though it were a
    JSON string nested inside the real JSON string. After normal JSON parsing,
    that leaves source-line separators and quotes on disk as literal ``\\n`` and
    ``\\\"`` text. Only recover documents with no real internal line breaks and
    strong evidence that the slash-newlines describe file structure. This
    avoids changing intentional escapes inside JavaScript/Python strings.
    """
    candidate = content.strip("\r\n")
    if "\n" in candidate or "\r" in candidate:
        return content

    escaped_breaks = candidate.count("\\n") + candidate.count("\\r")
    if escaped_breaks < 2:
        return content
    outside_strings = _escaped_line_breaks_outside_strings(candidate)
    escaped_quotes = candidate.count('\\"')
    if outside_strings < 2 and escaped_quotes < 2:
        return content

    try:
        restored = json.loads(f'"{candidate}"')
    except (json.JSONDecodeError, TypeError):
        return content
    if not isinstance(restored, str) or restored.count("\n") + restored.count("\r") < 2:
        return content
    return restored


@dataclass(frozen=True)
class FileWrite:
    path: str
    content: str


@dataclass(frozen=True)
class AgentAction:
    action: str
    message: str = ""
    command: str = ""
    files: list[FileWrite] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    objective: str = ""

    @classmethod
    def from_payload(cls, payload: dict) -> "AgentAction":
        action = str(payload.get("action", "respond")).strip().lower()
        message = str(payload.get("message", "")).strip()
        command = str(payload.get("command", "")).strip()
        files = []
        paths = []

        for item in payload.get("files", []) or []:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path", "")).strip()
            content = _restore_overescaped_file_content(str(item.get("content", "")))
            if path:
                files.append(FileWrite(path=path, content=content))

        for item in payload.get("paths", []) or []:
            path = str(item).strip()
            if path:
                paths.append(path)

        objective = str(payload.get("objective", "")).strip()

        if action not in {"respond", "shell", "screenshot", "code_write", "inspect"}:
            action = "respond"
            message = message or "I could not map that safely."

        return cls(
            action=action,
            message=message,
            command=command,
            files=files,
            paths=paths,
            objective=objective,
        )

