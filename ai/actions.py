from __future__ import annotations

from dataclasses import dataclass, field


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
            content = str(item.get("content", ""))
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

