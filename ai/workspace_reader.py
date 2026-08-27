from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


IGNORED_DIRECTORIES = {
    ".git",
    ".agents",
    ".ai_backups",
    ".codex",
    ".idea",
    ".pytest_cache",
    ".venv",
    ".vscode",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}

SENSITIVE_NAMES = {
    ".env",
    ".riftshell_ai_memory.json",
    ".sanushell_ai_memory.json",
    "credentials.json",
    "id_dsa",
    "id_ed25519",
    "id_rsa",
    "secrets.json",
}

BINARY_SUFFIXES = {
    ".7z", ".avi", ".bmp", ".class", ".dll", ".doc", ".docx", ".exe",
    ".gif", ".gz", ".ico", ".jar", ".jpeg", ".jpg", ".mov", ".mp3",
    ".mp4", ".pdf", ".png", ".pyc", ".so", ".tar", ".webp", ".xls",
    ".xlsx", ".zip",
}

PREFERRED_PROJECT_FILES = {
    "app.py",
    "cargo.toml",
    "dockerfile",
    "main.py",
    "package.json",
    "pyproject.toml",
    "readme.md",
    "requirements.txt",
}


@dataclass(frozen=True)
class WorkspaceInspection:
    content: str
    files: tuple[str, ...]
    warnings: tuple[str, ...] = ()


def inspect_workspace_paths(
    paths: list[str],
    *,
    workspace_root: Path,
    current_dir: Path,
    allow_outside_workspace: bool,
    max_files: int = 16,
    max_chars: int = 80_000,
) -> WorkspaceInspection:
    """Read bounded, non-sensitive workspace context for a model analysis pass."""
    workspace_root = workspace_root.resolve(strict=False)
    current_dir = current_dir.resolve(strict=False)
    requested = paths or ["."]
    sections: list[str] = []
    inspected: list[str] = []
    warnings: list[str] = []
    remaining = max_chars

    for path_text in requested[:8]:
        target = _resolve_target(path_text, current_dir)
        if not allow_outside_workspace and not _is_relative_to(target, workspace_root):
            warnings.append(f"Blocked path outside the AI workspace: {target}")
            continue
        if not target.exists():
            warnings.append(f"Path not found: {target}")
            continue
        if target.is_file():
            remaining = _append_file(
                target, workspace_root, sections, inspected, warnings, remaining
            )
        elif target.is_dir():
            tree = _directory_tree(target, workspace_root)
            tree_section = f"### Directory structure: {_display_path(target, workspace_root)}\n{tree}"
            if len(tree_section) <= remaining:
                sections.append(tree_section)
                remaining -= len(tree_section)
            for candidate in _candidate_files(target):
                if len(inspected) >= max_files or remaining <= 0:
                    break
                remaining = _append_file(
                    candidate, workspace_root, sections, inspected, warnings, remaining
                )
        if len(inspected) >= max_files or remaining <= 0:
            break

    if not sections:
        detail = "\n".join(warnings) or "No readable workspace context was found."
        return WorkspaceInspection(content=detail, files=(), warnings=tuple(warnings))

    if remaining <= 0 or len(inspected) >= max_files:
        warnings.append("Inspection was limited to keep the AI context bounded.")

    content = "\n\n".join(sections)
    if warnings:
        content += "\n\n### Inspection notes\n" + "\n".join(f"- {item}" for item in warnings)
    return WorkspaceInspection(content=content, files=tuple(inspected), warnings=tuple(warnings))


def _resolve_target(path_text: str, current_dir: Path) -> Path:
    target = Path(path_text.strip().strip("\"'")).expanduser()
    if not target.is_absolute():
        target = current_dir / target
    return target.resolve(strict=False)


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _is_sensitive(path: Path) -> bool:
    name = path.name.lower()
    if name == ".env.example":
        return False
    return name in SENSITIVE_NAMES or name.endswith((".pem", ".key", ".pfx", ".p12"))


def _is_readable_text(path: Path) -> bool:
    return path.suffix.lower() not in BINARY_SUFFIXES and not _is_sensitive(path)


def _display_path(path: Path, workspace_root: Path) -> str:
    try:
        return str(path.relative_to(workspace_root)) or "."
    except ValueError:
        return str(path)


def _append_file(
    path: Path,
    workspace_root: Path,
    sections: list[str],
    inspected: list[str],
    warnings: list[str],
    remaining: int,
) -> int:
    display = _display_path(path, workspace_root)
    if display in inspected:
        return remaining
    if _is_sensitive(path):
        warnings.append(f"Skipped sensitive file: {display}")
        return remaining
    if not _is_readable_text(path):
        warnings.append(f"Skipped binary file: {display}")
        return remaining
    try:
        raw = path.read_bytes()
    except OSError as exc:
        warnings.append(f"Could not read {display}: {exc}")
        return remaining
    if b"\x00" in raw[:4096]:
        warnings.append(f"Skipped binary file: {display}")
        return remaining

    text = raw.decode("utf-8", errors="replace")
    header = f"### File: {display}\n"
    allowance = max(0, remaining - len(header))
    if allowance <= 0:
        return 0
    if len(text) > allowance:
        text = text[: max(0, allowance - 80)] + "\n...[file truncated by RiftShell]"
        warnings.append(f"Truncated large file: {display}")
    sections.append(header + text)
    inspected.append(display)
    return max(0, remaining - len(header) - len(text))


def _directory_tree(root: Path, workspace_root: Path, max_entries: int = 220) -> str:
    lines: list[str] = []
    try:
        for current, directories, files in os.walk(root):
            current_path = Path(current)
            relative = current_path.relative_to(root)
            depth = 0 if relative == Path(".") else len(relative.parts)
            directories[:] = sorted(
                (name for name in directories if name.lower() not in IGNORED_DIRECTORIES),
                key=str.lower,
            )
            if depth >= 3:
                directories[:] = []

            for name in [*(f"{item}/" for item in directories), *sorted(files, key=str.lower)]:
                item_name = name.rstrip("/")
                item_path = current_path / item_name
                if _is_sensitive(item_path):
                    continue
                lines.append(f"{'  ' * depth}{name}")
                if len(lines) >= max_entries:
                    lines.append("...[directory listing truncated by RiftShell]")
                    return "\n".join(lines)
    except OSError as exc:
        return f"[Could not list directory: {exc}]"
    return "\n".join(lines) or "[empty directory]"


def _candidate_files(root: Path) -> list[Path]:
    candidates: list[Path] = []
    try:
        for current, directories, files in os.walk(root):
            current_path = Path(current)
            relative_dir = current_path.relative_to(root)
            depth = 0 if relative_dir == Path(".") else len(relative_dir.parts)
            directories[:] = [
                name for name in directories if name.lower() not in IGNORED_DIRECTORIES
            ]
            if depth >= 2:
                directories[:] = []
            for name in files:
                path = current_path / name
                if _is_readable_text(path):
                    candidates.append(path)
                if len(candidates) >= 2_000:
                    break
            if len(candidates) >= 2_000:
                break
    except OSError:
        return []

    def priority(path: Path) -> tuple[int, int, str]:
        name = path.name.lower()
        preferred = 0 if name in PREFERRED_PROJECT_FILES or name.startswith("readme") else 1
        depth = len(path.relative_to(root).parts)
        return preferred, depth, str(path).lower()

    return sorted(candidates, key=priority)
