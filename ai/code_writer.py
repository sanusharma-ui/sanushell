from __future__ import annotations

import difflib
import hashlib
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from ai.actions import FileWrite


@dataclass(frozen=True)
class WriteResult:
    output: str
    success: bool


@dataclass(frozen=True)
class FileSnapshot:
    """Exact on-disk state that a user reviewed before approving a write."""

    path: str
    exists: bool
    sha256: str = ""


@dataclass(frozen=True)
class WritePreview:
    output: str
    success: bool
    snapshots: tuple[FileSnapshot, ...] = ()


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _backup_relative_path(target: Path, workspace_root: Path) -> Path:
    if _is_relative_to(target, workspace_root):
        return target.relative_to(workspace_root)

    drive = target.drive.replace(":", "") if target.drive else "root"
    parts = [part for part in target.parts if part not in {target.anchor, target.drive, "\\"}]
    return Path("outside_workspace") / drive / Path(*parts)


def _resolve_target(path_text: str, base_dir: Path) -> Path:
    target = Path(path_text).expanduser()
    if not target.is_absolute():
        target = base_dir / target
    return target.resolve(strict=False)


def _snapshot(target: Path) -> FileSnapshot:
    if not target.exists():
        return FileSnapshot(path=str(target), exists=False)
    if not target.is_file():
        raise OSError(f"Target is not a regular file: {target}")
    return FileSnapshot(
        path=str(target),
        exists=True,
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    )


def _validate_targets(
    files: list[FileWrite],
    workspace_root: Path,
    base_dir: Path,
    allow_outside_workspace: bool,
) -> tuple[list[tuple[FileWrite, Path]], str]:
    targets: list[tuple[FileWrite, Path]] = []
    seen: set[str] = set()
    for file in files:
        target = _resolve_target(file.path, base_dir)
        target_key = os.path.normcase(str(target))
        if target_key in seen:
            return [], f"Blocked duplicate file target in one AI change: {target}"
        seen.add(target_key)

        if not allow_outside_workspace and not _is_relative_to(target, workspace_root):
            return [], (
                f"Blocked write outside AI workspace: {target}\n"
                f"Workspace: {workspace_root}\n"
                "Set AI_ALLOW_OUTSIDE_WORKSPACE=true to allow full-PC paths."
            )
        if target.exists() and not target.is_file():
            return [], f"Blocked file write because the target is not a regular file: {target}"
        targets.append((file, target))
    return targets, ""


def _stage_content(target: Path, content: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged_name = tempfile.mkstemp(
        prefix=f".{target.name}.orbit-",
        suffix=".tmp",
        dir=target.parent,
    )
    staged = Path(staged_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content.encode("utf-8", errors="replace"))
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists():
            shutil.copymode(target, staged)
        return staged
    except Exception:
        staged.unlink(missing_ok=True)
        raise


def _restore_target(target: Path, backup: Path | None) -> None:
    if backup is None:
        target.unlink(missing_ok=True)
        return
    descriptor, rollback_name = tempfile.mkstemp(
        prefix=f".{target.name}.rollback-", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    rollback = Path(rollback_name)
    try:
        shutil.copy2(backup, rollback)
        os.replace(rollback, target)
    finally:
        rollback.unlink(missing_ok=True)


def apply_file_writes(
    files: list[FileWrite],
    workspace_root: Path,
    allow_outside_workspace: bool,
    current_dir: Path | None = None,
    expected_snapshots: Sequence[FileSnapshot] | None = None,
) -> WriteResult:
    """Apply a reviewed set of writes as one rollback-capable transaction."""
    if not files:
        return WriteResult(output="No files were provided by the AI.", success=False)

    workspace_root = workspace_root.resolve(strict=False)
    base_dir = (current_dir or workspace_root).resolve(strict=False)
    targets, error = _validate_targets(files, workspace_root, base_dir, allow_outside_workspace)
    if error:
        return WriteResult(output=error, success=False)

    if expected_snapshots is None:
        return WriteResult(
            output="Blocked unreviewed AI change. Generate and approve a diff before applying.",
            success=False,
        )
    expected = {os.path.normcase(item.path): item for item in expected_snapshots}
    if len(expected) != len(targets):
        return WriteResult(
            output="The reviewed file snapshot is incomplete. Generate a fresh diff before applying.",
            success=False,
        )
    for _, target in targets:
        try:
            current = _snapshot(target)
        except OSError as exc:
            return WriteResult(output=f"Could not verify {target}: {exc}", success=False)
        reviewed = expected.get(os.path.normcase(str(target)))
        if reviewed != current:
            return WriteResult(
                output=(
                    f"Blocked stale AI change: {target} changed after the diff was shown.\n"
                    "Nothing was written. Ask Orbit to inspect again and review a fresh diff."
                ),
                success=False,
            )

    transaction_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{uuid.uuid4().hex[:8]}"
    backup_root = workspace_root / ".ai_backups" / transaction_id
    staged_files: dict[Path, Path] = {}
    backups: dict[Path, Path | None] = {}
    applied: list[Path] = []

    try:
        # Prepare every replacement and backup before changing any target.
        for file, target in targets:
            staged_files[target] = _stage_content(target, file.content)
            if target.exists():
                backup = backup_root / _backup_relative_path(target, workspace_root)
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup)
                backups[target] = backup
            else:
                backups[target] = None

        for _, target in targets:
            if expected[os.path.normcase(str(target))] != _snapshot(target):
                raise OSError(
                    f"{target} changed while the approved transaction was being prepared"
                )
            staged = staged_files[target]
            os.replace(staged, target)
            staged_files.pop(target)
            applied.append(target)
    except Exception as exc:
        rollback_errors: list[str] = []
        for target in reversed(applied):
            try:
                _restore_target(target, backups[target])
            except Exception as rollback_exc:
                rollback_errors.append(f"{target}: {rollback_exc}")
        detail = f"Atomic file update failed and applied changes were rolled back: {exc}"
        if rollback_errors:
            detail += "\nRollback needs manual attention:\n" + "\n".join(
                f"- {item}" for item in rollback_errors
            )
        return WriteResult(output=detail, success=False)
    finally:
        for staged in staged_files.values():
            staged.unlink(missing_ok=True)

    lines = [
        "AI code write completed atomically.",
        "",
        f"Base directory: {base_dir}",
        f"Transaction: {transaction_id}",
        "",
        "Written files:",
    ]
    lines.extend(f"- {target}" for _, target in targets)
    backed_up = [backup for backup in backups.values() if backup is not None]
    if backed_up:
        lines.extend(["", "Backups:"])
        lines.extend(f"- {path}" for path in backed_up)
    return WriteResult(output="\n".join(lines), success=True)


def preview_file_writes(
    files: list[FileWrite],
    workspace_root: Path,
    allow_outside_workspace: bool,
    current_dir: Path | None = None,
    max_chars: int = 24_000,
) -> WritePreview:
    """Build a complete bounded diff and capture the exact reviewed file states."""
    if not files:
        return WritePreview(output="No files were provided by the AI.", success=False)

    workspace_root = workspace_root.resolve(strict=False)
    base_dir = (current_dir or workspace_root).resolve(strict=False)
    targets, error = _validate_targets(files, workspace_root, base_dir, allow_outside_workspace)
    if error:
        return WritePreview(output=error, success=False)

    sections: list[str] = []
    snapshots: list[FileSnapshot] = []
    for file, target in targets:
        try:
            raw = target.read_bytes() if target.exists() else b""
            previous = raw.decode("utf-8", errors="replace")
            snapshots.append(
                FileSnapshot(
                    path=str(target),
                    exists=target.exists(),
                    sha256=hashlib.sha256(raw).hexdigest() if target.exists() else "",
                )
            )
        except OSError as exc:
            return WritePreview(output=f"Could not read {target}: {exc}", success=False)

        relative = _backup_relative_path(target, workspace_root)
        old_name = str(relative) if target.exists() else "/dev/null"
        new_name = str(relative)
        diff = "\n".join(
            difflib.unified_diff(
                previous.splitlines(),
                file.content.splitlines(),
                fromfile=old_name,
                tofile=new_name,
                lineterm="",
            )
        )
        sections.append(diff or f"No content changes for {new_name}.")

    output = "\n\n".join(sections)
    if len(output) > max_chars:
        return WritePreview(
            output=(
                f"The proposed diff is {len(output):,} characters, above the safe review limit "
                f"of {max_chars:,}. No approval is available for a partial diff. "
                "Ask Orbit to split the change into smaller edits."
            ),
            success=False,
        )
    return WritePreview(output=output, success=True, snapshots=tuple(snapshots))
