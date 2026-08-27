from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath

from core.parser import CommandParser


@dataclass(frozen=True)
class SafetyDecision:
    requires_approval: bool
    reason: str = ""
    level: str = "low"


class SafetyPolicy:
    """Classify impact without interrupting safe, everyday assistant actions."""

    DESTRUCTIVE_COMMANDS = {
        "delete",
        "remove",
        "del",
        "kill",
        "taskkill",
    }

    FILE_RELOCATION_COMMANDS = {
        "shift",
        "move",
        "rename",
        "ren",
    }

    EXTERNAL_OR_BULK_COMMANDS = {
        "download",
        "curl",
        "wget",
        "unzip",
        "extract",
        "gcommit",
        "gc",
        "gpull",
        "gpl",
        "gpush",
        "gps",
        "gcheckout",
        "gco",
    }

    NATIVE_WRAPPERS = {"run", "exec", "native"}

    # These programs cannot turn into general-purpose code execution through
    # their normal arguments, so inspection requests should remain frictionless.
    SAFE_NATIVE_PROGRAMS = {"hostname", "systeminfo", "tasklist", "whoami"}
    VERSION_ARGS_BY_PROGRAM = {
        "dotnet": {"--version"},
        "java": {"-version", "--version"},
        "node": {"-v", "--version"},
        "npm": {"-v", "--version"},
        "pip": {"-v", "--version"},
        "pip3": {"-v", "--version"},
        "py": {"-V", "--version"},
        "python": {"-V", "--version"},
        "python3": {"-V", "--version"},
    }
    SAFE_PIP_SUBCOMMANDS = {"check", "freeze", "list", "show"}
    SAFE_NPM_SUBCOMMANDS = {"audit", "list", "outdated", "view"}
    EXECUTABLE_FILE_SUFFIXES = {
        ".bat", ".cmd", ".com", ".exe", ".lnk", ".msi", ".msix", ".ps1", ".scr",
    }

    def __init__(self) -> None:
        self.parser = CommandParser()

    def check_action(self, action: str, command: str = "") -> SafetyDecision:
        action = action.lower().strip()
        if action == "code_write":
            return self._approval(
                "AI-generated file changes need review before they are applied.",
                level="high",
            )
        if action == "screenshot":
            # The planner only emits this action for an explicit screenshot
            # request, so a second confirmation would add needless friction.
            return SafetyDecision(False, "Explicit screenshot request.", "low")
        if action != "shell":
            return SafetyDecision(False, "Conversational response.", "low")
        return self.check_shell_command(command)

    def check_shell_command(self, command: str) -> SafetyDecision:
        raw = command.strip()
        if not raw:
            return SafetyDecision(False, "No executable action.", "low")

        try:
            parsed_parts = self.parser.parse_line(raw)
        except Exception:
            parsed_parts = []
        if not parsed_parts:
            # RiftShell uses the same parser and therefore will not execute an
            # invalid/empty line. A confirmation here would not add protection.
            return SafetyDecision(False, "No valid executable action.", "low")

        medium_reason = ""
        for part in parsed_parts:
            for parsed in part.pipeline.commands:
                name = parsed.name.lower()
                if name in self.DESTRUCTIVE_COMMANDS:
                    return self._approval(
                        f"`{name}` can permanently remove data or stop a process.",
                        level="critical",
                    )

                if name in self.FILE_RELOCATION_COMMANDS:
                    return self._approval(
                        f"`{name}` changes an existing file or folder location.",
                        level="high",
                    )

                if name in self.EXTERNAL_OR_BULK_COMMANDS:
                    return self._approval(
                        f"`{name}` can change many files, contact an external service, or alter Git history.",
                        level="high",
                    )

                if name in self.NATIVE_WRAPPERS:
                    native_decision = self._check_native_command(parsed.args)
                    if native_decision.requires_approval:
                        return native_decision

                if name in {"gbranch", "gb"} and not self._safe_branch_args(parsed.args):
                    return self._approval(
                        "This Git branch command can create, rename, or delete a branch.",
                        level="high",
                    )

                if name in {"open", "start"} and parsed.args:
                    suffix = PurePath(parsed.args[0].strip("\"'")).suffix.lower()
                    if suffix in self.EXECUTABLE_FILE_SUFFIXES:
                        return self._approval(
                            f"Opening a `{suffix}` file can execute code on this computer.",
                            level="high",
                        )

                if parsed.redirect_path:
                    medium_reason = "The command writes terminal output to a local file."

                if name in {
                    "alias", "duplicate", "copy", "makefile", "touch", "makefolder",
                    "mkdir", "open", "start", "save", "setvar", "let", "theme",
                    "zip", "compress",
                }:
                    medium_reason = f"`{name}` makes a local, limited, and recoverable change."

        if medium_reason:
            return SafetyDecision(False, medium_reason, "medium")
        return SafetyDecision(False, "Read-only or session-local action.", "low")

    def _check_native_command(self, args: list[str]) -> SafetyDecision:
        if not args:
            return self._approval("An empty native command cannot be validated.", level="high")

        program = PurePath(args[0].strip("\"'")).name.lower()
        if program.endswith(".exe"):
            program = program[:-4]
        native_args = [arg.strip("\"'") for arg in args[1:]]
        lowered_args = [arg.lower() for arg in native_args]

        if program in self.SAFE_NATIVE_PROGRAMS:
            return SafetyDecision(False, f"`{program}` only inspects system state.", "low")

        if program == "ipconfig" and all(
            arg in {"/all", "/displaydns", "/showclassid"} for arg in lowered_args
        ):
            return SafetyDecision(False, "`ipconfig` only inspects network state.", "low")

        if program in self.VERSION_ARGS_BY_PROGRAM and native_args and all(
            arg in self.VERSION_ARGS_BY_PROGRAM[program] for arg in native_args
        ):
            return SafetyDecision(False, f"`{program}` only reports version information.", "low")

        if program == "git" and self._is_safe_git_inspection(lowered_args):
            return SafetyDecision(False, f"`git {lowered_args[0]}` is a read-only Git inspection.", "low")

        if program in {"pip", "pip3"} and lowered_args and lowered_args[0] in self.SAFE_PIP_SUBCOMMANDS:
            return SafetyDecision(False, f"`{program} {lowered_args[0]}` inspects the Python environment.", "low")

        if (
            program == "npm"
            and lowered_args
            and lowered_args[0] in self.SAFE_NPM_SUBCOMMANDS
            and "--fix" not in lowered_args
            and "--force" not in lowered_args
        ):
            return SafetyDecision(False, f"`npm {lowered_args[0]}` inspects package information.", "low")

        return self._approval(
            f"Native command `{program}` can execute code outside RiftShell's controlled command set.",
            level="high",
        )

    @staticmethod
    def _is_safe_git_inspection(args: list[str]) -> bool:
        if not args:
            return False
        subcommand, rest = args[0], args[1:]
        if subcommand in {"diff", "log", "show", "status"}:
            return True
        if subcommand == "branch":
            return SafetyPolicy._safe_branch_args(rest)
        if subcommand == "remote":
            return not rest or rest[0] in {"-v", "get-url", "show"}
        return False

    @staticmethod
    def _safe_branch_args(args: list[str]) -> bool:
        safe_branch_args = {"-a", "--all", "--list", "--show-current", "-v", "-vv"}
        return all(arg.lower() in safe_branch_args for arg in args)

    @staticmethod
    def _approval(reason: str, *, level: str) -> SafetyDecision:
        return SafetyDecision(True, reason, level)

