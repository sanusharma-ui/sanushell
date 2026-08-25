from pathlib import Path
import os
import re

from core.base import CommandResult
from core.context import ShellContext
from core.parser import CommandParser, ParsedCommand
from commands import build_registry
from utils.safe_fs import ensure_path_allowed, resolve_path


class Shell:
    def __init__(
        self,
        start_dir: Path | None = None,
        workspace_root: Path | None = None,
        allow_outside_workspace: bool = True,
    ):
        workspace_root = workspace_root.resolve(strict=False) if workspace_root else None
        start_dir = (start_dir or workspace_root or Path.cwd()).resolve(strict=False)
        self.ctx = ShellContext(
            cwd=start_dir,
            workspace_root=workspace_root,
            allow_outside_workspace=allow_outside_workspace,
        )
        ensure_path_allowed(self.ctx, self.ctx.cwd)
        self.parser = CommandParser()
        self.registry = build_registry()
        self.ctx.registry = self.registry
        self.ctx.plugins = getattr(self.registry, "plugin_report", None)

    def prompt(self) -> str:
        return f"RiftShell {self.ctx.cwd}> "

    def request_cancel(self) -> None:
        """Signal cooperative commands (such as native processes) to stop."""
        self.ctx.cancel_event.set()

    def clear_cancel(self) -> None:
        self.ctx.cancel_event.clear()

    def execute_line(self, raw: str) -> CommandResult:
        raw = raw.strip()
        if not raw:
            return CommandResult(output="")

        parts = self.parser.parse_line(raw)
        if not parts:
            return CommandResult(output="")

        self.ctx.history.append(raw)

        outputs = []
        last_result = CommandResult(output="")

        for part in parts:
            if self.ctx.cancel_event.is_set():
                return CommandResult(output="Command cancelled.", success=False)
            if part.operator == "&&" and not last_result.success:
                continue
            if part.operator == "||" and last_result.success:
                continue

            last_result = self._execute_pipeline(part.pipeline.commands)
            if last_result.output:
                outputs.append(last_result.output)
            if last_result.exit_shell:
                self.ctx.running = False
                break

        combined = "\n".join(outputs)
        return CommandResult(
            output=combined,
            success=last_result.success,
            exit_shell=last_result.exit_shell,
            actions=last_result.actions,
        )

    def _execute_pipeline(self, commands: list[ParsedCommand]) -> CommandResult:
        previous_output = ""
        result = CommandResult(output="")

        for index, parsed in enumerate(commands):
            if self.ctx.cancel_event.is_set():
                return CommandResult(output="Command cancelled.", success=False)
            self.ctx.piped_input = previous_output if index > 0 else ""
            result = self._execute_command(parsed)
            previous_output = result.output
            if not result.success or result.exit_shell:
                break

        self.ctx.piped_input = ""
        return result

    def _execute_command(self, parsed: ParsedCommand) -> CommandResult:
        if self.ctx.cancel_event.is_set():
            return CommandResult(output="Command cancelled.", success=False)
        parsed = self._prepare_command(parsed)
        cmd = self.registry.get(parsed.name)
        if cmd is None:
            return CommandResult(output=f"Unknown command: {parsed.name}", success=False)

        try:
            result = cmd.execute(self.ctx, parsed.args)
            result = self._handle_redirect(parsed, result)
            self.ctx.last_output = result.output
            if result.exit_shell:
                self.ctx.running = False
            return result
        except Exception as e:
            return CommandResult(output=f"[error] {e}", success=False)

    def _prepare_command(self, parsed: ParsedCommand) -> ParsedCommand:
        expanded_raw = self._expand_alias(parsed.raw)
        if expanded_raw != parsed.raw:
            parsed = self.parser.parse(expanded_raw) or parsed

        return ParsedCommand(
            raw=parsed.raw,
            name=self._expand_vars(parsed.name).lower(),
            args=[self._expand_vars(arg) for arg in parsed.args],
            redirect_path=self._expand_vars(parsed.redirect_path) if parsed.redirect_path else None,
            redirect_append=parsed.redirect_append,
        )

    def _expand_alias(self, raw: str) -> str:
        stripped = raw.strip()
        if not stripped:
            return raw

        parts = stripped.split(maxsplit=1)
        alias_value = self.ctx.aliases.get(parts[0].lower())
        if not alias_value:
            return raw

        suffix = f" {parts[1]}" if len(parts) > 1 else ""
        return alias_value + suffix

    def _expand_vars(self, value: str) -> str:
        def dollar_replace(match):
            key = match.group(1)
            return self.ctx.variables.get(key, os.environ.get(key, ""))

        def percent_replace(match):
            key = match.group(1)
            return self.ctx.variables.get(key, os.environ.get(key, ""))

        value = re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", dollar_replace, value)
        value = re.sub(r"%([^%]+)%", percent_replace, value)
        return value

    def _handle_redirect(self, parsed: ParsedCommand, result: CommandResult) -> CommandResult:
        if not parsed.redirect_path:
            return result

        target = resolve_path(self.ctx, parsed.redirect_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if parsed.redirect_append else "w"
        with open(target, mode, encoding="utf-8", errors="replace") as file:
            file.write(result.output)
            if result.output and not result.output.endswith("\n"):
                file.write("\n")

        action = "Appended output to" if parsed.redirect_append else "Wrote output to"
        return CommandResult(
            output=f"{action}: {target}",
            success=result.success,
            exit_shell=result.exit_shell,
            actions=result.actions,
        )

    def run(self):
        print("RiftShell ready. Type 'help' to see commands.\n")

        while self.ctx.running:
            try:
                raw = input(self.prompt())
            except (KeyboardInterrupt, EOFError):
                print("\nexit")
                break

            result = self.execute_line(raw)
            if result.output:
                print(result.output)
