from dataclasses import dataclass, field
from pathlib import Path
from threading import Event


@dataclass
class ShellContext:
    cwd: Path
    workspace_root: Path | None = None
    allow_outside_workspace: bool = True
    history: list[str] = field(default_factory=list)
    last_output: str = ""
    piped_input: str = ""
    variables: dict[str, str] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    running: bool = True
    registry: object | None = None
    plugins: object | None = None
    current_theme: str = "vscode-dark-plus"
    cancel_event: Event = field(default_factory=Event, repr=False)

    def set_cwd(self, path: Path):
        from utils.safe_fs import ensure_path_allowed

        self.cwd = ensure_path_allowed(self, path)
