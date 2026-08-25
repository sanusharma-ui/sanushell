from __future__ import annotations

from html import escape
from pathlib import Path

from PySide6.QtCore import Qt, QSettings, QThread, Signal, QStringListModel
from PySide6.QtGui import QFont, QKeySequence, QShortcut, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QCompleter, QDialog, QDialogButtonBox,
    QFormLayout, QFrame, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton,
    QSpinBox, QSplitter, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

from core.shell import Shell
from ui.themes import build_stylesheet, get_theme, list_themes


RISKY_COMMANDS = {
    "delete", "remove", "del", "shift", "move", "rename", "run", "exec",
    "native", "open", "download", "kill", "taskkill", "zip", "unzip",
    "gpush", "gps", "gcheckout", "gco",
}


def fuzzy_score(query: str, text: str) -> int | None:
    """Small subsequence matcher: lower score means a more relevant result."""
    query, text = query.lower().strip(), text.lower()
    if not query:
        return 0
    if query in text:
        return text.index(query)
    position = -1
    gaps = 0
    for char in query:
        next_position = text.find(char, position + 1)
        if next_position < 0:
            return None
        if position >= 0:
            gaps += next_position - position - 1
        position = next_position
    return 50 + gaps


class CommandInput(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history: list[str] = []
        self.history_index = 0
        self.command_names: list[str] = []

    def set_history(self, history: list[str]):
        self.history = history[:]
        self.history_index = len(self.history)

    def set_completion_items(self, items: list[str]):
        self.command_names = sorted({item.lower() for item in items})

    def _complete_current_command(self) -> bool:
        text = self.text()
        if not text.strip():
            return False
        parts = text.split(maxsplit=1)
        prefix = parts[0]
        rest = f" {parts[1]}" if len(parts) > 1 else ""
        matches = [name for name in self.command_names if name.startswith(prefix.lower())]
        if not matches:
            return False
        self.setText(matches[0] + rest)
        self.setCursorPosition(len(matches[0]))
        return True

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Tab and self._complete_current_command():
            return
        if event.key() == Qt.Key_Up and self.history:
            self.history_index = max(0, self.history_index - 1)
            self.setText(self.history[self.history_index])
            self.setCursorPosition(len(self.text()))
            return
        if event.key() == Qt.Key_Down and self.history:
            self.history_index = min(len(self.history), self.history_index + 1)
            self.setText("" if self.history_index == len(self.history) else self.history[self.history_index])
            self.setCursorPosition(len(self.text()))
            return
        super().keyPressEvent(event)


class CommandWorker(QThread):
    result_ready = Signal(object)

    def __init__(self, shell: Shell, command: str, parent=None):
        super().__init__(parent)
        self.shell = shell
        self.command = command

    def run(self):
        self.shell.clear_cancel()
        if not self.isInterruptionRequested():
            self.result_ready.emit(self.shell.execute_line(self.command))


class CopilotWorker(QThread):
    planned = Signal(object)
    failed = Signal(str)

    def __init__(self, shell: Shell, prompt: str, parent=None):
        super().__init__(parent)
        self.shell = shell
        self.prompt = prompt

    def run(self):
        try:
            from ai.config import AIConfig
            from ai.llm import AgentPlanner

            config = AIConfig.from_env()
            planner = AgentPlanner(
                config=config,
                command_names=self.shell.registry.all_names(),
                command_catalog=self.shell.registry.catalog_entries(),
                current_dir_provider=lambda: self.shell.ctx.cwd,
            )
            self.planned.emit(planner.plan(self.prompt))
        except Exception as exc:
            self.failed.emit(str(exc))


class CommandPalette(QDialog):
    command_chosen = Signal(str)
    action_chosen = Signal(str)

    def __init__(self, shell: Shell, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Command Palette")
        self.setMinimumSize(660, 460)
        heading = QLabel("COMMAND PALETTE")
        heading.setObjectName("eyebrow")
        self.query = QLineEdit()
        self.query.setPlaceholderText("Search commands, actions, or plugins…")
        self.list = QListWidget()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.addWidget(heading)
        layout.addWidget(self.query)
        layout.addWidget(self.list, 1)
        self._entries: list[tuple[str, str, str]] = [
            ("action", "New terminal session", "Create a workspace session"),
            ("action", "Open settings", "Theme, font and safety preferences"),
            ("action", "Toggle Copilot", "Show or hide the AI planning panel"),
            ("action", "Show plugins", "Inspect loaded plugin status"),
        ]
        self._entries.extend(("command", meta.name, f"{meta.usage} — {meta.description}") for meta in shell.registry.list_metadata())
        self.query.textChanged.connect(self._refresh)
        self.query.returnPressed.connect(self._choose_current)
        self.list.itemActivated.connect(lambda _: self._choose_current())
        self._refresh("")
        self.query.setFocus()

    def _refresh(self, query: str):
        ranked = []
        for kind, name, detail in self._entries:
            score = fuzzy_score(query, f"{name} {detail}")
            if score is not None:
                ranked.append((score, kind, name, detail))
        self.list.clear()
        for _, kind, name, detail in sorted(ranked, key=lambda item: (item[0], item[2]))[:70]:
            item = QListWidgetItem(f"{'›' if kind == 'action' else '>_'}  {name}\n     {detail}")
            item.setData(Qt.UserRole, (kind, name))
            self.list.addItem(item)
        if self.list.count():
            self.list.setCurrentRow(0)

    def _choose_current(self):
        item = self.list.currentItem()
        if item is None:
            return
        kind, name = item.data(Qt.UserRole)
        (self.command_chosen if kind == "command" else self.action_chosen).emit(name)
        self.accept()


class PreferencesDialog(QDialog):
    preferences_saved = Signal(dict)

    def __init__(self, preferences: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Workspace settings")
        self.setMinimumWidth(440)
        title = QLabel("WORKSPACE SETTINGS")
        title.setObjectName("eyebrow")
        self.theme = QComboBox()
        for item in list_themes():
            self.theme.addItem(item.display_name, item.key)
        self.theme.setCurrentIndex(max(0, self.theme.findData(preferences["theme"])))
        self.font = QComboBox()
        self.font.addItems(["Cascadia Code", "JetBrains Mono", "Consolas", "Fira Code"])
        self.font.setCurrentText(preferences["font_family"])
        self.font_size = QSpinBox()
        self.font_size.setRange(9, 20)
        self.font_size.setValue(preferences["font_size"])
        self.confirm = QCheckBox("Confirm commands that can change files or run native tools")
        self.confirm.setChecked(preferences["confirm_risky"])
        self.restore = QCheckBox("Restore recent command history on launch")
        self.restore.setChecked(preferences["restore_history"])
        form = QFormLayout()
        form.addRow("Theme", self.theme)
        form.addRow("Terminal font", self.font)
        form.addRow("Font size", self.font_size)
        form.addRow("Safety", self.confirm)
        form.addRow("Workspace", self.restore)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.addWidget(title)
        layout.addLayout(form)
        layout.addStretch()
        layout.addWidget(buttons)

    def _save(self):
        self.preferences_saved.emit({
            "theme": self.theme.currentData(), "font_family": self.font.currentText(),
            "font_size": self.font_size.value(), "confirm_risky": self.confirm.isChecked(),
            "restore_history": self.restore.isChecked(),
        })
        self.accept()


class TerminalSession(QWidget):
    session_changed = Signal()
    history_added = Signal(str)
    close_requested = Signal(object)

    def __init__(self, shell: Shell, title: str, preferences: dict, parent=None):
        super().__init__(parent)
        self.shell, self.title, self.preferences = shell, title, preferences
        self.worker: CommandWorker | None = None
        self.name_label = QLabel(title)
        self.name_label.setFont(QFont(preferences["font_family"], 11, QFont.Bold))
        self.path_label = QLabel(str(shell.ctx.cwd))
        self.path_label.setObjectName("sessionPath")
        self.state_label = QLabel("READY")
        self.state_label.setObjectName("statusPill")
        header = QFrame()
        header.setObjectName("terminalHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)
        header_layout.addWidget(self.name_label)
        header_layout.addWidget(self.path_label, 1)
        header_layout.addWidget(self.state_label)
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.document().setMaximumBlockCount(5000)
        self.input = CommandInput()
        self.input.setPlaceholderText("Run a RiftShell command…  Ctrl+K for command palette")
        self.input.set_history(shell.ctx.history)
        names = shell.registry.all_names()
        self.input.set_completion_items(names)
        completer = QCompleter(QStringListModel(names, self), self)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        self.input.setCompleter(completer)
        self.run_button = QPushButton("Run  ↵")
        self.run_button.setObjectName("primaryButton")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.clear_button = QPushButton("Clear")
        self.copy_button = QPushButton("Copy output")
        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        bottom.addWidget(self.input, 1)
        bottom.addWidget(self.run_button)
        bottom.addWidget(self.cancel_button)
        bottom.addWidget(self.clear_button)
        bottom.addWidget(self.copy_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(header)
        layout.addWidget(self.console, 1)
        layout.addLayout(bottom)
        self.run_button.clicked.connect(self.run_command)
        self.input.returnPressed.connect(self.run_command)
        self.cancel_button.clicked.connect(self.request_cancel)
        self.clear_button.clicked.connect(self.clear_console)
        self.copy_button.clicked.connect(self.copy_output)
        self.apply_preferences(preferences)
        self.append_system("Workspace terminal ready. Type help to explore available commands.")

    @property
    def is_busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def set_title(self, title: str):
        self.title = title
        self.name_label.setText(title)
        self.session_changed.emit()

    def apply_preferences(self, preferences: dict):
        self.preferences = preferences
        font = QFont(preferences["font_family"], preferences["font_size"])
        self.console.setFont(font)
        self.input.setFont(font)
        self.name_label.setFont(QFont(preferences["font_family"], 11, QFont.Bold))

    def _theme(self):
        return get_theme(self.preferences["theme"])

    def append_html(self, text: str, color: str, label: str = ""):
        safe = escape(text).replace("\n", "<br>")
        prefix = f'<span style="color:{self._theme().muted};">{escape(label)}</span> ' if label else ""
        self.console.moveCursor(QTextCursor.End)
        self.console.append(f'<div style="color:{color}; white-space:pre-wrap;">{prefix}{safe}</div>')
        self.console.moveCursor(QTextCursor.End)

    def append_system(self, text: str):
        self.append_html(text, self._theme().text, "•")

    def append_output(self, text: str):
        self.append_html(text, self._theme().output)

    def append_error(self, text: str):
        self.append_html(text, self._theme().error, "!")

    def clear_console(self):
        self.console.clear()
        self.append_system("Console cleared.")

    def copy_output(self):
        QApplication.clipboard().setText(self.console.toPlainText())
        self.state_label.setText("COPIED")

    def _needs_confirmation(self, command: str) -> bool:
        first = command.strip().split(maxsplit=1)[0].lower() if command.strip() else ""
        return self.preferences["confirm_risky"] and first in RISKY_COMMANDS

    def run_command(self):
        command = self.input.text().strip()
        if not command or self.is_busy:
            return
        if command.lower() in {"clear", "cls"}:
            self.clear_console()
            self.input.clear()
            return
        if self._needs_confirmation(command):
            dialog = QMessageBox(self)
            dialog.setIcon(QMessageBox.Warning)
            dialog.setWindowTitle("Confirm command")
            dialog.setText("This command can change files, launch a program, or affect a system process.")
            dialog.setInformativeText(command)
            dialog.setStandardButtons(QMessageBox.Cancel | QMessageBox.Yes)
            dialog.setDefaultButton(QMessageBox.Cancel)
            if dialog.exec() != QMessageBox.Yes:
                self.append_system("Command cancelled before execution.")
                return
        self.append_html(f"{self.shell.prompt()}{command}", self._theme().accent_alt, "$")
        self._set_running(True)
        self.worker = CommandWorker(self.shell, command, self)
        self.worker.result_ready.connect(self._on_finished)
        self.worker.finished.connect(self._worker_finished)
        self.worker.start()

    def request_cancel(self):
        if self.worker:
            self.worker.requestInterruption()
            self.shell.request_cancel()
            self.cancel_button.setEnabled(False)
            self.append_system("Stop requested. Native processes stop immediately; other commands finish their current safe step.")

    def _worker_finished(self):
        self._set_running(False)

    def _on_finished(self, result):
        if result.output:
            (self.append_output if result.success else self.append_error)(result.output)
        if result.actions.get("theme"):
            self.window().apply_theme(str(result.actions["theme"]))
        self.path_label.setText(str(self.shell.ctx.cwd))
        self.input.set_history(self.shell.ctx.history)
        if self.shell.ctx.history:
            self.history_added.emit(self.shell.ctx.history[-1])
        self.input.clear()
        self.input.setFocus()
        self.session_changed.emit()
        if result.exit_shell:
            self.close_requested.emit(self)

    def _set_running(self, running: bool):
        self.input.setEnabled(not running)
        self.run_button.setEnabled(not running)
        self.clear_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.state_label.setText("RUNNING" if running else "READY")
        self.session_changed.emit()

    def can_close(self) -> bool:
        if not self.is_busy:
            return True
        QMessageBox.warning(self, "Command still running", "Request stop, then wait for the command to finish before closing this session.")
        return False


class CopilotPanel(QFrame):
    command_requested = Signal(str)

    def __init__(self, session_provider, parent=None):
        super().__init__(parent)
        self.setObjectName("inspector")
        self.session_provider = session_provider
        self.worker: CopilotWorker | None = None
        self.pending_command = ""
        heading = QLabel("COPILOT")
        heading.setObjectName("sectionTitle")
        self.status = QLabel("Plan first. You stay in control.")
        self.status.setWordWrap(True)
        self.conversation = QTextEdit()
        self.conversation.setReadOnly(True)
        self.conversation.setMinimumHeight(200)
        self.input = QLineEdit()
        self.input.setPlaceholderText("Ask: ‘show Python files’")
        self.plan_button = QPushButton("Create plan")
        self.plan_button.setObjectName("primaryButton")
        self.approve_button = QPushButton("Approve & run")
        self.approve_button.setEnabled(False)
        self.reject_button = QPushButton("Dismiss")
        self.reject_button.setEnabled(False)
        actions = QHBoxLayout()
        actions.addWidget(self.plan_button)
        actions.addWidget(self.approve_button)
        actions.addWidget(self.reject_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.addWidget(heading)
        layout.addWidget(self.status)
        layout.addWidget(self.conversation, 1)
        layout.addWidget(self.input)
        layout.addLayout(actions)
        self.plan_button.clicked.connect(self.create_plan)
        self.input.returnPressed.connect(self.create_plan)
        self.approve_button.clicked.connect(self.approve)
        self.reject_button.clicked.connect(self.dismiss)

    def create_plan(self):
        prompt, session = self.input.text().strip(), self.session_provider()
        if not prompt or session is None or self.worker and self.worker.isRunning():
            return
        self._append("You", prompt)
        self.input.clear()
        self.status.setText("Planning with your current workspace context…")
        self.plan_button.setEnabled(False)
        self.worker = CopilotWorker(session.shell, prompt, self)
        self.worker.planned.connect(self._show_plan)
        self.worker.failed.connect(self._show_failure)
        self.worker.finished.connect(lambda: self.plan_button.setEnabled(True))
        self.worker.start()

    def _show_plan(self, action):
        self.pending_command = ""
        message = action.message or "No additional explanation."
        if action.action == "shell" and action.command:
            self.pending_command = action.command
            self._append("Plan", f"{message}\n\nProposed command:\n{action.command}")
            self.status.setText("Review the proposed command before approving it.")
            self.approve_button.setEnabled(True)
            self.reject_button.setEnabled(True)
        elif action.action == "code_write":
            self._append("Plan", f"{message}\n\nThis plan writes {len(action.files)} file(s). Use the existing Telegram approval flow to apply code-write actions.")
            self.status.setText("Code-write plans are intentionally approval-only here.")
        else:
            self._append("Copilot", message)
            self.status.setText("No command will run for this response.")

    def _show_failure(self, error: str):
        self._append("Copilot", f"Planning failed: {error}")
        self.status.setText("Could not reach the configured AI provider.")

    def _append(self, label: str, text: str):
        self.conversation.append(f"<b>{escape(label)}</b><br>{escape(text).replace(chr(10), '<br>')}<br>")

    def approve(self):
        if self.pending_command:
            self.command_requested.emit(self.pending_command)
            self._append("System", "Approved. Sending command to the active terminal.")
        self.dismiss()

    def dismiss(self):
        self.pending_command = ""
        self.approve_button.setEnabled(False)
        self.reject_button.setEnabled(False)
        self.status.setText("Plan first. You stay in control.")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("RiftShell", "RiftShell")
        self.preferences = self._load_preferences()
        self.theme = get_theme(self.preferences["theme"])
        self.history = self._load_history() if self.preferences["restore_history"] else []
        self.setWindowTitle("RiftShell — Developer Workspace")
        self.resize(1440, 880)
        self.setMinimumSize(1080, 680)
        self.setStyleSheet(build_stylesheet(self.theme))
        self.setFont(QFont(self.preferences["font_family"], self.preferences["font_size"]))
        self._build_workspace()
        self._install_shortcuts()
        self.new_session()
        self._refresh_workspace()

    def _load_preferences(self) -> dict:
        return {
            "theme": self.settings.value("appearance/theme", "vscode-dark-plus", type=str),
            "font_family": self.settings.value("appearance/font_family", "Cascadia Code", type=str),
            "font_size": self.settings.value("appearance/font_size", 11, type=int),
            "confirm_risky": self.settings.value("safety/confirm_risky", True, type=bool),
            "restore_history": self.settings.value("workspace/restore_history", True, type=bool),
        }

    def _load_history(self) -> list[str]:
        value = self.settings.value("workspace/history", [])
        return value if isinstance(value, list) else []

    def _build_workspace(self):
        root = QSplitter(Qt.Horizontal)
        root.setChildrenCollapsible(False)
        self.setCentralWidget(root)
        root.addWidget(self._build_sidebar())
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.setDocumentMode(True)
        self.tabs.currentChanged.connect(self._refresh_workspace)
        self.tabs.tabCloseRequested.connect(self.close_session_at)
        root.addWidget(self.tabs)
        self.copilot = CopilotPanel(self.current_session)
        self.copilot.command_requested.connect(self._run_copilot_command)
        root.addWidget(self.copilot)
        root.setSizes([250, 900, 300])
        self.root_splitter = root
        self.status_label = QLabel()
        self.statusBar().addWidget(self.status_label, 1)
        self.safety_label = QLabel("SAFE MODE")
        self.safety_label.setObjectName("statusPill")
        self.statusBar().addPermanentWidget(self.safety_label)

    def _build_sidebar(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("sidebar")
        panel.setMinimumWidth(220)
        panel.setMaximumWidth(310)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 18, 16, 16)
        layout.setSpacing(8)
        brand = QLabel("RiftShell")
        brand.setObjectName("brand")
        layout.addWidget(brand)
        subtitle = QLabel("DEVELOPER WORKSPACE")
        subtitle.setObjectName("eyebrow")
        layout.addWidget(subtitle)
        layout.addSpacing(16)
        self.new_button = QPushButton("+  New terminal")
        self.new_button.setObjectName("primaryButton")
        self.new_button.clicked.connect(self.new_session)
        layout.addWidget(self.new_button)
        for label, callback in [
            ("⌘  Command palette", self.open_palette), ("◷  Command history", self.show_history),
            ("◈  Command explorer", self.show_command_explorer), ("⚙  Workspace settings", self.open_settings),
        ]:
            button = QPushButton(label)
            button.setObjectName("navButton")
            button.clicked.connect(callback)
            layout.addWidget(button)
        layout.addSpacing(12)
        sessions_title = QLabel("SESSIONS")
        sessions_title.setObjectName("sectionTitle")
        layout.addWidget(sessions_title)
        self.session_list = QListWidget()
        self.session_list.setMaximumHeight(180)
        self.session_list.itemClicked.connect(self._activate_session_item)
        self.session_list.itemDoubleClicked.connect(self._rename_session_item)
        layout.addWidget(self.session_list)
        plugins_title = QLabel("EXTENSIONS")
        plugins_title.setObjectName("sectionTitle")
        layout.addWidget(plugins_title)
        self.plugin_summary = QLabel()
        self.plugin_summary.setWordWrap(True)
        layout.addWidget(self.plugin_summary)
        plugins_button = QPushButton("Manage plugins")
        plugins_button.setObjectName("navButton")
        plugins_button.clicked.connect(self.show_plugins)
        layout.addWidget(plugins_button)
        layout.addStretch()
        hint = QLabel("Ctrl+K palette  •  Ctrl+Shift+T new session")
        hint.setWordWrap(True)
        hint.setObjectName("sessionPath")
        layout.addWidget(hint)
        return panel

    def _install_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+K"), self, activated=self.open_palette)
        QShortcut(QKeySequence("Ctrl+Shift+T"), self, activated=self.new_session)
        QShortcut(QKeySequence("Ctrl+,"), self, activated=self.open_settings)
        QShortcut(QKeySequence("Ctrl+L"), self, activated=lambda: self.current_session() and self.current_session().clear_console())
        QShortcut(QKeySequence("Ctrl+Shift+C"), self, activated=lambda: self.current_session() and self.current_session().copy_output())

    def new_session(self, start_dir: Path | None = None, title: str | None = None):
        shell = Shell(start_dir=start_dir or Path.cwd())
        shell.ctx.current_theme = self.preferences["theme"]
        shell.ctx.history = self.history[-300:]
        session = TerminalSession(shell, title or f"Terminal {self.tabs.count() + 1}", self.preferences, self.tabs)
        session.history_added.connect(self.record_history)
        session.session_changed.connect(self._refresh_workspace)
        session.close_requested.connect(self.close_session)
        self.tabs.setCurrentIndex(self.tabs.addTab(session, session.title))
        self._refresh_workspace()
        session.input.setFocus()

    def current_session(self) -> TerminalSession | None:
        widget = self.tabs.currentWidget()
        return widget if isinstance(widget, TerminalSession) else None

    def close_session_at(self, index: int):
        session = self.tabs.widget(index)
        if isinstance(session, TerminalSession):
            self.close_session(session)

    def close_session(self, session: TerminalSession):
        if not session.can_close():
            return
        index = self.tabs.indexOf(session)
        if index < 0:
            return
        self.tabs.removeTab(index)
        session.deleteLater()
        if self.tabs.count() == 0:
            self.new_session()
        self._refresh_workspace()

    def _refresh_workspace(self, *_):
        session = self.current_session()
        self.session_list.blockSignals(True)
        self.session_list.clear()
        for index in range(self.tabs.count()):
            tab = self.tabs.widget(index)
            if not isinstance(tab, TerminalSession):
                continue
            item = QListWidgetItem(f"{'●' if tab.is_busy else '○'}  {tab.title}")
            item.setData(Qt.UserRole, index)
            item.setToolTip(str(tab.shell.ctx.cwd))
            self.session_list.addItem(item)
            if tab is session:
                self.session_list.setCurrentItem(item)
            self.tabs.setTabText(index, tab.title)
        self.session_list.blockSignals(False)
        if session:
            plugins = session.shell.ctx.plugins
            loaded = len(plugins.loaded) if plugins else 0
            failed = len(plugins.failed) if plugins else 0
            self.plugin_summary.setText(f"{loaded} loaded  •  {failed} needs attention")
            self.status_label.setText(f"{'running' if session.is_busy else 'ready'}  ·  {session.shell.ctx.cwd}  ·  {len(session.shell.registry.all_names())} commands")

    def _activate_session_item(self, item):
        self.tabs.setCurrentIndex(item.data(Qt.UserRole))

    def _rename_session_item(self, item):
        session = self.tabs.widget(item.data(Qt.UserRole))
        if not isinstance(session, TerminalSession):
            return
        value, accepted = QInputDialog.getText(self, "Rename session", "Session name", text=session.title)
        if accepted and value.strip():
            session.set_title(value.strip())

    def record_history(self, command: str):
        if command and (not self.history or self.history[-1] != command):
            self.history.append(command)
        self.history = self.history[-500:]
        self.settings.setValue("workspace/history", self.history)

    def open_palette(self):
        session = self.current_session()
        if not session:
            return
        palette = CommandPalette(session.shell, self)
        palette.command_chosen.connect(lambda command: self._insert_command(command))
        palette.action_chosen.connect(self._handle_palette_action)
        palette.exec()

    def _insert_command(self, command: str):
        session = self.current_session()
        if session:
            session.input.setText(command + " ")
            session.input.setFocus()

    def _handle_palette_action(self, action: str):
        callbacks = {
            "New terminal session": self.new_session, "Open settings": self.open_settings,
            "Toggle Copilot": self.toggle_copilot, "Show plugins": self.show_plugins,
        }
        if callback := callbacks.get(action):
            callback()

    def show_history(self):
        session = self.current_session()
        dialog = QDialog(self)
        dialog.setWindowTitle("Command history")
        dialog.resize(650, 480)
        search = QLineEdit()
        search.setPlaceholderText("Filter command history…")
        listing = QListWidget()
        use = QPushButton("Use command")
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.addWidget(search)
        layout.addWidget(listing, 1)
        layout.addWidget(use)
        def refresh(query=""):
            listing.clear()
            for entry in reversed(self.history):
                if fuzzy_score(query, entry) is not None:
                    listing.addItem(entry)
        def choose():
            if item := listing.currentItem():
                if session:
                    session.input.setText(item.text())
                    session.input.setFocus()
                dialog.accept()
        search.textChanged.connect(refresh)
        listing.itemDoubleClicked.connect(lambda _: choose())
        use.clicked.connect(choose)
        refresh()
        dialog.exec()

    def show_command_explorer(self):
        session = self.current_session()
        if not session:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Command explorer")
        dialog.resize(820, 560)
        search = QLineEdit()
        search.setPlaceholderText("Search commands, aliases, descriptions…")
        listing = QListWidget()
        details = QLabel("Select a command to inspect its usage and source.")
        details.setWordWrap(True)
        use = QPushButton("Use selected command")
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.addWidget(search)
        layout.addWidget(listing, 1)
        layout.addWidget(details)
        layout.addWidget(use)
        metas = session.shell.registry.list_metadata()
        def refresh(query=""):
            listing.clear()
            ranked = []
            for meta in metas:
                score = fuzzy_score(query, f"{meta.name} {' '.join(meta.aliases)} {meta.description}")
                if score is not None:
                    ranked.append((score, meta))
            for _, meta in sorted(ranked, key=lambda result: result[0]):
                item = QListWidgetItem(f"{meta.name}  —  {meta.description}")
                item.setData(Qt.UserRole, meta)
                listing.addItem(item)
        def describe(item):
            if item:
                meta = item.data(Qt.UserRole)
                details.setText(f"{meta.description}\n\nUsage: {meta.usage}\nAliases: {', '.join(meta.aliases) or 'None'}\nSource: {'Plugin: ' + meta.plugin if meta.plugin else 'Built-in command'}")
        def use_command():
            if item := listing.currentItem():
                self._insert_command(item.data(Qt.UserRole).name)
                dialog.accept()
        search.textChanged.connect(refresh)
        listing.currentItemChanged.connect(lambda current, _: describe(current))
        listing.itemDoubleClicked.connect(lambda _: use_command())
        use.clicked.connect(use_command)
        refresh()
        dialog.exec()

    def open_settings(self):
        dialog = PreferencesDialog(self.preferences, self)
        dialog.preferences_saved.connect(self.save_preferences)
        dialog.exec()

    def save_preferences(self, preferences: dict):
        self.preferences = preferences
        for key, value in preferences.items():
            group = "appearance" if key in {"theme", "font_family", "font_size"} else "safety" if key == "confirm_risky" else "workspace"
            self.settings.setValue(f"{group}/{key}", value)
        self.apply_theme(preferences["theme"])
        self.setFont(QFont(preferences["font_family"], preferences["font_size"]))
        for index in range(self.tabs.count()):
            if session := self.tabs.widget(index):
                session.apply_preferences(preferences)

    def apply_theme(self, theme_key: str):
        try:
            self.theme = get_theme(theme_key)
        except KeyError:
            return
        self.preferences["theme"] = self.theme.key
        self.settings.setValue("appearance/theme", self.theme.key)
        self.setStyleSheet(build_stylesheet(self.theme))
        for index in range(self.tabs.count()):
            if session := self.tabs.widget(index):
                session.shell.ctx.current_theme = self.theme.key
                session.apply_preferences(self.preferences)
        self._refresh_workspace()

    def show_plugins(self):
        session = self.current_session()
        if not session:
            return
        report = session.shell.ctx.plugins
        dialog = QDialog(self)
        dialog.setWindowTitle("Plugin manager")
        dialog.resize(650, 430)
        title = QLabel("PLUGINS & EXTENSIONS")
        title.setObjectName("eyebrow")
        text = QTextEdit()
        text.setReadOnly(True)
        lines = ["Loaded plugins"]
        lines.extend(f"• {item.name}  v{item.version}\n  {item.description}" for item in report.loaded) if report and report.loaded else lines.append("No plugins loaded.")
        lines.append("\nFailed plugins")
        lines.extend(f"• {item.name}\n  {item.error}" for item in report.failed) if report and report.failed else lines.append("None — all discovered plugins loaded correctly.")
        lines.append("\nPlugin enable/disable is determined at launch. Restart RiftShell after changing plugin files.")
        text.setPlainText("\n".join(lines))
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.addWidget(title)
        layout.addWidget(text, 1)
        dialog.exec()

    def toggle_copilot(self):
        self.copilot.setVisible(not self.copilot.isVisible())

    def _run_copilot_command(self, command: str):
        if session := self.current_session():
            session.input.setText(command)
            session.run_command()

    def closeEvent(self, event):
        for index in range(self.tabs.count()):
            if session := self.tabs.widget(index):
                if session.is_busy:
                    QMessageBox.warning(self, "Command still running", "Finish or cancel active commands before closing RiftShell.")
                    event.ignore()
                    return
        self.settings.sync()
        event.accept()
