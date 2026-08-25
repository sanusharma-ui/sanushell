from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    key: str
    display_name: str
    aliases: tuple[str, ...]
    background: str
    surface: str
    surface_alt: str
    border: str
    text: str
    muted: str
    accent: str
    accent_alt: str
    output: str
    error: str
    selection: str
    button: str
    button_hover: str
    button_pressed: str
    console_glow: str


THEMES: dict[str, Theme] = {
    "vscode-dark-plus": Theme(
        key="vscode-dark-plus",
        display_name="VS Code Dark+ (Default)",
        aliases=("default", "dark+", "dark-plus", "vscode", "vs-code", "vs-code-dark-plus"),
        background="#1e1e1e",
        surface="#252526",
        surface_alt="#2d2d30",
        border="#3e3e42",
        text="#d4d4d4",
        muted="#858585",
        accent="#569cd6",
        accent_alt="#4ec9b0",
        output="#b5cea8",
        error="#f48771",
        selection="#264f78",
        button="#333333",
        button_hover="#3c3c3c",
        button_pressed="#2a2d2e",
        console_glow="#000000",
    ),
    "tokyo-night": Theme(
        key="tokyo-night",
        display_name="Tokyo Night",
        aliases=("tokyo", "tokyonight", "tokyo-night"),
        background="#1a1b26",
        surface="#16161e",
        surface_alt="#24283b",
        border="#292e42",
        text="#c0caf5",
        muted="#9aa5ce",
        accent="#7aa2f7",
        accent_alt="#7dcfff",
        output="#9ece6a",
        error="#f7768e",
        selection="#33467c",
        button="#24283b",
        button_hover="#2f354d",
        button_pressed="#1f2335",
        console_glow="#000000",
    ),
    "nord": Theme(
        key="nord",
        display_name="Nord",
        aliases=("nord",),
        background="#2e3440",
        surface="#242933",
        surface_alt="#3b4252",
        border="#4c566a",
        text="#d8dee9",
        muted="#a7b1c2",
        accent="#88c0d0",
        accent_alt="#8fbcbb",
        output="#a3be8c",
        error="#bf616a",
        selection="#434c5e",
        button="#3b4252",
        button_hover="#434c5e",
        button_pressed="#2e3440",
        console_glow="#1f242d",
    ),
    "dracula": Theme(
        key="dracula",
        display_name="Dracula",
        aliases=("dracula",),
        background="#282a36",
        surface="#21222c",
        surface_alt="#343746",
        border="#44475a",
        text="#f8f8f2",
        muted="#b6b6c8",
        accent="#bd93f9",
        accent_alt="#8be9fd",
        output="#50fa7b",
        error="#ff5555",
        selection="#44475a",
        button="#343746",
        button_hover="#44475a",
        button_pressed="#282a36",
        console_glow="#191a21",
    ),
    "github-dark": Theme(
        key="github-dark",
        display_name="GitHub Dark",
        aliases=("github", "github-dark", "gh-dark"),
        background="#0d1117",
        surface="#010409",
        surface_alt="#161b22",
        border="#30363d",
        text="#c9d1d9",
        muted="#8b949e",
        accent="#58a6ff",
        accent_alt="#79c0ff",
        output="#7ee787",
        error="#ff7b72",
        selection="#1f6feb",
        button="#21262d",
        button_hover="#30363d",
        button_pressed="#161b22",
        console_glow="#000000",
    ),
}

DEFAULT_THEME_KEY = "vscode-dark-plus"

_ALIAS_TO_KEY = {
    alias.lower(): key
    for key, theme in THEMES.items()
    for alias in (theme.key, theme.display_name, *theme.aliases)
}


def normalize_theme_name(name: str) -> str:
    return "-".join(name.strip().lower().replace("+", " plus").split())


def get_theme(name: str | None = None) -> Theme:
    if not name:
        return THEMES[DEFAULT_THEME_KEY]

    raw = name.strip().strip('"\'')
    candidates = [
        raw.lower(),
        normalize_theme_name(raw),
        normalize_theme_name(raw).replace("-plus", "+"),
    ]
    for candidate in candidates:
        if candidate in THEMES:
            return THEMES[candidate]
        if candidate in _ALIAS_TO_KEY:
            return THEMES[_ALIAS_TO_KEY[candidate]]
    raise KeyError(name)


def list_themes() -> list[Theme]:
    return list(THEMES.values())


def build_stylesheet(theme: Theme) -> str:
    return f"""
QMainWindow {{
    background-color: {theme.background};
}}

QWidget {{
    background-color: {theme.background};
    color: {theme.text};
}}

QLabel {{
    color: {theme.muted};
    font-size: 10pt;
    font-family: Consolas;
}}

QTextEdit {{
    background-color: {theme.surface};
    color: {theme.text};
    border: 1px solid {theme.border};
    border-radius: 10px;
    padding: 10px;
    font-size: 11pt;
    selection-background-color: {theme.selection};
    selection-color: {theme.text};
}}

QLineEdit {{
    background-color: {theme.surface};
    color: {theme.text};
    border: 1px solid {theme.border};
    border-radius: 8px;
    padding: 9px 10px;
    font-size: 11pt;
    selection-background-color: {theme.selection};
    selection-color: {theme.text};
}}

QLineEdit:focus {{
    border: 1px solid {theme.accent};
}}

QListWidget {{
    background-color: {theme.surface};
    color: {theme.text};
    border: 1px solid {theme.border};
    border-radius: 8px;
    padding: 4px;
    font-size: 10pt;
}}

QListWidget::item {{
    padding: 7px 8px;
    border-radius: 10px;
}}

QListWidget::item:selected {{
    background-color: {theme.surface_alt};
    color: {theme.accent_alt};
}}

QPushButton {{
    background-color: {theme.button};
    color: {theme.text};
    border: 1px solid {theme.border};
    border-radius: 7px;
    padding: 8px 12px;
    font-weight: 600;
}}

QPushButton:hover {{
    background-color: {theme.button_hover};
    border: 1px solid {theme.accent};
    color: {theme.text};
}}

QPushButton:pressed {{
    background-color: {theme.button_pressed};
}}

QStatusBar {{
    background-color: {theme.background};
    color: {theme.accent_alt};
}}

QFrame#sidebar, QFrame#inspector, QFrame#terminalHeader {{
    background-color: {theme.surface};
    border: 1px solid {theme.border};
    border-radius: 10px;
}}

QFrame#sidebar {{
    border-radius: 0;
    border-top: 0;
    border-bottom: 0;
    border-left: 0;
}}

QLabel#brand {{
    color: {theme.accent_alt};
    font-size: 18pt;
    font-weight: 700;
}}

QLabel#eyebrow {{
    color: {theme.muted};
    font-size: 8pt;
    font-weight: 700;
    letter-spacing: 1px;
}}

QLabel#sectionTitle {{
    color: {theme.accent};
    font-size: 9pt;
    font-weight: 700;
}}

QLabel#sessionPath {{
    color: {theme.muted};
    font-size: 9pt;
}}

QLabel#statusPill {{
    background-color: {theme.surface_alt};
    color: {theme.accent_alt};
    border-radius: 9px;
    padding: 3px 8px;
    font-size: 8pt;
    font-weight: 700;
}}

QPushButton#primaryButton {{
    background-color: {theme.accent};
    color: {theme.background};
    border-color: {theme.accent};
}}

QPushButton#primaryButton:hover {{
    background-color: {theme.accent_alt};
    border-color: {theme.accent_alt};
}}

QPushButton#navButton {{
    background-color: transparent;
    border-color: transparent;
    text-align: left;
    padding: 8px;
    color: {theme.muted};
}}

QPushButton#navButton:hover, QPushButton#navButton:checked {{
    background-color: {theme.surface_alt};
    border-color: {theme.border};
    color: {theme.accent_alt};
}}

QTabWidget::pane {{
    border: 0;
    background: transparent;
}}

QTabBar::tab {{
    background: {theme.background};
    color: {theme.muted};
    border: 1px solid transparent;
    border-bottom: 2px solid transparent;
    padding: 9px 13px;
    margin-right: 3px;
}}

QTabBar::tab:selected {{
    background: {theme.surface};
    color: {theme.text};
    border-color: {theme.border};
    border-bottom-color: {theme.accent};
}}

QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 3px;
}}

QScrollBar::handle:vertical {{
    background: {theme.border};
    border-radius: 4px;
    min-height: 24px;
}}
"""
