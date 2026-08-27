# RiftShell

RiftShell is a Python-based custom shell and desktop-style command environment for Windows. It combines a dark cyberpunk UI with a flexible command system, plugin architecture, and optional AI-powered workflow layer.

The goal is simple: keep the shell familiar enough to use, but make it extensible enough for developers to add their own commands, tools, and automation without fighting the core app.

## What this project is

RiftShell is made of a few simple layers:

- Shell runtime: parses user input and executes commands
- Command registry: stores built-in and plugin commands
- UI layer: PySide6-based terminal window with suggestions/history
- Plugin system: lets you add new features without editing the core code
- AI layer: optional Telegram bot that converts plain language requests into shell actions

This is not a traditional Windows terminal clone. It is a developer-friendly command shell designed to be extended and customized.

## Quick start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run the shell

```bash
python main.py
```

If `python` is not found on your system, try:

```bash
py main.py
```

## Core architecture

### Shell engine
The main runtime starts from `main.py` and creates the PySide6 window. The shell reads command text, routes it through a parser, matches a command in the registry, and executes it in a controlled way.

### Plugin system
Plugins live under the `plugins/` folder. Each plugin exposes a `plugin` object that can register commands into the shell. The loader dynamically imports plugin files and wires them into the command registry.

This keeps the project modular and easy to expand.

### UI layer
The interface is built with PySide6 and provides:

- terminal output area
- command input box
- history navigation
- tab completion
- live suggestions
- dark neon theme
- non-blocking execution for heavier commands

### AI layer
The AI layer is optional and sits on top of the shell rather than replacing it.

- Entry point: `run_ai_bot.py`
- Config: `ai/config.py`
- Telegram bot logic: `ai/telegram_bot.py`
- Model integration: `ai/llm.py`

This layer allows natural-language commands such as:

```text
show files
what is the current folder
list processes
capture screenshot
```

The AI can trigger shell actions, but dangerous actions are gated by approval flows and workspace restrictions.

## Example commands

```text
files
where
goto C:\Users
makefolder demo
zip demo backup.zip
ip
processes
calc 5 + 7 * 2
echo hello > note.txt
files | filter py | count
setvar PROJECT RiftShell
echo $PROJECT
run python --version
exit
```

You can also chain commands:

```text
where ; files
makefolder logs && echo created
read missing.txt || echo fallback
files | sort | take 5
```

## Project layout

```text
RiftShell/
├── main.py                  # App entry point
├── run_ai_bot.py            # Optional AI bot runner
├── core/                    # Shell core, parser, registry, plugin loader
├── commands/                # Default command definitions
├── plugins/                 # Plugin modules
├── ai/                      # AI layer and Telegram integration
├── ui/                      # PySide6 UI components
├── utils/                   # Helper utilities
├── assets/                  # UI images/resources
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── LICENSE
```

## Developer-friendly notes

- The shell is built to be modular, not monolithic.
- New features are usually added as plugins instead of modifying the main shell logic.
- The command system is centralized so commands can be reused across the UI and AI layer.
- The AI layer is intentionally separate, so the shell remains useful even without the bot enabled.
- Safety checks are important for file actions, process operations, and external commands.

## AI setup

Create a `.env` file in the project root with values like:

```env
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_ALLOWED_USER_IDS=your_numeric_telegram_user_id
AI_PROVIDER=auto
AI_PROVIDER_ORDER=gemini,groq,ollama
GEMINI_API_KEY=your_gemini_key
AI_WORKSPACE_ROOT=D:\riftshell
AI_ALLOW_OUTSIDE_WORKSPACE=false
```

### AI model providers

Orbit supports Gemini, Groq, and local Ollama models without changing its command routing, memory, or safety logic.

Select one provider explicitly:

```env
AI_PROVIDER=gemini
```

```env
AI_PROVIDER=groq
```

```env
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_TIMEOUT_SECONDS=120
```

For Ollama, install and start Ollama first, then download the exact model configured above:

```text
ollama pull qwen2.5:7b
```

No additional Python package is required for Ollama; RiftShell uses its local HTTP API.

When multiple providers are configured, `AI_PROVIDER` removes ambiguity:

- `gemini`, `groq`, or `ollama` uses only that provider and does not silently send the request elsewhere.
- `auto` tries configured providers in `AI_PROVIDER_ORDER` and falls back only when a provider is unavailable.
- The default auto order is `gemini,groq,ollama`, preserving the existing cloud-provider behavior.

To prefer local inference while retaining cloud fallback:

```env
AI_PROVIDER=auto
AI_PROVIDER_ORDER=ollama,gemini,groq
```

To guarantee that Orbit never falls back to a cloud provider, use `AI_PROVIDER=ollama` rather than `auto`. With a localhost `OLLAMA_BASE_URL`, prompts stay on the local machine.

Then run:

```bash
python run_ai_bot.py
```

### AI behavior

The AI layer is designed to help with shell tasks using plain language, while still respecting safety rules:

- file and folder access can be limited to a workspace
- dangerous commands require approval
- system-level actions are checked before execution
- the bot acts as an assistant layer, not as a replacement for the core shell

Orbit communicates in professional English. File understanding is separate from
the shell's raw `read` command: requests such as `Explain README.md`, `Review
ai/safety.py for bugs`, or `Explain the structure of this project` load bounded
workspace context and return an explanation in the Orbit panel. Sensitive files
such as `.env`, private keys, AI memory, binary files, dependency folders, and
backup folders are not included in project inspection context.

When a review produces file changes, the desktop app shows a unified diff and
requires one explicit approval before writing. Existing files are backed up under
`.ai_backups/` before the approved replacement is applied. Use `/cmd read <file>`
when raw file content in the terminal is explicitly desired.

## Extending the project

If you want to add a new command, create a plugin under `plugins/<plugin_name>/plugin.py` and expose a `plugin` instance.

Example shape:

```python
from core.base import BaseCommand, CommandResult
from core.plugin_base import BasePlugin


class MyCommand(BaseCommand):
    name = "mycommand"
    aliases = []
    description = "My custom command"
    usage = "mycommand <args>"

    def execute(self, ctx, args):
        return CommandResult(output="Hello from RiftShell!")


class MyPlugin(BasePlugin):
    name = "myplugin"
    version = "1.0.0"
    description = "Adds my custom command"

    def register(self, registry):
        registry.register(MyCommand())


plugin = MyPlugin()
```

This keeps the code organized and allows the shell, help system, and AI integration to recognize the command automatically.

## Notes

RiftShell is meant to be practical and extensible. It is especially useful for developers who want:

- a custom desktop-style shell
- Windows-friendly system commands
- safe command wrappers
- a plugin-based architecture
- a simple AI assistant layer on top of their shell

The project is intentionally easy to understand and extend, while still being powerful enough for real workflows.

Run `plugins` inside RiftShell to see loaded and failed plugins. The AI Telegram layer reads the same registry catalog, so plugin commands become available to natural-language planning after restart.

## Future ideas
Possible next upgrades:
* autocomplete dropdown refinement
* command palette
* multi-tab terminal sessions
* plugin settings and enable/disable switches
* richer AI assistant workflows
* sound effects and typing animation

## License
MIT License.

## Author
Sanu Sharma - (sanusharma.dev) ❤️
