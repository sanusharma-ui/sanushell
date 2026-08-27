from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ai.actions import AgentAction
from ai.config import AIConfig
from core.parser import CommandParser


def _decode_json_value(text: str):
    """Decode a model JSON response, tolerating fences and leading prose."""
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            return None


def _extract_json(text: str) -> dict:
    """Return the invisible action envelope without leaking it into Orbit chat."""
    original = text.strip()
    value = _decode_json_value(original)

    # Some models double-encode the complete action object as a JSON string.
    for _ in range(2):
        if not isinstance(value, str):
            break
        decoded = _decode_json_value(value)
        if decoded is None:
            break
        value = decoded

    if not isinstance(value, dict):
        return {"action": "respond", "message": original or "No response from model."}

    # A common failure mode is putting another complete action envelope inside
    # `message`. Unwrap it so users see the answer, not Orbit's transport JSON.
    if str(value.get("action", "")).lower() == "respond":
        message = value.get("message")
        if isinstance(message, str):
            nested = _decode_json_value(message)
            if isinstance(nested, dict) and "action" in nested:
                value = nested

    return value


class MemoryManager:
    def __init__(self, path: str, limit: int, enabled: bool):
        self.path = Path(path)
        self.limit = limit
        self.enabled = enabled

    def load(self) -> list[dict]:
        if not self.enabled or not self.path.exists():
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

    def add(self, role: str, content: str) -> None:
        if not self.enabled:
            return
        history = self.load()
        history.append({"role": role, "content": content})

        # Multiply limit by 2 because 1 turn = 1 user message + 1 assistant message
        max_messages = self.limit * 2
        if len(history) > max_messages:
            history = history[-max_messages:]

        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def format_for_prompt(self) -> str:
        history = self.load()
        if not history:
            return "No previous memory."
        return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])


@dataclass
class AgentPlanner:
    config: AIConfig
    command_names: list[str]
    command_catalog: list[str]
    current_dir_provider: Callable[[], Path] | None = None
    last_output_provider: Callable[[], str] | None = None

    _NO_ARGUMENT_COMMANDS = {
        "help", "plugins", "exit", "quit", "clear", "cls", "where", "pwd",
        "up", "home", "processes", "system", "me", "pc", "today",
        "now", "history", "env", "version", "drives", "ip", "netstat", "path",
        "memory", "uptime", "vars", "last", "gstatus", "gbranch", "glog",
        "gdiff", "gpull", "gpush", "gremote", "hello",
    }

    def __post_init__(self) -> None:
        # 1. Memory Setup using env vars or config
        memory_enabled = str(getattr(self.config, "ai_memory_enabled", os.getenv("AI_MEMORY_ENABLED", "true"))).lower() == "true"
        memory_path = getattr(self.config, "ai_memory_path", os.getenv("AI_MEMORY_PATH", ".riftshell_ai_memory.json"))
        memory_turns = int(getattr(self.config, "ai_memory_recent_turns", os.getenv("AI_MEMORY_RECENT_TURNS", 12)))

        self.memory = MemoryManager(path=memory_path, limit=memory_turns, enabled=memory_enabled)

        self._selected_provider = str(getattr(self.config, "ai_provider", os.getenv("AI_PROVIDER", "auto"))).strip().lower()
        configured_order = getattr(self.config, "ai_provider_order", None)
        if configured_order:
            self._provider_order = tuple(str(item).strip().lower() for item in configured_order)
        else:
            self._provider_order = tuple(
                item.strip().lower()
                for item in os.getenv("AI_PROVIDER_ORDER", "gemini,groq,ollama").split(",")
                if item.strip()
            )

        self._gemini_key = str(getattr(self.config, "gemini_api_key", "") or "").strip()
        self._groq_key = str(getattr(self.config, "groq_api_key", os.getenv("GROQ_API_KEY", "")) or "").strip()
        self._ollama_model = str(getattr(self.config, "ollama_model", os.getenv("OLLAMA_MODEL", "")) or "").strip()
        self._ollama_base_url = str(
            getattr(self.config, "ollama_base_url", os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
            or "http://127.0.0.1:11434"
        ).strip().rstrip("/")
        self._ollama_timeout = max(
            1,
            int(getattr(self.config, "ollama_timeout_seconds", os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")) or 120),
        )

        # 2. Gemini Setup
        self._gemini = None
        self._gemini_setup_error = ""
        if self._gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self._gemini_key)
                self._gemini = genai.GenerativeModel(
                    getattr(self.config, "gemini_model", "gemini-2.5-flash"),
                    generation_config={"response_mime_type": "application/json"},
                )
            except Exception as exc:
                self._gemini_setup_error = str(exc)

        # 3. Groq Setup
        self._groq = None
        self._groq_setup_error = ""
        self._groq_model = str(
            getattr(self.config, "groq_model", os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
            or "llama-3.3-70b-versatile"
        ).strip()
        if self._groq_key:
            try:
                import groq
                self._groq = groq.Groq(api_key=self._groq_key)
            except Exception as exc:
                self._groq_setup_error = str(exc)

    def plan(self, user_text: str) -> AgentAction:
        direct = self._fallback_plan(user_text)
        if direct.action != "respond" or direct.message or not self._model_routing_enabled():
            return self._remember_action(user_text, self._validate_action(direct))

        prompt = self._build_prompt(user_text)
        action = None
        errors: list[str] = []

        try:
            providers = self._provider_sequence()
        except ValueError as exc:
            providers = []
            errors.append(str(exc))

        for provider in providers:
            try:
                text = self._call_provider(provider, prompt)
                action = AgentAction.from_payload(_extract_json(text))
                break
            except Exception as exc:
                errors.append(f"{provider.title()}: {exc}")

        # If both fail
        if action is None:
            details = " | ".join(errors) or "No configured provider is available."
            return self._remember_action(user_text, AgentAction(
                action="respond",
                message=f"I could not reach the selected AI provider.\nDetails: {details}",
            ))

        return self._remember_action(user_text, self._validate_action(action))

    def _model_routing_enabled(self) -> bool:
        if self._selected_provider != "auto":
            return True
        return bool(self._provider_sequence())

    def _provider_sequence(self) -> list[str]:
        valid = {"gemini", "groq", "ollama"}
        if self._selected_provider != "auto":
            if self._selected_provider not in valid:
                raise ValueError(
                    f"Invalid AI provider '{self._selected_provider}'. Choose auto, gemini, groq, or ollama."
                )
            return [self._selected_provider]

        invalid = [provider for provider in self._provider_order if provider not in valid]
        if invalid:
            raise ValueError(f"Invalid provider in AI_PROVIDER_ORDER: {invalid[0]}")
        return [provider for provider in self._provider_order if self._provider_is_configured(provider)]

    def _provider_is_configured(self, provider: str) -> bool:
        return {
            "gemini": bool(self._gemini_key),
            "groq": bool(self._groq_key),
            "ollama": bool(self._ollama_model),
        }.get(provider, False)

    def _call_provider(self, provider: str, prompt: str) -> str:
        if provider == "gemini":
            return self._call_gemini(prompt)
        if provider == "groq":
            return self._call_groq(prompt)
        if provider == "ollama":
            return self._call_ollama(prompt)
        raise ValueError(f"Unsupported AI provider: {provider}")

    def _call_gemini(self, prompt: str) -> str:
        if not self._gemini_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        if self._gemini is None:
            detail = f" ({self._gemini_setup_error})" if self._gemini_setup_error else ""
            raise RuntimeError(f"Gemini client is unavailable{detail}.")
        response = self._gemini.generate_content(prompt)
        text = getattr(response, "text", "") or ""
        if not text.strip():
            raise RuntimeError("Gemini returned an empty response.")
        return text

    def _call_groq(self, prompt: str) -> str:
        if not self._groq_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")
        if self._groq is None:
            detail = f" ({self._groq_setup_error})" if self._groq_setup_error else ""
            raise RuntimeError(f"Groq client is unavailable{detail}.")
        response = self._groq.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            model=self._groq_model,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content or ""
        if not text.strip():
            raise RuntimeError("Groq returned an empty response.")
        return text

    def _call_ollama(self, prompt: str) -> str:
        if not self._ollama_model:
            raise RuntimeError("OLLAMA_MODEL is not configured.")

        endpoint = f"{self._ollama_base_url}/api/generate"
        payload = json.dumps({
            "model": self._ollama_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        }).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._ollama_timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail or exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama is not reachable at {self._ollama_base_url}. Start Ollama and confirm OLLAMA_BASE_URL."
            ) from exc

        try:
            response_payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned an invalid HTTP response.") from exc
        if response_payload.get("error"):
            raise RuntimeError(str(response_payload["error"]))
        text = str(response_payload.get("response", ""))
        if not text.strip():
            raise RuntimeError("Ollama returned an empty response.")
        return text

    def _remember_action(self, user_text: str, action: AgentAction) -> AgentAction:
        self.memory.add("user", user_text)
        summary = f"Action: {action.action}; Command: {action.command or '(none)'}; Response: {action.message}"
        self.memory.add("assistant", summary)
        return action

    def _validate_action(self, action: AgentAction) -> AgentAction:
        """Never let Orbit present conversational text as an executable command."""
        if action.action != "shell":
            return action

        command = action.command.strip()
        try:
            parts = CommandParser().parse_line(command)
        except Exception:
            parts = []

        known_names = {name.lower() for name in self.command_names}
        is_valid = bool(parts) and all(
            parsed.name.lower() in known_names
            and not (parsed.name.lower() in self._NO_ARGUMENT_COMMANDS and parsed.args)
            for part in parts
            for parsed in part.pipeline.commands
        )
        if is_valid:
            return action

        return AgentAction(
            action="respond",
            message=(
                action.message
                if action.message and len(action.message.split()) >= 6
                else "I could not map that request to a valid RiftShell command. Please clarify the computer action you want me to perform."
            ),
        )

    def _fallback_plan(self, user_text: str) -> AgentAction:
        text = user_text.strip()
        lowered = text.lower()
        words = lowered.split()
        first_word = words[0] if words else ""
        command_names = {name.lower() for name in self.command_names}

        if not text:
            return AgentAction(
                action="respond",
                message="Tell me what you need. I can answer questions, explain things, or run RiftShell commands when you ask for computer work.",
            )

        if self._is_greeting(lowered):
            if self._model_routing_enabled():
                return AgentAction(action="respond", message="")
            return AgentAction(
                action="respond",
                message="Hello. I am Orbit, your RiftShell workspace assistant. How can I help today?",
            )

        if self._is_simple_general_question(lowered):
            if self._model_routing_enabled():
                return AgentAction(action="respond", message="")
            return AgentAction(action="respond", message=self._answer_simple_general_question(lowered))

        if self._asks_where_output_is(lowered):
            return AgentAction(
                action="respond",
                message=(
                    "Command output appears in the active terminal panel. I also post a completion note here in Orbit. "
                    "If you want, ask me to explain or summarize the latest terminal output."
                ),
            )

        local_action = self._route_workspace_request(text, lowered)
        if local_action is not None:
            return local_action

        if lowered.startswith("/cmd "):
            return AgentAction(action="shell", command=text[5:].strip(), message="I will run the requested RiftShell command after your approval.")

        if len(words) == 1 and first_word in command_names:
            return AgentAction(action="shell", command=text, message="I will run that RiftShell command after your approval.")

        if not self._model_routing_enabled():
            return self._offline_response(text)
        return AgentAction(action="respond", message="")

    @staticmethod
    def _shell(command: str, message: str) -> AgentAction:
        return AgentAction(action="shell", command=command, message=message)

    def _route_workspace_request(self, text: str, lowered: str) -> AgentAction | None:
        """Fast, offline routing for the common English ways people describe shell work."""
        normalized = re.sub(r"\s+", " ", lowered).strip(" .?!")

        if "screenshot" in normalized or "screen shot" in normalized or ("screen" in normalized and "photo" in normalized):
            return AgentAction(action="screenshot", message="I will capture a screenshot.")

        if match := re.match(r"(?:please\s+)?(?:execute|run)\s+(?:the\s+)?(.+?)(?:\s+command)?$", text, flags=re.IGNORECASE):
            requested = match.group(1).strip()
            routed = self._route_workspace_request(requested, requested.lower())
            if routed is not None:
                return routed
            if requested.split(maxsplit=1)[0].lower() in {name.lower() for name in self.command_names}:
                return self._shell(requested, "I will run the requested RiftShell command.")

        if normalized in {"help", "commands", "command list", "show commands", "list commands"}:
            return self._shell("help", "I will show the available RiftShell commands.")

        if normalized in {"where", "pwd", "where am i", "where we are", "where are we", "current directory", "current folder", "working directory", "show current directory", "show current folder", "what folder am i in", "what directory am i in", "show my location"}:
            return self._shell("where", "I will show the current workspace directory.")
        if re.fullmatch(r"(?:show|display)(?:\s+me)?\s+(?:the\s+)?(?:current|working)\s+(?:directory|folder)", normalized):
            return self._shell("where", "I will show the current workspace directory.")

        if normalized in {"ls", "dir", "files", "list files", "show files", "display files", "show me the files", "list the files", "what files are here", "what is in this folder", "what's in this folder", "show folder contents", "list folder contents"}:
            return self._shell("files", "I will list the files and folders in the current directory.")

        if normalized in {"folders", "list folders", "show folders", "show directories", "list directories", "only folders"}:
            return self._shell("folders", "I will list the folders in the current directory.")

        if normalized in {"up", "go up", "go back", "parent folder", "go to parent folder"}:
            return self._shell("up", "I will move to the parent directory.")
        if normalized in {"home", "go home", "go to home folder", "go to my home folder"}:
            return self._shell("home", "I will open your home directory in this session.")

        if match := re.match(r"(?:go to|change to|switch to|open directory|navigate to)\s+(.+)$", text, flags=re.IGNORECASE):
            return self._shell(f"cd {match.group(1).strip()}", "I will change this session to the requested directory.")

        if match := re.match(r"(?:create|make|new)\s+(?:a\s+)?(?:folder|directory)\s+(?:named\s+)?(.+)$", text, flags=re.IGNORECASE):
            return self._shell(f"makefolder {match.group(1).strip()}", "I will create the requested folder.")
        if match := re.match(r"(?:create|make|new)\s+(?:an?\s+)?file\s+(?:named\s+)?(.+)$", text, flags=re.IGNORECASE):
            return self._shell(f"makefile {match.group(1).strip()}", "I will create the requested file.")

        if match := re.match(r"(?:show|list|display)\s+(?:all\s+)?(.+?)\s+files?$", text, flags=re.IGNORECASE):
            kind = match.group(1).strip().lower()
            extensions = {
                "python": "py", "py": "py", "javascript": "js", "typescript": "ts",
                "json": "json", "markdown": "md", "text": "txt", "image": "png",
            }
            if extension := extensions.get(kind):
                return self._shell(f"files | filter {extension}", f"I will show the {kind} files in the current directory.")

        if match := re.match(r"(?:find|search)\s+(?:text\s+)?['\"](.+?)['\"]\s+(?:in|inside)\s+(.+)$", text, flags=re.IGNORECASE):
            return self._shell(f"findtext {match.group(1).strip()} {match.group(2).strip()}", "I will search that file for the requested text.")

        if match := re.match(r"(?:open|launch)\s+(?:the\s+)?(.+)$", text, flags=re.IGNORECASE):
            return self._shell(f"open {match.group(1).strip()}", "I will open the requested file or folder.")

        if match := re.match(r"(?:show|read|display)\s+(?:the\s+)?first\s+(\d+)\s+lines?\s+(?:of|from)\s+(?:the\s+)?(?:file\s+)?(.+)$", text, flags=re.IGNORECASE):
            return self._shell(f"head {match.group(2).strip()} {match.group(1)}", "I will show the requested first lines of the file.")
        if match := re.match(r"(?:show|read|display)\s+(?:the\s+)?last\s+(\d+)\s+lines?\s+(?:of|from)\s+(?:the\s+)?(?:file\s+)?(.+)$", text, flags=re.IGNORECASE):
            return self._shell(f"tail {match.group(2).strip()} {match.group(1)}", "I will show the requested last lines of the file.")
        if match := re.match(r"(?:count|show)\s+(?:the\s+)?(?:lines|words|characters|stats)\s+(?:in|of)\s+(?:the\s+)?(?:file\s+)?(.+)$", text, flags=re.IGNORECASE):
            return self._shell(f"wc {match.group(1).strip()}", "I will count the file's lines, words, and characters.")

        if match := re.match(r"read\s+(?:the\s+)?(?:contents?\s+of\s+)?(?:file\s+)?(.+)$", text, flags=re.IGNORECASE):
            target = match.group(1).strip()
            incomplete_targets = {"files", "folders", "directory", "current directory", "current folder", "first lines", "last lines"}
            if re.fullmatch(r"(?:the\s+)?(?:first|last)\s+\d+\s+lines?", target, flags=re.IGNORECASE):
                return AgentAction(action="respond", message="Please tell me which file you would like me to inspect.")
            if target and target.lower() not in incomplete_targets:
                return self._shell(f"read {target}", "I will read the requested file.")

        if match := re.match(r"(?:show|display)\s+(?:the\s+)?(?:contents?\s+of|file)\s+(.+)$", text, flags=re.IGNORECASE):
            return self._shell(f"read {match.group(1).strip()}", "I will read the requested file.")

        if match := re.match(r"(?:find|locate|where is)\s+(?:the\s+)?(?:command|executable|program)\s+(.+)$", text, flags=re.IGNORECASE):
            return self._shell(f"which {match.group(1).strip()}", "I will locate the requested executable.")

        if match := re.match(r"(?:find|search for|look for)\s+(?:a\s+)?(?:file|folder|directory)?\s*['\"]?(.+?)['\"]?$", text, flags=re.IGNORECASE):
            query = match.group(1).strip()
            if query:
                return self._shell(f"search {query}", "I will search for matching files and folders.")

        if match := re.match(r"(?:copy|duplicate)\s+(.+?)\s+(?:to|into)\s+(.+)$", text, flags=re.IGNORECASE):
            return self._shell(f"duplicate {match.group(1).strip()} {match.group(2).strip()}", "I will copy the requested item after your approval.")
        if match := re.match(r"(?:move|shift)\s+(.+?)\s+(?:to|into)\s+(.+)$", text, flags=re.IGNORECASE):
            return self._shell(f"shift {match.group(1).strip()} {match.group(2).strip()}", "I will move the requested item after your approval.")
        if match := re.match(r"(?:rename)\s+(.+?)\s+(?:to|as)\s+(.+)$", text, flags=re.IGNORECASE):
            return self._shell(f"rename {match.group(1).strip()} {match.group(2).strip()}", "I will rename the requested item after your approval.")
        if match := re.match(r"(?:delete|remove)\s+(?:the\s+)?(.+)$", text, flags=re.IGNORECASE):
            return self._shell(f"delete confirm {match.group(1).strip()}", "I will delete the requested item only after your approval and the built-in safety confirmation.")

        if normalized in {"processes", "show processes", "list processes", "running processes", "running tasks", "tasklist", "what is running"}:
            return self._shell("processes", "I will show the running processes.")
        if normalized in {"system", "system info", "system information", "computer info", "show system info"}:
            return self._shell("system", "I will show the system information.")
        if normalized in {"network", "network info", "network information", "show network info"}:
            return self._shell("network", "I will show the network information.")
        if normalized in {"ip", "ipconfig", "ip address", "show ip", "network config", "network configuration"}:
            return self._shell("ip", "I will show the IP configuration.")
        if normalized in {"connections", "network connections", "netstat", "active connections"}:
            return self._shell("netstat", "I will show the active network connections.")
        if normalized in {"memory", "ram", "ram usage", "memory usage"}:
            return self._shell("memory", "I will show current memory usage.")
        if normalized in {"uptime", "system uptime", "how long has the computer been on"}:
            return self._shell("uptime", "I will show the system uptime.")
        if normalized in {"drives", "show drives", "list drives"}:
            return self._shell("drives", "I will show the available drives.")
        if normalized in {"disk", "disk space", "disk usage", "show disk space", "show disk usage"}:
            return self._shell("disk", "I will show the available disk space for the current directory.")
        if normalized in {"path", "show path", "environment path"}:
            return self._shell("path", "I will show the active PATH value.")
        if normalized in {"environment", "environment variables", "show environment variables"}:
            return self._shell("env", "I will show the environment variables.")
        if normalized in {"history", "command history", "show history"}:
            return self._shell("history", "I will show this terminal session's command history.")
        if normalized in {"time", "current time", "what time is it", "show time"}:
            return self._shell("now", "I will show the current time.")
        if normalized in {"date", "today", "current date", "show date"}:
            return self._shell("today", "I will show today's date.")
        if normalized in {"who am i", "current user", "show current user", "my username"}:
            return self._shell("me", "I will show the signed-in user.")
        if normalized in {"computer name", "pc name", "machine name", "show pc name"}:
            return self._shell("pc", "I will show the computer name.")
        if normalized in {"version", "shell version", "riftshell version"}:
            return self._shell("version", "I will show the RiftShell version.")
        if normalized in {"plugins", "show plugins", "list plugins", "plugin status"}:
            return self._shell("plugins", "I will show the installed plugin status.")

        if match := re.match(r"(?:calculate|calc|what is)\s+([\d\s+*/().%-]+)$", normalized):
            return self._shell(f"calc {match.group(1).strip()}", "I will calculate that expression.")
        if match := re.match(r"(?:ping)\s+(.+)$", text, flags=re.IGNORECASE):
            return self._shell(f"ping {match.group(1).strip()}", "I will ping the requested host.")
        if match := re.match(r"(?:hash|checksum)\s+(.+)$", text, flags=re.IGNORECASE):
            return self._shell(f"hash {match.group(1).strip()}", "I will generate a hash for the requested text or file.")
        if match := re.match(r"(?:download)\s+(https?://\S+)(?:\s+(?:as|to)\s+(.+))?$", text, flags=re.IGNORECASE):
            filename = f" {match.group(2).strip()}" if match.group(2) else ""
            return self._shell(f"download {match.group(1)}{filename}", "I will download the requested file after your approval.")
        if match := re.match(r"(?:show|display)\s+(?:a\s+)?tree(?:\s+of)?\s*(.*)$", text, flags=re.IGNORECASE):
            suffix = match.group(1).strip()
            return self._shell(f"tree {suffix}".strip(), "I will show the directory tree.")
        if match := re.match(r"(?:switch|change|set)\s+(?:the\s+)?theme\s+(?:to\s+)?(.+)$", text, flags=re.IGNORECASE):
            return self._shell(f"theme {match.group(1).strip()}", "I will switch the workspace theme.")

        git_routes = {
            "git status": ("gstatus", "I will show the Git working tree status."),
            "git branches": ("gbranch", "I will show the Git branches."),
            "git log": ("glog", "I will show the recent Git commit history."),
            "git diff": ("gdiff", "I will show the unstaged Git changes."),
            "git remotes": ("gremote", "I will show the configured Git remotes."),
        }
        if normalized in git_routes:
            command, message = git_routes[normalized]
            return self._shell(command, message)

        return None

    def _is_greeting(self, lowered: str) -> bool:
        cleaned = re.sub(r"[^a-z0-9\s]", "", lowered).strip()
        words = cleaned.split()
        if not words:
            return False
        greetings = {
            "hello",
            "hi",
            "hey",
            "hii",
            "helo",
            "namaste",
            "namaskar",
            "salam",
            "yo",
        }
        if cleaned in greetings:
            return True
        return words[0] in greetings and len(words) <= 3

    def _looks_like_chat(self, lowered: str) -> bool:
        if lowered.endswith("?"):
            return True
        chat_starters = (
            "how ",
            "what ",
            "why ",
            "when ",
            "who ",
            "which ",
            "can you explain",
            "tell me",
            "explain ",
            "kya ",
            "kaise ",
            "kyu ",
            "batao ",
        )
        return lowered.startswith(chat_starters)

    @staticmethod
    def _asks_where_output_is(lowered: str) -> bool:
        normalized = re.sub(r"[^a-z0-9\s]", "", lowered)
        return bool(
            re.search(r"\bwhere\s+(?:is|did)\s+(?:the\s+)?(?:output|result|response)\b", normalized)
            or re.search(r"\b(?:show|see|find)\s+(?:the\s+)?(?:output|result)\b", normalized)
        )

    def _is_simple_general_question(self, lowered: str) -> bool:
        return bool(
            re.search(r"\bhow\s+many\s+continents\b", lowered)
            or re.search(r"\bcontinents\s+(are|in)\b", lowered)
        )

    def _answer_simple_general_question(self, lowered: str) -> str:
        if "continent" in lowered:
            return "There are 7 continents: Asia, Africa, North America, South America, Antarctica, Europe, and Australia/Oceania."
        return "I can answer that, but I need the question to be a little clearer."

    def _offline_response(self, text: str) -> AgentAction:
        return AgentAction(
            action="respond",
            message=(
                "I can help with the built-in RiftShell workspace requests locally. "
                "For broader reasoning and more complex phrasing, configure a Gemini or Groq API key. "
                "You can also use the Command Explorer to browse every available command."
            ),
        )

    def _build_prompt(self, user_text: str) -> str:
        catalog = "\n".join(f"- {item}" for item in self.command_catalog)
        history = self.memory.format_for_prompt()
        current_dir = self.current_dir_provider() if self.current_dir_provider else self.config.workspace_root
        asks_about_terminal_output = bool(
            re.search(r"\b(?:output|result|terminal|previous command|last command)\b", user_text, flags=re.IGNORECASE)
        )
        last_output = self.last_output_provider() if self.last_output_provider and asks_about_terminal_output else ""
        last_output = last_output[-3000:] if last_output else "(not included for this request)"
        access_mode = "FULL_PC" if self.config.allow_outside_workspace else "WORKSPACE_ONLY"

        return f"""
You are Orbit: a friendly, professional workspace assistant inside RiftShell.
You can chat naturally, answer general questions, clarify intent, and execute safe shell actions when the user clearly wants computer work.
Your output must always be STRICT, VALID JSON. This JSON is an internal transport envelope and is never shown to the user.

Allowed JSON shapes:
{{"action":"shell","command":"STRICT_COMMAND_HERE","message":"short explanation"}}
{{"action":"screenshot","message":"taking screenshot"}}
{{"action":"code_write","message":"writing code","files":[{{"path":"EXACT_GIVEN_PATH","content":"full code"}}]}}
{{"action":"respond","message":"chat or clarification in Markdown"}}

Runtime context:
- Current directory: {current_dir}
- AI workspace root: {self.config.workspace_root}
- Access mode: {access_mode}
- Interface: Orbit is a chat panel beside the active terminal. Shell command output is rendered in that terminal, not inside the chat panel.
- Latest terminal output (may be truncated):
{last_output}

UNIVERSAL RULES (READ AND OBEY):
1. ASSISTANT FIRST: Your primary role is a capable conversational assistant. General knowledge, explanations, planning, brainstorming, recommendations, coding guidance, follow-up questions, and casual conversation must return {{"action":"respond","message":"..."}}.
2. COMMAND INTENT: Use "shell" only when the user clearly asks to inspect or change the computer/workspace, run a listed command, navigate files, show system info, capture a screenshot, or execute a specific machine task. A word that happens to match a command name is not enough.
3. READ THE CATALOG: Never guess command syntax. Look at the "Available Commands" catalog below. Format shell commands EXACTLY as the catalog requires.
4. UNKNOWN COMMANDS: If the user asks for something that is not supported by the catalog, respond conversationally and explain what you can do instead. Never say "No such commands" for general chat.
5. LOCATION: Relative paths run from the Current directory above. Use `cd <path>` when the user asks to move to a location. Use absolute paths only when the user gives or clearly asks for one.
6. FULL-PC MODE: If Access mode is FULL_PC, commands and code_write may target any user-provided location on this PC. If Access mode is WORKSPACE_ONLY, stay inside the AI workspace root.
7. WINDOWS PROCESSES: If killing or finding a process, append `.exe` (e.g., `kill chrome.exe`).
8. EXACT FILE PATHS (CRITICAL): Use the EXACT filename and path the user provides. Escape backslashes (e.g., `D:\\j.txt`). Do NOT change the filename.
9. NO CHAT IN COMMAND: The "command" field MUST ONLY contain executable RiftShell syntax. Put explanation in "message".
10. PIPING: You can use `|` if the catalog supports it.
11. CODE WRITES: Use "code_write" only when the user explicitly asks you to create or rewrite files with code/content. Keep file paths exact and content complete.
12. TONE: Use clear, friendly, professional English. Keep explanations concise and specific. Do not pretend a command ran unless the action is "shell", "screenshot", or "code_write".
13. TRANSLATION: Convert natural-language requests into the closest valid command from the catalog. Never place the user's natural-language sentence in the "command" field unless it is already valid RiftShell syntax.
14. CONTEXT: Understand follow-ups using Recent Conversation History and Latest terminal output. If the user asks where command output is, explain that it appears in the active terminal; do not run `last` unless they explicitly ask to print the previous output again.
15. COLLISIONS: Words such as "now", "last", "path", "where", "history", and "help" may occur in ordinary conversation. Only treat them as commands when the overall sentence clearly requests a computer action.
16. PLANNING: When asked to make a plan, think through a problem, compare options, or provide advice, respond with a useful plan in natural language. Do not turn planning into shell commands unless the user explicitly asks you to inspect or modify the workspace.
17. RESPONSE FORMAT: For "respond" messages, use clean GitHub-Flavored Markdown when structure helps: short headings, blank lines, numbered or bulleted lists, and fenced code blocks with a language tag. Do not put the action JSON itself, or a second action envelope, inside "message". Avoid HTML and decorative clutter.

Routing examples:
- User: "hello" -> {{"action":"respond","message":"Hello. How can I help?"}}
- User: "How many continents are there?" -> {{"action":"respond","message":"There are 7 continents: Asia, Africa, North America, South America, Antarctica, Europe, and Australia/Oceania."}}
- User: "list files" -> {{"action":"shell","command":"files","message":"Listing files."}}
- User: "show my current folder" -> {{"action":"shell","command":"where","message":"Showing the current folder."}}
- User: "now tell me who Elon Musk is" -> {{"action":"respond","message":"Elon Musk is a technology entrepreneur..."}}
- User: "help me plan a Python project" -> {{"action":"respond","message":"Here is a practical project plan..."}}
- User: "where is the output?" -> {{"action":"respond","message":"The command output appears in the active terminal panel."}}
- User: "take a screenshot" -> {{"action":"screenshot","message":"Taking a screenshot."}}
- User: "write a Python script at hello.py that prints hello" -> {{"action":"code_write","message":"Preparing hello.py.","files":[{{"path":"hello.py","content":"print('hello')\n"}}]}}

Available Commands & Aliases:
{catalog}

Recent Conversation History:
{history}

History note: workspace action records describe what happened earlier; they are context, not examples of how the current request must be routed. Re-evaluate the current user's intent independently.

User request:
{user_text}
""".strip()
