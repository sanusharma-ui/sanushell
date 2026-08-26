from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_user_ids(value: str | None) -> set[int]:
    result: set[int] = set()
    if not value:
        return result

    for item in value.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            result.add(int(item))
        except ValueError:
            continue
    return result


VALID_AI_PROVIDERS = ("gemini", "groq", "ollama")


def _parse_provider(value: str | None) -> str:
    provider = (value or "auto").strip().lower()
    if provider not in {"auto", *VALID_AI_PROVIDERS}:
        choices = ", ".join(("auto", *VALID_AI_PROVIDERS))
        raise ValueError(f"Invalid AI_PROVIDER '{provider}'. Choose one of: {choices}")
    return provider


def _parse_provider_order(value: str | None) -> tuple[str, ...]:
    raw_items = (value or "gemini,groq,ollama").replace(";", ",").split(",")
    result: list[str] = []
    for raw_item in raw_items:
        item = raw_item.strip().lower()
        if not item:
            continue
        if item not in VALID_AI_PROVIDERS:
            choices = ", ".join(VALID_AI_PROVIDERS)
            raise ValueError(f"Invalid provider '{item}' in AI_PROVIDER_ORDER. Choose from: {choices}")
        if item not in result:
            result.append(item)
    if not result:
        raise ValueError("AI_PROVIDER_ORDER must contain at least one provider.")
    return tuple(result)


@dataclass(frozen=True)
class AIConfig:
    telegram_bot_token: str
    telegram_allowed_user_ids: set[int]
    telegram_allow_unlisted_users: bool
    gemini_api_key: str
    gemini_model: str
    groq_api_key: str
    groq_model: str
    ai_provider: str
    ai_provider_order: tuple[str, ...]
    ollama_base_url: str
    ollama_model: str
    ollama_timeout_seconds: int
    workspace_root: Path
    allow_outside_workspace: bool
    approval_timeout_minutes: int
    command_output_limit: int

    @classmethod
    def from_env(cls) -> "AIConfig":
        _load_dotenv(PROJECT_ROOT / ".env")

        workspace_text = os.getenv("AI_WORKSPACE_ROOT", str(PROJECT_ROOT))
        try:
            workspace_root = Path(workspace_text).expanduser().resolve(strict=False)
        except Exception:
            workspace_root = PROJECT_ROOT

        return cls(
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            telegram_allowed_user_ids=_parse_user_ids(os.getenv("TELEGRAM_ALLOWED_USER_IDS")),
            telegram_allow_unlisted_users=_parse_bool(
                os.getenv("TELEGRAM_ALLOW_UNLISTED_USERS"),
                default=False,
            ),
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(),
            groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
            groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip(),
            ai_provider=_parse_provider(os.getenv("AI_PROVIDER")),
            ai_provider_order=_parse_provider_order(os.getenv("AI_PROVIDER_ORDER")),
            ollama_base_url=(os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").strip().rstrip("/"),
            ollama_model=os.getenv("OLLAMA_MODEL", "").strip(),
            ollama_timeout_seconds=max(1, int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120") or "120")),
            workspace_root=workspace_root,
            allow_outside_workspace=_parse_bool(os.getenv("AI_ALLOW_OUTSIDE_WORKSPACE"), default=False),
            approval_timeout_minutes=int(os.getenv("AI_APPROVAL_TIMEOUT_MINUTES", "30") or "30"),
            command_output_limit=int(os.getenv("AI_COMMAND_OUTPUT_LIMIT", "3500") or "3500"),
        )

    def validate_for_telegram(self) -> list[str]:
        problems = []
        if not self.telegram_bot_token:
            problems.append("TELEGRAM_BOT_TOKEN is missing")
        if self.ai_provider == "gemini" and not self.gemini_api_key:
            problems.append("AI_PROVIDER is gemini but GEMINI_API_KEY is missing")
        elif self.ai_provider == "groq" and not self.groq_api_key:
            problems.append("AI_PROVIDER is groq but GROQ_API_KEY is missing")
        elif self.ai_provider == "ollama" and not self.ollama_model:
            problems.append("AI_PROVIDER is ollama but OLLAMA_MODEL is missing")
        elif self.ai_provider == "auto":
            configured = {
                "gemini": bool(self.gemini_api_key),
                "groq": bool(self.groq_api_key),
                "ollama": bool(self.ollama_model),
            }
            if not any(configured[provider] for provider in self.ai_provider_order):
                problems.append("No provider in AI_PROVIDER_ORDER is configured")
        if not self.telegram_allowed_user_ids and not self.telegram_allow_unlisted_users:
            problems.append("TELEGRAM_ALLOWED_USER_IDS is missing")
        return problems
