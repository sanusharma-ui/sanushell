import json
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ai.config import _parse_int
from ai.llm import AgentPlanner, _extract_json
from core.shell import Shell


class FakeGemini:
    def __init__(self, message="cloud response"):
        self.calls = 0
        self.message = message

    def generate_content(self, _prompt):
        self.calls += 1
        payload = {"action": "respond", "message": self.message}
        return SimpleNamespace(text=json.dumps(payload))


class FakeGroqCompletions:
    def __init__(self, owner):
        self.owner = owner

    def create(self, **_kwargs):
        self.owner.calls += 1
        message = SimpleNamespace(content=json.dumps({
            "action": "respond",
            "message": self.owner.message,
        }))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeGroq:
    def __init__(self, message="groq response"):
        self.calls = 0
        self.message = message
        self.chat = SimpleNamespace(completions=FakeGroqCompletions(self))


class FakeHTTPResponse:
    def __init__(self, message="local response"):
        model_payload = json.dumps({"action": "respond", "message": message})
        self.body = json.dumps({"response": model_payload}).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


def make_config(**overrides):
    values = {
        "ai_provider": "auto",
        "ai_provider_order": ("gemini", "groq", "ollama"),
        "gemini_api_key": "",
        "gemini_model": "gemini-test",
        "groq_api_key": "",
        "groq_model": "groq-test",
        "ollama_base_url": "http://127.0.0.1:11434",
        "ollama_model": "",
        "ollama_timeout_seconds": 5,
        "workspace_root": Path.cwd(),
        "allow_outside_workspace": False,
        "ai_memory_enabled": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_planner(config):
    shell = Shell()
    return AgentPlanner(config, shell.registry.all_names(), shell.registry.catalog_entries())


class ProviderSelectionTests(unittest.TestCase):
    def test_forced_ollama_uses_local_http_only(self):
        planner = make_planner(make_config(ai_provider="ollama", ollama_model="qwen-test"))
        cloud = FakeGemini()
        planner._gemini = cloud

        with patch("ai.llm.urllib.request.urlopen", return_value=FakeHTTPResponse()) as request:
            action = planner.plan("Explain dependency injection")

        self.assertEqual(action.action, "respond")
        self.assertEqual(action.message, "local response")
        self.assertEqual(cloud.calls, 0)
        sent_request = request.call_args.args[0]
        sent_payload = json.loads(sent_request.data.decode("utf-8"))
        self.assertEqual(sent_payload["model"], "qwen-test")
        self.assertFalse(sent_payload["stream"])

    def test_auto_falls_back_from_ollama_to_gemini(self):
        planner = make_planner(make_config(
            ai_provider="auto",
            ai_provider_order=("ollama", "gemini", "groq"),
            ollama_model="qwen-test",
            gemini_api_key="test-key",
        ))
        cloud = FakeGemini("fallback response")
        planner._gemini = cloud

        with patch(
            "ai.llm.urllib.request.urlopen",
            side_effect=urllib.error.URLError("offline"),
        ):
            action = planner.plan("Explain dependency injection")

        self.assertEqual(action.message, "fallback response")
        self.assertEqual(cloud.calls, 1)

    def test_forced_ollama_never_falls_back_to_cloud(self):
        planner = make_planner(make_config(
            ai_provider="ollama",
            ollama_model="qwen-test",
            gemini_api_key="test-key",
        ))
        cloud = FakeGemini()
        planner._gemini = cloud

        with patch(
            "ai.llm.urllib.request.urlopen",
            side_effect=urllib.error.URLError("offline"),
        ):
            action = planner.plan("Explain dependency injection")

        self.assertEqual(action.action, "respond")
        self.assertIn("Ollama is not reachable", action.message)
        self.assertEqual(cloud.calls, 0)

    def test_auto_skips_unconfigured_providers(self):
        planner = make_planner(make_config(
            ai_provider="auto",
            ai_provider_order=("ollama", "gemini", "groq"),
            gemini_api_key="test-key",
        ))
        cloud = FakeGemini()
        planner._gemini = cloud

        with patch("ai.llm.urllib.request.urlopen") as request:
            action = planner.plan("Explain dependency injection")

        self.assertEqual(action.message, "cloud response")
        self.assertEqual(cloud.calls, 1)
        request.assert_not_called()

    def test_forced_gemini_uses_only_gemini(self):
        planner = make_planner(make_config(
            ai_provider="gemini",
            gemini_api_key="test-key",
            groq_api_key="test-key",
        ))
        gemini = FakeGemini("gemini response")
        groq = FakeGroq()
        planner._gemini = gemini
        planner._groq = groq

        action = planner.plan("Explain dependency injection")

        self.assertEqual(action.message, "gemini response")
        self.assertEqual(gemini.calls, 1)
        self.assertEqual(groq.calls, 0)

    def test_forced_groq_uses_only_groq(self):
        planner = make_planner(make_config(
            ai_provider="groq",
            gemini_api_key="test-key",
            groq_api_key="test-key",
        ))
        gemini = FakeGemini()
        groq = FakeGroq()
        planner._gemini = gemini
        planner._groq = groq

        action = planner.plan("Explain dependency injection")

        self.assertEqual(action.message, "groq response")
        self.assertEqual(gemini.calls, 0)
        self.assertEqual(groq.calls, 1)


class ConfigParsingTests(unittest.TestCase):
    def test_integer_setting_tolerates_markdown_backticks(self):
        self.assertEqual(_parse_int("3500`   `", 100), 3500)

    def test_invalid_integer_setting_uses_safe_default(self):
        self.assertEqual(_parse_int("not-a-number", 120, minimum=1), 120)

    def test_integer_setting_respects_minimum(self):
        self.assertEqual(_parse_int("0", 120, minimum=1), 1)


class ResponseEnvelopeTests(unittest.TestCase):
    def test_nested_action_envelope_is_unwrapped(self):
        inner = {"action": "respond", "message": "## Plan\n\n1. Build API"}
        outer = {"action": "respond", "message": json.dumps(inner)}

        self.assertEqual(_extract_json(json.dumps(outer)), inner)

    def test_double_encoded_action_envelope_is_unwrapped(self):
        payload = {"action": "respond", "message": "**Clean answer**"}

        self.assertEqual(_extract_json(json.dumps(json.dumps(payload))), payload)

    def test_plain_markdown_becomes_a_response(self):
        markdown = "## Plan\n\n- Add routes"

        self.assertEqual(_extract_json(markdown), {"action": "respond", "message": markdown})


if __name__ == "__main__":
    unittest.main()
