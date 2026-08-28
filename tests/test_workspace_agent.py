import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

from ai.actions import AgentAction, FileWrite
from ai.code_writer import apply_file_writes, preview_file_writes
from ai.llm import AgentPlanner
from ai.workspace_reader import inspect_workspace_paths
from core.shell import Shell
from ui.main_window import pending_action_reply


class CapturingGemini:
    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    def generate_content(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(text=json.dumps(self.payload))


class SequencedGemini:
    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.prompts = []

    def generate_content(self, prompt):
        self.prompts.append(prompt)
        payload = self.payloads.pop(0)
        return SimpleNamespace(text=json.dumps(payload))


def make_config(workspace_root: Path):
    return SimpleNamespace(
        ai_provider="gemini",
        ai_provider_order=("gemini",),
        gemini_api_key="test-key",
        gemini_model="gemini-test",
        groq_api_key="",
        groq_model="groq-test",
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="",
        ollama_timeout_seconds=5,
        workspace_root=workspace_root,
        allow_outside_workspace=False,
        ai_memory_enabled=False,
    )


def workspace_temp():
    return tempfile.TemporaryDirectory(dir=Path.cwd())


def make_planner(workspace_root: Path, payload: dict):
    shell = Shell(start_dir=workspace_root)
    planner = AgentPlanner(
        make_config(workspace_root),
        shell.registry.all_names(),
        shell.registry.catalog_entries(),
        current_dir_provider=lambda: workspace_root,
    )
    model = CapturingGemini(payload)
    planner._gemini = model
    return planner, model


class WorkspaceInspectionTests(unittest.TestCase):
    def test_file_is_read_as_private_model_context(self):
        with workspace_temp() as temp:
            root = Path(temp)
            (root / "README.md").write_text("# Sample\nA useful project.", encoding="utf-8")
            planner, model = make_planner(
                root,
                {"action": "respond", "message": "This is a useful sample project."},
            )

            action = planner.plan("Explain README.md")

            self.assertEqual(action.action, "respond")
            self.assertEqual(action.message, "This is a useful sample project.")
            self.assertEqual(len(model.prompts), 1)
            self.assertIn("### File: README.md", model.prompts[0])
            self.assertIn("A useful project.", model.prompts[0])
            self.assertIn("UNTRUSTED DATA", model.prompts[0])

    def test_read_request_routes_to_inspection_not_shell_output(self):
        with workspace_temp() as temp:
            root = Path(temp)
            (root / "notes.txt").write_text("Important design notes.", encoding="utf-8")
            planner, _ = make_planner(
                root,
                {"action": "respond", "message": "The notes describe the design."},
            )

            action = planner.plan("Read notes.txt")

            self.assertEqual(action.action, "respond")
            self.assertNotEqual(action.command, "read notes.txt")

    def test_project_structure_request_inspects_directory(self):
        with workspace_temp() as temp:
            root = Path(temp)
            (root / "README.md").write_text("# Demo", encoding="utf-8")
            (root / "main.py").write_text("print('hello')", encoding="utf-8")
            planner, model = make_planner(
                root,
                {"action": "respond", "message": "The project has a README and one entry point."},
            )

            action = planner.plan("Explain the structure of this project")

            self.assertEqual(action.action, "respond")
            self.assertIn("### Directory structure: .", model.prompts[0])
            self.assertIn("### File: main.py", model.prompts[0])

    def test_review_can_return_a_code_write_for_user_approval(self):
        with workspace_temp() as temp:
            root = Path(temp)
            (root / "app.py").write_text("print('old')\n", encoding="utf-8")
            planner, _ = make_planner(
                root,
                {
                    "action": "code_write",
                    "message": "I fixed the issue.",
                    "files": [{"path": "app.py", "content": "print('fixed')\n"}],
                },
            )

            action = planner.plan("Review app.py and fix the bug")

            self.assertEqual(action.action, "code_write")
            self.assertEqual(action.files, [FileWrite(path="app.py", content="print('fixed')\n")])
            self.assertEqual((root / "app.py").read_text(encoding="utf-8"), "print('old')\n")

    def test_hinglish_write_request_routes_to_the_named_file(self):
        with workspace_temp() as temp:
            root = Path(temp)
            (root / "app.py").write_text("", encoding="utf-8")
            planner, _ = make_planner(
                root,
                {
                    "action": "code_write",
                    "message": "I prepared app.py.",
                    "files": [{"path": "app.py", "content": "print('saved')\n"}],
                },
            )

            action = planner.plan("app.py ke andar code likho jo saved print kare")

            self.assertEqual(action.action, "code_write")
            self.assertEqual(action.files[0].path, "app.py")

    def test_chat_code_response_is_retried_as_an_actual_file_write(self):
        with workspace_temp() as temp:
            root = Path(temp)
            (root / "app.py").write_text("", encoding="utf-8")
            planner, _ = make_planner(root, {"action": "respond", "message": "unused"})
            model = SequencedGemini(
                {"action": "respond", "message": "```python\nprint('only chat')\n```"},
                {
                    "action": "code_write",
                    "message": "I prepared the actual file change.",
                    "files": [{"path": "app.py", "content": "print('saved')\n"}],
                },
            )
            planner._gemini = model

            action = planner.plan("app.py ke andar code likho")

            self.assertEqual(action.action, "code_write")
            self.assertEqual(action.files[0].content, "print('saved')\n")
            self.assertEqual(len(model.prompts), 2)
            self.assertIn("Do not put source code in a chat response", model.prompts[1])

    def test_write_to_an_unrequested_model_path_is_blocked(self):
        with workspace_temp() as temp:
            root = Path(temp)
            (root / "app.py").write_text("", encoding="utf-8")
            planner, _ = make_planner(
                root,
                {
                    "action": "code_write",
                    "message": "Writing elsewhere.",
                    "files": [{"path": "other.py", "content": "print('wrong')\n"}],
                },
            )

            action = planner.plan("app.py ke andar code likho")

            self.assertEqual(action.action, "respond")
            self.assertIn("file you did not request", action.message)
            self.assertEqual((root / "app.py").read_text(encoding="utf-8"), "")
            self.assertFalse((root / "other.py").exists())

    def test_sensitive_files_are_not_sent_to_the_model(self):
        with workspace_temp() as temp:
            root = Path(temp)
            (root / ".env").write_text("API_KEY=secret", encoding="utf-8")
            (root / ".env.example").write_text("API_KEY=example", encoding="utf-8")

            result = inspect_workspace_paths(
                ["."],
                workspace_root=root,
                current_dir=root,
                allow_outside_workspace=False,
            )

            self.assertNotIn("API_KEY=secret", result.content)
            self.assertIn("API_KEY=example", result.content)

    def test_outside_workspace_path_is_blocked(self):
        with workspace_temp() as temp, workspace_temp() as outside:
            root = Path(temp)
            secret = Path(outside) / "private.txt"
            secret.write_text("private", encoding="utf-8")

            result = inspect_workspace_paths(
                [str(secret)],
                workspace_root=root,
                current_dir=root,
                allow_outside_workspace=False,
            )

            self.assertFalse(result.files)
            self.assertIn("Blocked path outside", result.content)
            self.assertNotIn("private\n", result.content)


class FileWriteReviewTests(unittest.TestCase):
    def test_double_escaped_document_content_is_restored_before_preview(self):
        escaped_html = (
            r'<!DOCTYPE html>\n<html lang=\"en\">\n<body>\n'
            r'<h1 class=\"title\">Hello</h1>\n</body>\n</html>\n'
        )

        action = AgentAction.from_payload({
            "action": "code_write",
            "files": [{"path": "index.html", "content": escaped_html}],
        })

        self.assertEqual(
            action.files[0].content,
            '<!DOCTYPE html>\n<html lang="en">\n<body>\n'
            '<h1 class="title">Hello</h1>\n</body>\n</html>\n',
        )

    def test_intentional_string_newline_escapes_are_preserved(self):
        javascript = r'const message = "first\nsecond\nthird";'

        action = AgentAction.from_payload({
            "action": "code_write",
            "files": [{"path": "app.js", "content": javascript}],
        })

        self.assertEqual(action.files[0].content, javascript)

    def test_double_escaped_python_without_double_quotes_is_restored(self):
        escaped_python = r"def greet():\n    print('hello')\n\ngreet()\n"

        action = AgentAction.from_payload({
            "action": "code_write",
            "files": [{"path": "hello.py", "content": escaped_python}],
        })

        self.assertEqual(
            action.files[0].content,
            "def greet():\n    print('hello')\n\ngreet()\n",
        )

    def test_apply_requires_a_reviewed_snapshot(self):
        with workspace_temp() as temp:
            root = Path(temp)
            target = root / "app.py"
            target.write_text("developer code\n", encoding="utf-8")

            result = apply_file_writes(
                [FileWrite(path="app.py", content="unreviewed change\n")],
                root,
                False,
                root,
            )

            self.assertFalse(result.success)
            self.assertIn("Blocked unreviewed AI change", result.output)
            self.assertEqual(target.read_text(encoding="utf-8"), "developer code\n")

    def test_preview_is_non_mutating_and_apply_creates_backup(self):
        with workspace_temp() as temp:
            root = Path(temp)
            target = root / "app.py"
            target.write_text("print('old')\n", encoding="utf-8")
            files = [FileWrite(path="app.py", content="print('new')\n")]

            preview = preview_file_writes(files, root, False, root)

            self.assertTrue(preview.success)
            self.assertIn("-print('old')", preview.output)
            self.assertIn("+print('new')", preview.output)
            self.assertEqual(target.read_text(encoding="utf-8"), "print('old')\n")

            result = apply_file_writes(
                files, root, False, root, expected_snapshots=preview.snapshots
            )

            self.assertTrue(result.success)
            self.assertEqual(target.read_text(encoding="utf-8"), "print('new')\n")
            backups = list((root / ".ai_backups").rglob("app.py"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(encoding="utf-8"), "print('old')\n")

    def test_apply_blocks_when_file_changed_after_preview(self):
        with workspace_temp() as temp:
            root = Path(temp)
            target = root / "app.py"
            target.write_text("reviewed\n", encoding="utf-8")
            files = [FileWrite(path="app.py", content="orbit change\n")]
            preview = preview_file_writes(files, root, False, root)

            target.write_text("developer change\n", encoding="utf-8")
            result = apply_file_writes(
                files, root, False, root, expected_snapshots=preview.snapshots
            )

            self.assertFalse(result.success)
            self.assertIn("changed after the diff was shown", result.output)
            self.assertEqual(target.read_text(encoding="utf-8"), "developer change\n")
            self.assertFalse((root / ".ai_backups").exists())

    def test_multi_file_failure_rolls_back_all_applied_targets(self):
        with workspace_temp() as temp:
            root = Path(temp)
            first = root / "first.py"
            second = root / "second.py"
            first.write_text("first old\n", encoding="utf-8")
            second.write_text("second old\n", encoding="utf-8")
            files = [
                FileWrite(path="first.py", content="first new\n"),
                FileWrite(path="second.py", content="second new\n"),
            ]
            preview = preview_file_writes(files, root, False, root)

            from ai import code_writer

            real_replace = code_writer.os.replace
            calls = 0

            def fail_second_replace(source, target):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated replace failure")
                return real_replace(source, target)

            with patch("ai.code_writer.os.replace", side_effect=fail_second_replace):
                result = apply_file_writes(
                    files, root, False, root, expected_snapshots=preview.snapshots
                )

            self.assertFalse(result.success)
            self.assertIn("rolled back", result.output)
            self.assertEqual(first.read_text(encoding="utf-8"), "first old\n")
            self.assertEqual(second.read_text(encoding="utf-8"), "second old\n")
            self.assertFalse(list(root.glob(".*.orbit-*.tmp")))
            self.assertFalse(list(root.glob(".*.rollback-*.tmp")))

    def test_incomplete_diff_is_not_approvable(self):
        with workspace_temp() as temp:
            root = Path(temp)
            target = root / "large.txt"
            target.write_text("old\n", encoding="utf-8")

            preview = preview_file_writes(
                [FileWrite(path="large.txt", content="new line\n" * 100)],
                root,
                False,
                root,
                max_chars=80,
            )

            self.assertFalse(preview.success)
            self.assertFalse(preview.snapshots)
            self.assertIn("No approval is available for a partial diff", preview.output)

    def test_inspect_action_payload_keeps_paths_and_objective(self):
        action = AgentAction.from_payload({
            "action": "inspect",
            "paths": ["README.md", "ai/llm.py"],
            "objective": "Explain the architecture",
        })

        self.assertEqual(action.action, "inspect")
        self.assertEqual(action.paths, ["README.md", "ai/llm.py"])
        self.assertEqual(action.objective, "Explain the architecture")


class PendingApprovalReplyTests(unittest.TestCase):
    def test_typed_confirmation_approves_the_reviewed_change(self):
        self.assertEqual(
            pending_action_reply("okay write this code inside index.html"),
            "approve",
        )
        self.assertEqual(pending_action_reply("haan, apply kar do"), "approve")
        self.assertEqual(pending_action_reply("Approve"), "approve")

    def test_typed_rejection_dismisses_the_reviewed_change(self):
        self.assertEqual(pending_action_reply("nahi, mat likho"), "reject")
        self.assertEqual(pending_action_reply("cancel it"), "reject")

    def test_follow_up_edit_is_not_mistaken_for_approval(self):
        self.assertIsNone(pending_action_reply("okay but change the title first"))
        self.assertIsNone(pending_action_reply("make the heading blue"))


if __name__ == "__main__":
    unittest.main()
