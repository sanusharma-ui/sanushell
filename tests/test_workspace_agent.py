import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ai.actions import AgentAction, FileWrite
from ai.code_writer import apply_file_writes, preview_file_writes
from ai.llm import AgentPlanner
from ai.workspace_reader import inspect_workspace_paths
from core.shell import Shell


class CapturingGemini:
    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    def generate_content(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(text=json.dumps(self.payload))


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

            result = apply_file_writes(files, root, False, root)

            self.assertTrue(result.success)
            self.assertEqual(target.read_text(encoding="utf-8"), "print('new')\n")
            backups = list((root / ".ai_backups").rglob("app.py"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(encoding="utf-8"), "print('old')\n")

    def test_inspect_action_payload_keeps_paths_and_objective(self):
        action = AgentAction.from_payload({
            "action": "inspect",
            "paths": ["README.md", "ai/llm.py"],
            "objective": "Explain the architecture",
        })

        self.assertEqual(action.action, "inspect")
        self.assertEqual(action.paths, ["README.md", "ai/llm.py"])
        self.assertEqual(action.objective, "Explain the architecture")


if __name__ == "__main__":
    unittest.main()
