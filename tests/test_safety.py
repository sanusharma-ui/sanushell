import unittest

from ai.safety import SafetyPolicy


class SmartSafetyPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = SafetyPolicy()

    def assert_auto(self, command: str, expected_level: str | None = None):
        decision = self.policy.check_shell_command(command)
        self.assertFalse(decision.requires_approval, decision.reason)
        if expected_level:
            self.assertEqual(decision.level, expected_level)

    def assert_approval(self, command: str):
        decision = self.policy.check_shell_command(command)
        self.assertTrue(decision.requires_approval, command)
        self.assertIn(decision.level, {"high", "critical"})

    def test_read_only_queries_flow_without_approval(self):
        for command in (
            "files",
            "where ; files | count",
            "read README.md",
            "processes",
            "gstatus",
            "calc 5 + 7",
        ):
            with self.subTest(command=command):
                self.assert_auto(command, "low")

    def test_small_recoverable_local_actions_flow_without_approval(self):
        for command in (
            "makefolder reports",
            "makefile notes.txt",
            "duplicate notes.txt notes-copy.txt",
            "open README.md",
            "zip reports reports.zip",
            "echo hello > notes.txt",
        ):
            with self.subTest(command=command):
                self.assert_auto(command, "medium")

    def test_destructive_action_anywhere_in_chain_requires_approval(self):
        for command in (
            "delete confirm notes.txt",
            "where ; delete confirm notes.txt",
            "files | count ; kill 1234",
            "echo safe && gpush",
        ):
            with self.subTest(command=command):
                self.assert_approval(command)

    def test_external_and_bulk_changes_require_approval(self):
        for command in (
            "download https://example.com/file.zip",
            "extract archive.zip output",
            "rename old.txt new.txt",
            "gpull",
            "gcommit -m test",
        ):
            with self.subTest(command=command):
                self.assert_approval(command)

    def test_safe_native_inspection_does_not_require_approval(self):
        for command in (
            "run python --version",
            "run node -v",
            "run git status --short",
            "run git branch --show-current",
            "run pip list",
            "run whoami",
            "run ipconfig /all",
        ):
            with self.subTest(command=command):
                self.assert_auto(command, "low")

    def test_native_execution_or_mutation_requires_approval(self):
        for command in (
            "run python script.py",
            "run python",
            "run cmd /c dir",
            "run git branch new-feature",
            "run git remote add origin https://example.com/repo.git",
            "run npm audit --fix",
            "run ipconfig /release",
            "gbranch -D main",
            "open installer.exe",
            "start deploy.cmd",
        ):
            with self.subTest(command=command):
                self.assert_approval(command)

    def test_action_types_use_intent_aware_defaults(self):
        self.assertFalse(self.policy.check_action("respond").requires_approval)
        self.assertFalse(self.policy.check_action("screenshot").requires_approval)
        self.assertTrue(self.policy.check_action("code_write").requires_approval)

    def test_invalid_command_does_not_prompt_for_an_action_that_will_not_run(self):
        self.assert_auto('read "unterminated', "low")


if __name__ == "__main__":
    unittest.main()
