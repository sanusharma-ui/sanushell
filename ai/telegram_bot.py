from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

from ai.actions import AgentAction
from ai.code_writer import FileSnapshot, apply_file_writes, preview_file_writes
from ai.config import AIConfig
from ai.llm import AgentPlanner
from ai.safety import SafetyPolicy
from ai.screenshot import capture_screenshot
from core.shell import Shell


@dataclass
class PendingAction:
    id: str
    requester_id: int
    action: AgentAction
    reason: str
    created_at: datetime
    snapshots: tuple[FileSnapshot, ...] = ()


class TelegramAIBot:
    def __init__(self, config: AIConfig):
        self.config = config
        self.shell = Shell(
            start_dir=config.workspace_root,
            workspace_root=config.workspace_root,
            allow_outside_workspace=config.allow_outside_workspace,
        )
        self.safety = SafetyPolicy()
        self.planner = AgentPlanner(
            config=config,
            command_names=self.shell.registry.all_names(),
            command_catalog=self.shell.registry.catalog_entries(),
            current_dir_provider=lambda: self.shell.ctx.cwd,
        )
        self.pending: dict[str, PendingAction] = {}
        self.shell_lock = asyncio.Lock()

    def run(self) -> None:
        problems = self.config.validate_for_telegram()
        if problems:
            raise RuntimeError("Cannot start AI bot: " + "; ".join(problems))

        from telegram.ext import Application, CommandHandler, MessageHandler, filters

        app = Application.builder().token(self.config.telegram_bot_token).build()
        app.add_handler(CommandHandler("start", self.start))
        app.add_handler(CommandHandler("help", self.start))
        app.add_handler(CommandHandler("cmd", self.cmd))
        app.add_handler(CommandHandler("screenshot", self.screenshot))
        app.add_handler(CommandHandler("approve", self.approve))
        app.add_handler(CommandHandler("deny", self.deny))
        app.add_handler(CommandHandler("pending", self.pending_list))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.message))
        app.run_polling(allowed_updates=["message"])

    def _is_allowed(self, user_id: int | None) -> bool:
        if user_id is None:
            return False
        if self.config.telegram_allow_unlisted_users:
            return True
        return user_id in self.config.telegram_allowed_user_ids

    async def _guard(self, update) -> bool:
        user_id = update.effective_user.id if update.effective_user else None
        if self._is_allowed(user_id):
            return True
        await update.message.reply_text("Unauthorized. Add your Telegram user id to TELEGRAM_ALLOWED_USER_IDS.")
        return False

    async def start(self, update, context) -> None:
        if not await self._guard(update):
            return
        await update.message.reply_text(
            "RiftShell AI bot online.\n"
            "Send natural language, /cmd <riftshell command>, /screenshot, /pending, /approve <id>, /deny <id>."
        )

    async def cmd(self, update, context) -> None:
        if not await self._guard(update):
            return
        command = " ".join(context.args).strip()
        if not command:
            await update.message.reply_text("Usage: /cmd <riftshell command>")
            return
        await self._handle_action(update, AgentAction(action="shell", command=command, message="Direct command."))

    async def screenshot(self, update, context) -> None:
        if not await self._guard(update):
            return
        await self._handle_action(update, AgentAction(action="screenshot", message="Taking screenshot."))

    async def message(self, update, context) -> None:
        if not await self._guard(update):
            return
        text = update.message.text or ""
        progress_task = asyncio.create_task(
            self._delayed_progress(update, "Thinking... I will run a command only if your request needs system action.")
        )
        try:
            action = await asyncio.to_thread(self.planner.plan, text)
        except Exception as exc:
            progress_task.cancel()
            await update.message.reply_text(f"I hit an internal planning error: {exc}")
            return
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        await self._handle_action(update, action)

    async def approve(self, update, context) -> None:
        if not await self._guard(update):
            return
        if not context.args:
            await update.message.reply_text("Usage: /approve <id>")
            return

        action_id = context.args[0].strip()
        pending = self.pending.pop(action_id, None)
        if pending is None:
            await update.message.reply_text(f"No pending action found for {action_id}.")
            return
        if pending.requester_id != update.effective_user.id:
            await update.message.reply_text("Only the requester can approve this action.")
            self.pending[action_id] = pending
            return

        await update.message.reply_text(f"Approved {action_id}. Running now...")
        await self._execute_action(update, pending.action, pending.snapshots)

    async def deny(self, update, context) -> None:
        if not await self._guard(update):
            return
        if not context.args:
            await update.message.reply_text("Usage: /deny <id>")
            return
        action_id = context.args[0].strip()
        if self.pending.pop(action_id, None):
            await update.message.reply_text(f"Denied {action_id}.")
        else:
            await update.message.reply_text(f"No pending action found for {action_id}.")

    async def pending_list(self, update, context) -> None:
        if not await self._guard(update):
            return
        self._expire_old_pending()
        if not self.pending:
            await update.message.reply_text("No pending approvals.")
            return

        lines = ["Pending approvals:"]
        for item in self.pending.values():
            lines.append(f"- {item.id}: {self._describe_action(item.action)} | {item.reason}")
        await update.message.reply_text("\n".join(lines))

    async def _handle_action(self, update, action: AgentAction) -> None:
        if action.action == "respond":
            await update.message.reply_text(action.message or "I need a clearer instruction.")
            return

        decision = self.safety.check_action(action.action, action.command)
        if decision.requires_approval:
            snapshots: tuple[FileSnapshot, ...] = ()
            preview_text = ""
            if action.action == "code_write":
                preview = preview_file_writes(
                    action.files,
                    self.config.workspace_root,
                    self.config.allow_outside_workspace,
                    self.shell.ctx.cwd,
                    max_chars=max(800, min(3_000, self.config.command_output_limit - 700)),
                )
                if not preview.success:
                    await update.message.reply_text(
                        "The proposed file change cannot be safely reviewed:\n\n"
                        + self._trim(preview.output)
                    )
                    return
                snapshots = preview.snapshots
                preview_text = f"\nDiff:\n{preview.output}\n"
            action_id = uuid.uuid4().hex[:8]
            self.pending[action_id] = PendingAction(
                id=action_id,
                requester_id=update.effective_user.id,
                action=action,
                reason=decision.reason,
                created_at=datetime.now(),
                snapshots=snapshots,
            )
            await update.message.reply_text(
                "Approval needed.\n"
                f"ID: {action_id}\n"
                f"Reason: {decision.reason}\n"
                f"Action: {self._describe_action(action)}\n"
                f"{preview_text}"
                f"Run /approve {action_id} or /deny {action_id}"
            )
            return

        await self._execute_action(update, action)

    async def _execute_action(
        self,
        update,
        action: AgentAction,
        expected_snapshots: tuple[FileSnapshot, ...] | None = None,
    ) -> None:
        if action.action == "shell":
            await self._execute_shell(update, action.command)
            return
        if action.action == "screenshot":
            await update.message.reply_text("Taking a screenshot...")
            await self._send_screenshot(update)
            return
        if action.action == "code_write":
            await update.message.reply_text(
                "Code in progress... preparing files and writing them with backups where needed."
            )
            result = await asyncio.to_thread(
                apply_file_writes,
                action.files,
                self.config.workspace_root,
                self.config.allow_outside_workspace,
                self.shell.ctx.cwd,
                expected_snapshots,
            )
            await update.message.reply_text(self._trim(result.output))
            return
        await update.message.reply_text(action.message or "Done.")

    async def _execute_shell(self, update, command: str) -> None:
        await update.message.reply_text(f"Running command: {command}")
        async with self.shell_lock:
            result = await asyncio.to_thread(self.shell.execute_line, command)

        text = result.output or "(no output)"
        prefix = "OK" if result.success else "FAILED"
        await update.message.reply_text(self._trim(f"{prefix}: {command}\n\n{text}"))

    async def _send_screenshot(self, update) -> None:
        try:
            path = await asyncio.to_thread(capture_screenshot)
            with path.open("rb") as image:
                await update.message.reply_photo(photo=image, caption="Screenshot")
        except Exception as exc:
            await update.message.reply_text(f"Screenshot failed: {exc}")

    async def _delayed_progress(self, update, message: str, delay: float = 1.2) -> None:
        await asyncio.sleep(delay)
        await update.message.reply_text(message)

    def _describe_action(self, action: AgentAction) -> str:
        if action.action == "shell":
            return f"shell: {action.command}"
        if action.action == "code_write":
            paths = ", ".join(file.path for file in action.files) or "(no files)"
            return f"code_write: {paths}"
        return action.action

    def _trim(self, text: str) -> str:
        limit = max(500, self.config.command_output_limit)
        if len(text) <= limit:
            return text
        return text[: limit - 80] + "\n\n...[trimmed; output was too long]"

    def _expire_old_pending(self) -> None:
        expires_before = datetime.now() - timedelta(minutes=self.config.approval_timeout_minutes)
        expired = [key for key, item in self.pending.items() if item.created_at < expires_before]
        for key in expired:
            self.pending.pop(key, None)
