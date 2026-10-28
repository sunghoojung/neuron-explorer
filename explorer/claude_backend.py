"""Local Claude Code CLI transport for the explorer chat.

Each turn runs one `claude -p` process under the user's existing Claude Code login. This module never reads or
writes Claude credentials, settings, or session files; the CLI manages its own session store. The explorer MCP
server (agent_mcp.py) is the only tool source: built-in tools are disabled and only the explorer's MCP tools are
allowed.

Continuity: the chat's Claude session id is kept in the model-scoped local history file and the next turn resumes it
(`--resume`). When there is no usable session (new chat, edit/reroll rewind, a legacy history written by the Codex
transport, or a session the CLI no longer has), the first turn of a fresh session replays the local history inside
its prompt, so history is never lost. The history file format is shared with the Codex transport.
"""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import sys
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

try:  # imported as explorer.claude_backend (tests) or as a top-level module (server.py)
    from .chatgpt_backend import (
        _NAME_RE,
        _NAMESPACE_RE,
        ChatHistoryError,
        LocalChatHistory,
        _event_error_text,
        _format_tool_output,
        _normalize_attachments,
        _safe_chat_id,
        _safe_text,
        _stop_process_group,
        build_prompt,
    )
except ImportError:
    from chatgpt_backend import (
        _NAME_RE,
        _NAMESPACE_RE,
        ChatHistoryError,
        LocalChatHistory,
        _event_error_text,
        _format_tool_output,
        _normalize_attachments,
        _safe_chat_id,
        _safe_text,
        _stop_process_group,
        build_prompt,
    )

__all__ = ["ChatHistoryError", "ClaudeBackend", "ClaudeChatHistory", "ClaudeEventTranslator", "build_claude_argv"]

MCP_SERVER_NAME = "explorer"
# Tools the chat must not have even though the server exposes them (the person names neurons themselves).
DISABLED_MCP_TOOLS = ("set_name",)
_SESSION_MISSING = "no conversation found"


def build_claude_argv(
    *,
    claude_bin: str = "claude",
    model: str | None = None,
    system_prompt: str = "",
    session_id: str,
    resume: bool,
    mcp_command: str,
    mcp_args: Sequence[str],
    mcp_env: Mapping[str, str],
    mcp_server_name: str = MCP_SERVER_NAME,
) -> list[str]:
    """Build a `claude -p` argv. User and chat text never enter argv; it goes through stdin as stream-json."""
    if not claude_bin:
        raise ValueError("claude executable is required")
    if not mcp_command:
        raise ValueError("MCP server command is required")
    if not _NAME_RE.fullmatch(mcp_server_name):
        raise ValueError("MCP server name must contain only letters, numbers, underscores, or hyphens")
    if model and model.startswith("-"):
        raise ValueError("model name cannot start with a dash")
    uuid.UUID(session_id)  # raises ValueError when malformed

    mcp_config = {
        "mcpServers": {
            mcp_server_name: {
                "command": str(mcp_command),
                "args": [str(arg) for arg in mcp_args],
                "env": {str(key): str(value) for key, value in sorted(mcp_env.items())},
            }
        }
    }
    prefix = f"mcp__{mcp_server_name}"
    argv = [
        str(claude_bin),
        "-p",
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--mcp-config", json.dumps(mcp_config, ensure_ascii=False),
        "--strict-mcp-config",
        # No built-in tools at all (no Bash, Edit, Write, WebFetch); only this MCP server's tools are allowed.
        "--tools", "",
        "--allowedTools", prefix,
        "--permission-mode", "dontAsk",
        # Project settings only: the agent cwd has none, so user hooks, CLAUDE.md-style settings and plugins stay out.
        "--setting-sources", "project",
        "--disable-slash-commands",
        # Render the prompt fresh each turn so edits to the prompt files apply to resumed chats.
        "--system-prompt-snapshot", "off",
    ]
    if DISABLED_MCP_TOOLS:
        argv.extend(["--disallowedTools", *(f"{prefix}__{tool}" for tool in DISABLED_MCP_TOOLS)])
    if system_prompt.strip():
        argv.extend(["--system-prompt", system_prompt])
    if model:
        argv.extend(["--model", model])
    argv.extend(["--resume", session_id] if resume else ["--session-id", session_id])
    return argv


class ClaudeChatHistory(LocalChatHistory):
    """The shared model-scoped history file, plus the chat's Claude session id."""

    def session_id(self, chat_id: str) -> str | None:
        value = self.load(chat_id).get("claude_session_id")
        try:
            return str(uuid.UUID(str(value))) if value else None
        except ValueError:
            return None

    def set_session(self, chat_id: str, session_id: str | None) -> None:
        data = self.load(chat_id)
        if session_id:
            data["claude_session_id"] = session_id
        else:
            data.pop("claude_session_id", None)
        self._save(chat_id, data)

    def append_turn(self, chat_id: str, *, session_id: str | None = None, **kwargs: Any) -> None:
        super().append_turn(chat_id, **kwargs)
        if session_id and self.session_id(chat_id) != session_id:
            self.set_session(chat_id, session_id)

    def rewind_chat(self, chat_id: str, message: str) -> dict[str, bool]:
        result = super().rewind_chat(chat_id, message)
        if result.get("cut"):
            # The Claude session still holds the removed turns and its files are not ours to edit: start a new one
            # that is seeded from the remaining local history.
            self.set_session(chat_id, None)
        return result


class ClaudeEventTranslator:
    """Translate `claude -p --output-format stream-json` events to the explorer's NDJSON event schema."""

    def __init__(self, *, max_tool_output: int = 1500):
        self.max_tool_output = max_tool_output
        self.session_id: str | None = None
        self.failed = False
        self.done = False
        self.session_missing = False
        self._streamed_messages: set[str] = set()
        self._emitted_tools: set[str] = set()
        self._has_text = False

    @staticmethod
    def _tool_name(name: str) -> str:
        return name.removeprefix(f"mcp__{MCP_SERVER_NAME}__")

    def _text(self, text: str) -> list[dict[str, Any]]:
        if not text:
            return []
        self._has_text = True
        return [{"t": "text", "text": text}]

    def feed(self, event: Mapping[str, Any]) -> list[dict[str, Any]]:
        kind = event.get("type")
        if isinstance(event.get("session_id"), str) and not self.session_id:
            self.session_id = event["session_id"]
        if event.get("parent_tool_use_id"):
            return []  # sub-agent chatter is not part of this chat

        if kind == "stream_event":
            inner = event.get("event")
            if not isinstance(inner, Mapping):
                return []
            if inner.get("type") == "content_block_delta":
                delta = inner.get("delta")
                if isinstance(delta, Mapping) and delta.get("type") == "text_delta":
                    message_id = event.get("api_message_id")
                    if message_id:
                        self._streamed_messages.add(str(message_id))
                    return self._text(str(delta.get("text", "")))
            return []

        if kind == "assistant":
            message = event.get("message")
            if not isinstance(message, Mapping):
                return []
            out: list[dict[str, Any]] = []
            streamed = str(message.get("id", "")) in self._streamed_messages
            for block in message.get("content") or []:
                if not isinstance(block, Mapping):
                    continue
                if block.get("type") == "text" and not streamed:
                    # Without partial messages the whole block arrives at once.
                    out.extend(self._text(("\n" if self._has_text else "") + str(block.get("text", ""))))
                elif block.get("type") == "tool_use":
                    tool_id = str(block.get("id") or "")
                    if not tool_id:
                        self.failed = True
                        out.append({"t": "error", "error": "Claude CLI returned a tool call without an ID."})
                    elif tool_id not in self._emitted_tools:
                        self._emitted_tools.add(tool_id)
                        out.append({
                            "t": "tool",
                            "id": tool_id,
                            "name": self._tool_name(str(block.get("name", "tool"))),
                            "input": block.get("input") or {},
                        })
            return out

        if kind == "user":
            message = event.get("message")
            content = message.get("content") if isinstance(message, Mapping) else None
            out = []
            for block in content if isinstance(content, list) else []:
                if isinstance(block, Mapping) and block.get("type") == "tool_result":
                    out.append({
                        "t": "tool_result",
                        "id": block.get("tool_use_id"),
                        "error": bool(block.get("is_error")),
                        "output": _format_tool_output(block.get("content"), self.max_tool_output),
                    })
            return out

        if kind == "result":
            self.done = True
            text = event.get("result")
            if event.get("is_error") or str(event.get("subtype", "")).startswith("error"):
                self.failed = True
                errors = event.get("errors")
                detail = text or (errors[0] if isinstance(errors, list) and errors else None) or event.get("subtype")
                message_text = _event_error_text(detail or "Claude turn failed")
                if _SESSION_MISSING in message_text.lower():
                    self.session_missing = True
                return [{"t": "error", "error": message_text}, {"t": "done"}]
            out = []
            if not self._has_text and isinstance(text, str) and text.strip():
                out.extend(self._text(text))
            out.append({"t": "done", "cost": event.get("total_cost_usd")})
            return out
        return []


def _image_block(path: str) -> dict[str, Any]:
    media_type = mimetypes.guess_type(path)[0] or "image/png"
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


class ClaudeBackend:
    """Run `claude -p` turns and keep model-scoped local history alongside the CLI's own session."""

    def __init__(
        self,
        cache_root: str | os.PathLike[str],
        model_namespace: str,
        *,
        claude_bin: str = "claude",
        model: str | None = None,
        cwd: str | os.PathLike[str] | None = None,
        mcp_command: str = sys.executable,
        mcp_args: Sequence[str] = (),
        mcp_env: Mapping[str, str] | None = None,
        upload_root: str | os.PathLike[str] | None = None,
        process_factory: Callable[..., Any] | None = None,
    ):
        if not _NAMESPACE_RE.fullmatch(model_namespace) or model_namespace in {".", ".."}:
            raise ValueError("model namespace contains unsupported characters")
        self.model_namespace = model_namespace
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.history = ClaudeChatHistory(self.cache_root / model_namespace / "agent" / "chats", model_namespace)
        self.claude_bin = claude_bin
        self.model = model
        self.cwd = Path(cwd).expanduser().resolve() if cwd else self.cache_root / model_namespace / "agent"
        self.upload_root = (
            Path(upload_root).expanduser().resolve()
            if upload_root
            else self.cache_root / model_namespace / "agent" / "uploads"
        )
        self.mcp_command = str(mcp_command)
        self.mcp_args = tuple(str(arg) for arg in mcp_args)
        self.mcp_env = {str(key): str(value) for key, value in (mcp_env or {}).items()}
        self.process_factory = process_factory or asyncio.create_subprocess_exec
        self._active_chats: set[str] = set()
        self._chat_state_lock = threading.Lock()

    # ---- chat reservations (same contract as the Codex backend) ----
    def reserve_chat(self, chat_id: str) -> bool:
        chat_id = _safe_chat_id(chat_id)
        with self._chat_state_lock:
            if chat_id in self._active_chats:
                return False
            self._active_chats.add(chat_id)
            return True

    def is_chat_active(self, chat_id: str) -> bool:
        try:
            chat_id = _safe_chat_id(chat_id)
        except ChatHistoryError:
            return False
        with self._chat_state_lock:
            return chat_id in self._active_chats

    def _release_chat(self, chat_id: str) -> None:
        with self._chat_state_lock:
            self._active_chats.discard(chat_id)

    def release_chat(self, chat_id: str) -> None:
        try:
            chat_id = _safe_chat_id(chat_id)
        except ChatHistoryError:
            return
        self._release_chat(chat_id)

    def rewind_chat(self, chat_id: str, message: str) -> dict[str, Any]:
        try:
            chat_id = _safe_chat_id(chat_id)
        except ChatHistoryError:
            return {"ok": True, "cut": False}
        with self._chat_state_lock:
            if chat_id in self._active_chats:
                return {"ok": False, "cut": False, "error": "A turn is still running for this chat."}
            return self.history.rewind_chat(chat_id, message)

    def _validated_image_paths(self, paths: Sequence[str | os.PathLike[str]]) -> list[str]:
        allowed_suffixes = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
        validated = []
        root = self.upload_root.resolve()
        for path_value in paths:
            try:
                path = Path(path_value).expanduser().resolve(strict=True)
            except (OSError, RuntimeError):
                raise ValueError("An image attachment is missing or inaccessible.") from None
            if not path.is_relative_to(root):
                raise ValueError("Image attachments must come from this model's local upload cache.")
            if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
                raise ValueError("Only uploaded PNG, JPEG, GIF, or WebP image files are supported.")
            validated.append(str(path))
        return validated

    def _argv(self, chat_id: str, system_prompt: str, session_id: str, resume: bool) -> list[str]:
        mcp_env = dict(self.mcp_env)
        mcp_env["EXPLORER_CHAT"] = chat_id
        return build_claude_argv(
            claude_bin=self.claude_bin,
            model=self.model,
            system_prompt=system_prompt,
            session_id=session_id,
            resume=resume,
            mcp_command=self.mcp_command,
            mcp_args=self.mcp_args,
            mcp_env=mcp_env,
        )

    @staticmethod
    def _content(
        *,
        history: Mapping[str, Any],
        resume: bool,
        message: str,
        context: str,
        attachments: Sequence[str],
        image_paths: Sequence[str],
        history_images: Sequence[str],
    ) -> list[dict[str, Any]]:
        """The stream-json user message content: text (with replayed history for a fresh session), then attachments."""
        if resume:
            head = f"Explorer context: {context.strip()}\n\n{message}" if context.strip() else str(message)
        else:
            head = build_prompt(
                system_prompt="",
                history=history.get("turns", []),
                message=message,
                context=context,
                image_paths=image_paths,
            )
        blocks: list[dict[str, Any]] = [{"type": "text", "text": head}]
        blocks.extend({"type": "text", "text": text} for text in attachments)
        blocks.extend(_image_block(path) for path in [*history_images, *image_paths])
        return blocks

    async def stream_turn(
        self,
        chat_id: str | None,
        message: str,
        *,
        system_prompt: str = "",
        context: str = "",
        attachments: Sequence[Any] = (),
        image_paths: Sequence[str | os.PathLike[str]] = (),
        on_process: Callable[[Any], None] | None = None,
        ui_queue: asyncio.Queue | None = None,
        reserved: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield explorer stream events for one user turn (chat, text, tool, tool_result, ui, error, done)."""
        chat_id = _safe_chat_id(chat_id or uuid.uuid4().hex)
        yield {"t": "chat", "chat": chat_id}
        owns_reservation = self.is_chat_active(chat_id) if reserved else self.reserve_chat(chat_id)
        if not owns_reservation:
            yield {"t": "error", "error": "A turn is already active for this conversation."}
            yield {"t": "done"}
            return

        proc = None
        stdout_task = stderr_task = ui_task = stdin_task = None
        assistant_parts: list[str] = []
        recorded_events: list[dict[str, Any]] = []
        cancelled = False
        history_committed = False
        session_started = False
        session_id: str | None = None
        stored_session: str | None = None
        normalized_attachments: list[str] = []
        current_images: list[str] = []
        translator = ClaudeEventTranslator()

        try:
            normalized_attachments = _normalize_attachments(attachments)
            history = self.history.load(chat_id)
            stored_session = self.history.session_id(chat_id)
            current_images = self._validated_image_paths(image_paths)
            for attempt in range(2):
                resume = stored_session is not None and attempt == 0
                session_id = stored_session if resume else str(uuid.uuid4())
                history_images: list[str] = []
                if not resume:
                    seen = set(current_images)
                    for turn in history.get("turns", []):
                        for path in self._validated_image_paths(turn.get("image_paths", [])):
                            if path not in seen:
                                seen.add(path)
                                history_images.append(path)
                content = self._content(
                    history=history,
                    resume=resume,
                    message=message,
                    context=context,
                    attachments=normalized_attachments,
                    image_paths=current_images,
                    history_images=history_images,
                )
                payload = json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n"
                argv = self._argv(chat_id, system_prompt, session_id, resume)
                self.cwd.mkdir(parents=True, exist_ok=True)
                translator = ClaudeEventTranslator()
                assistant_parts.clear()
                recorded_events.clear()
                proc = await self.process_factory(
                    *argv,
                    cwd=str(self.cwd),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    limit=1 << 26,
                    start_new_session=True,
                )
                if on_process:
                    on_process(proc)
                if proc.stdin is None or proc.stdout is None:
                    raise RuntimeError("Claude CLI did not expose stdin and stdout pipes")
                stdout_task = asyncio.create_task(proc.stdout.readline())
                stderr_task = asyncio.create_task(proc.stderr.read()) if proc.stderr is not None else None
                if ui_queue is not None and ui_task is None:
                    ui_task = asyncio.create_task(ui_queue.get())

                async def send_prompt(proc=proc, payload=payload):
                    try:
                        proc.stdin.write(payload.encode("utf-8"))
                        await proc.stdin.drain()
                        proc.stdin.close()
                        try:
                            await proc.stdin.wait_closed()
                        except (AttributeError, BrokenPipeError, ConnectionResetError):
                            pass
                    except (BrokenPipeError, ConnectionResetError):
                        pass

                stdin_task = asyncio.create_task(send_prompt())
                malformed_output = False
                held: list[dict[str, Any]] = []  # error and done events, released after the exit status is known

                while stdout_task is not None:
                    wait_tasks = {stdout_task}
                    if ui_task is not None:
                        wait_tasks.add(ui_task)
                    ready, _pending = await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)

                    if ui_task is not None and ui_task in ready:
                        queued = ui_task.result()
                        ui_task = asyncio.create_task(ui_queue.get()) if ui_queue is not None else None
                        ui_event = self._ui_event(queued)
                        if ui_event is not None:
                            yield ui_event

                    if stdout_task in ready:
                        line = stdout_task.result()
                        if not line:
                            stdout_task = None
                            break
                        stdout_task = asyncio.create_task(proc.stdout.readline())
                        try:
                            cli_event = json.loads(line)
                        except (ValueError, UnicodeDecodeError):
                            continue  # the CLI may print non-JSON notices; the exit status decides success
                        if not isinstance(cli_event, Mapping):
                            if not malformed_output:
                                malformed_output = True
                                translator.failed = True
                                held.append({"t": "error", "error": "Claude CLI returned an invalid stream event."})
                            continue
                        if cli_event.get("type") == "system" and cli_event.get("subtype") == "init":
                            session_started = True
                        for event in translator.feed(cli_event):
                            if event.get("t") == "text":
                                assistant_parts.append(event.get("text", ""))
                            elif event.get("t") in {"tool", "tool_result"}:
                                recorded_events.append(event)
                            elif event.get("t") in {"error", "done"}:
                                held.append(event)
                                continue
                            yield event

                await stdin_task
                if ui_queue is not None:
                    while not ui_queue.empty():
                        ui_event = self._ui_event(ui_queue.get_nowait())
                        if ui_event is not None:
                            yield ui_event

                return_code = await proc.wait()
                stderr_data = await stderr_task if stderr_task is not None else b""
                if resume and translator.failed and translator.session_missing and not assistant_parts:
                    # The CLI no longer has the session: start a fresh one seeded from local history.
                    stored_session = None
                    self.history.set_session(chat_id, None)
                    continue
                break

            done_event = next((e for e in held if e.get("t") == "done"), None)
            errors = [e for e in held if e.get("t") == "error"]
            if return_code != 0 and not translator.failed:
                errors.append({"t": "error", "error": _exit_error(return_code, stderr_data)})
                translator.failed = True
            if return_code == 0 and not translator.done and not translator.failed:
                errors.append({"t": "error", "error": "Claude CLI exited without completing the turn."})
                translator.failed = True
            has_tool_content = any(
                e.get("t") == "tool_result" and not e.get("error") and str(e.get("output", "")).strip()
                for e in recorded_events
            )
            if (
                return_code == 0
                and translator.done
                and not translator.failed
                and not "".join(assistant_parts).strip()
                and not has_tool_content
            ):
                errors.append({"t": "error", "error": "Claude CLI completed without any assistant or tool content."})
                translator.failed = True
            for error in errors:
                yield error
            if translator.failed:
                # The CLI session may hold a half-finished turn the local history does not: reseed next time.
                if session_started and stored_session is not None:
                    try:
                        self.history.set_session(chat_id, None)
                    except (OSError, ChatHistoryError):
                        pass
            else:
                self.history.append_turn(
                    chat_id,
                    session_id=session_id,
                    message=message,
                    context=context,
                    attachments=normalized_attachments,
                    image_paths=current_images,
                    assistant="".join(assistant_parts),
                    events=recorded_events,
                )
                history_committed = True
            yield done_event or {"t": "done"}
        except asyncio.CancelledError:
            cancelled = True
            raise
        except (OSError, RuntimeError, ValueError, TypeError, AttributeError) as exc:
            translator.failed = True
            yield {"t": "error", "error": _safe_text(f"Claude chat failed: {exc}")}
            yield {"t": "done"}
        finally:
            for task in (stdin_task, stdout_task, ui_task):
                if task is not None and not task.done():
                    task.cancel()
            if proc is not None and proc.returncode is None:
                await _stop_process_group(proc)
            if on_process:
                on_process(None)
            if stderr_task is not None and not stderr_task.done():
                stderr_task.cancel()
            if cancelled and proc is not None and not history_committed:
                try:
                    self.history.append_turn(
                        chat_id,
                        # Keep the session only when the CLI actually created it; otherwise reseed from history.
                        session_id=session_id if session_started else None,
                        message=message,
                        context=context,
                        attachments=normalized_attachments,
                        image_paths=current_images,
                        assistant="".join(assistant_parts),
                        events=recorded_events,
                        interrupted=True,
                    )
                    if not session_started:
                        self.history.set_session(chat_id, None)
                except (OSError, ChatHistoryError):
                    pass
            if owns_reservation:
                self._release_chat(chat_id)

    @staticmethod
    def _ui_event(queued: Any) -> dict[str, Any] | None:
        if isinstance(queued, tuple) and len(queued) == 2:
            kind, value = queued
            if kind == "ui" and isinstance(value, Mapping):
                return {"t": "ui", **value}
            return None
        if isinstance(queued, Mapping) and queued.get("t") == "ui":
            return dict(queued)
        return None


def _exit_error(return_code: int, stderr_data: bytes | str) -> str:
    stderr = stderr_data.decode("utf-8", errors="replace") if isinstance(stderr_data, bytes) else str(stderr_data)
    lowered = stderr.lower()
    if "not logged in" in lowered or "/login" in lowered or "authentication" in lowered:
        return "Claude CLI is not signed in. Run `claude` once, sign in with /login, and retry."
    if "mcp" in lowered and ("failed" in lowered or "error" in lowered):
        return "Claude CLI could not start the explorer MCP server. Check that the explorer server is running."
    detail = _safe_text(stderr.strip()[-300:])
    return f"Claude CLI exited with status {return_code}." + (f" {detail}" if detail else "")
