"""Mocked tests for the Codex CLI transport and local explorer chat history."""

import asyncio
import json
import sys

import pytest

from explorer.chatgpt_backend import (
    ChatGPTBackend,
    ChatHistoryError,
    CodexEventTranslator,
    LocalChatHistory,
    _normalize_attachments,
    build_codex_argv,
    build_prompt,
)

MODEL_NAMESPACE = "meta-llama-llama-3.2-3b-instruct-1642e64c6774"


def test_build_codex_argv_is_safe_and_configures_only_local_mcp(tmp_path):
    image = tmp_path / "a photo.png"
    argv = build_codex_argv(
        codex_bin="/opt/codex",
        model="gpt-test",
        mcp_command=sys.executable,
        mcp_args=["/tmp/explorer agent_mcp.py"],
        mcp_env={"EXPLORER_CHAT": "chat_123", "EXPLORER_URL": "http://127.0.0.1:8791"},
        image_paths=[image],
    )

    assert argv[:2] == ["/opt/codex", "exec"]
    assert "--json" in argv
    assert "--ignore-user-config" in argv
    assert "--ephemeral" in argv
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert argv[argv.index("--model") + 1] == "gpt-test"
    assert argv[-1] == "-"
    assert "--image" in argv
    assert argv[argv.index("--image") + 1] == str(image.resolve())
    assert all("user message" not in arg for arg in argv)
    assert any('mcp_servers.explorer.default_tools_approval_mode="approve"' == value for value in argv)
    assert any('mcp_servers.explorer.env.EXPLORER_CHAT="chat_123"' == value for value in argv)
    assert not any("CODEX_HOME" in value or "token=" in value.lower() for value in argv)


def test_build_codex_argv_rejects_unsafe_mcp_configuration():
    with pytest.raises(ValueError, match="MCP server name"):
        build_codex_argv(
            mcp_command="python",
            mcp_args=[],
            mcp_env={},
            mcp_server_name="../other",
        )
    with pytest.raises(ValueError, match="environment key"):
        build_codex_argv(
            mcp_command="python",
            mcp_args=[],
            mcp_env={"BAD.KEY": "value"},
        )


def test_local_history_is_model_scoped_atomic_and_rewindable(tmp_path):
    history = LocalChatHistory(tmp_path / "cache" / MODEL_NAMESPACE / "agent" / "chats", MODEL_NAMESPACE)
    history.append_turn("chat-123", message="first", assistant="answer one")
    history.append_turn(
        "chat-123",
        message="second",
        assistant="answer two",
        events=[{"t": "tool_result", "output": "result"}],
    )

    data = history.load("chat-123")
    assert len(data["turns"]) == 2
    assert history.history_dir.parts[-3:] == (MODEL_NAMESPACE, "agent", "chats")
    assert history.rewind_chat("chat-123", "second") == {"ok": True, "cut": True}
    assert [turn["message"] for turn in history.load("chat-123")["turns"]] == ["first"]
    assert history.rewind_chat("chat-123", "missing") == {"ok": True, "cut": False}
    with pytest.raises(ChatHistoryError, match="chat id"):
        history.load("../escape")


def test_prompt_carries_system_history_context_and_attachments():
    prompt = build_prompt(
        system_prompt="Use the explorer tools.",
        history=[{
            "message": "What is this neuron?",
            "assistant": "It activates on names.",
            "events": [
                {"t": "tool", "name": "mcp__explorer__neuron", "input": {"layer": 2, "n": 4}},
                {"t": "tool_result", "output": "cached neuron details"},
            ],
        }],
        message="Compare it with the neighbor.",
        context="L2 N4 is open.",
        attachments=["<file name=notes.txt>local note</file>"],
        image_paths=["/tmp/chart.png"],
    )

    assert "EXPLORER APPLICATION INSTRUCTIONS" in prompt
    assert "PRIOR CONVERSATION" in prompt
    assert "Tool result: cached neuron details" in prompt
    assert "Explorer context: L2 N4 is open." in prompt
    assert "Compare it with the neighbor." in prompt
    assert "local note" in prompt
    assert "Attached image count: 1" in prompt


def test_attachment_normalizer_rejects_unsupported_binary_blocks():
    assert _normalize_attachments(["plain text", {"type": "text", "text": "file text"}]) == [
        "plain text",
        "file text",
    ]
    with pytest.raises(ValueError, match="image paths"):
        _normalize_attachments([{"type": "image", "source": {"type": "base64", "data": "abc"}}])
    with pytest.raises(ValueError, match="Document attachments"):
        _normalize_attachments([{"type": "document", "source": {"type": "base64", "data": "abc"}}])


def test_codex_event_translation_handles_text_tools_results_and_errors():
    translator = CodexEventTranslator()
    started = translator.feed({
        "type": "item.started",
        "item": {
            "id": "item_1",
            "type": "mcp_tool_call",
            "server": "explorer",
            "tool": "overview",
            "arguments": {},
            "status": "in_progress",
        },
    })
    assert started == [{"t": "tool", "id": "item_1", "name": "mcp__explorer__overview", "input": {}}]

    completed = translator.feed({
        "type": "item.completed",
        "item": {
            "id": "item_1",
            "type": "mcp_tool_call",
            "server": "explorer",
            "tool": "overview",
            "arguments": {},
            "result": {"content": [{"type": "text", "text": "overview data"}], "structured_content": None},
            "error": None,
            "status": "completed",
        },
    })
    assert completed == [{"t": "tool_result", "id": "item_1", "error": False, "output": "overview data"}]

    assert translator.feed({
        "type": "item.updated",
        "item": {"id": "message_1", "type": "agent_message", "text": "Hel"},
    }) == [{"t": "text", "text": "Hel"}]
    assert translator.feed({
        "type": "item.completed",
        "item": {"id": "message_1", "type": "agent_message", "text": "Hello"},
    }) == [{"t": "text", "text": "lo"}]
    assert translator.feed({"type": "turn.completed", "usage": {"total_tokens": 12}}) == [
        {"t": "done", "usage": {"total_tokens": 12}}
    ]

    failed = CodexEventTranslator().feed({
        "type": "item.completed",
        "item": {
            "id": "item_2",
            "type": "mcp_tool_call",
            "server": "explorer",
            "tool": "overview",
            "arguments": {},
            "result": {"content": [{"type": "text", "text": "denied"}], "isError": True},
            "error": None,
            "status": "completed",
        },
    })
    assert failed[0]["t"] == "tool"
    assert failed[1] == {"t": "tool_result", "id": "item_2", "error": True, "output": "denied"}


class FakeReader:
    def __init__(self, contents):
        self.lines = list(contents) if isinstance(contents, (list, tuple)) else []
        self.contents = contents if isinstance(contents, bytes) else b"".join(self.lines)

    async def readline(self):
        await asyncio.sleep(0)
        return self.lines.pop(0) if self.lines else b""

    async def read(self):
        await asyncio.sleep(0)
        return self.contents


class FakeWriter:
    def __init__(self):
        self.data = bytearray()
        self.closed = False

    def write(self, data):
        self.data.extend(data)

    async def drain(self):
        await asyncio.sleep(0)

    def close(self):
        self.closed = True

    async def wait_closed(self):
        await asyncio.sleep(0)


class FakeProcess:
    def __init__(self, output_lines, stderr=b""):
        self.stdin = FakeWriter()
        self.stdout = FakeReader(output_lines)
        self.stderr = FakeReader(stderr)
        self.returncode = 0
        self.pid = None

    async def wait(self):
        await asyncio.sleep(0)
        return self.returncode


@pytest.mark.asyncio
async def test_stream_turn_uses_stdin_emits_events_and_persists_chat(tmp_path):
    cli_events = [
        {"type": "thread.started", "thread_id": "ephemeral"},
        {
            "type": "item.started",
            "item": {
                "id": "item_1",
                "type": "mcp_tool_call",
                "server": "explorer",
                "tool": "overview",
                "arguments": {},
                "status": "in_progress",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "item_1",
                "type": "mcp_tool_call",
                "server": "explorer",
                "tool": "overview",
                "arguments": {},
                "result": {"content": [{"type": "text", "text": "model info"}]},
                "status": "completed",
            },
        },
        {"type": "item.completed", "item": {"id": "msg_1", "type": "agent_message", "text": "Ready."}},
        {"type": "turn.completed", "usage": {"total_tokens": 20}},
    ]
    process = FakeProcess([json.dumps(event).encode() + b"\n" for event in cli_events], stderr=b"safe diagnostic")
    calls = []

    async def fake_factory(*argv, **kwargs):
        calls.append((argv, kwargs))
        return process

    backend = ChatGPTBackend(
        tmp_path / "cache",
        MODEL_NAMESPACE,
        codex_bin="/fake/codex",
        mcp_command=sys.executable,
        mcp_args=["/tmp/agent_mcp.py"],
        mcp_env={"EXPLORER_URL": "http://127.0.0.1:8791"},
        process_factory=fake_factory,
    )
    process_state = []
    events = [event async for event in backend.stream_turn(
        None,
        "Summarize this model.",
        system_prompt="Use explorer data.",
        context="L1 N2 is open.",
        on_process=process_state.append,
    )]

    chat_id = events[0]["chat"]
    assert events[0] == {"t": "chat", "chat": chat_id}
    assert [event["t"] for event in events] == ["chat", "tool", "tool_result", "text", "done"]
    assert events[2]["output"] == "model info"
    assert events[3]["text"] == "Ready."
    assert process_state[0] is process and process_state[-1] is None
    assert process.stdin.closed
    assert "Summarize this model." in process.stdin.data.decode()
    assert all("Summarize this model." not in arg for arg in calls[0][0])
    assert calls[0][0][-1] == "-"
    assert any(f'mcp_servers.explorer.env.EXPLORER_CHAT="{chat_id}"' == arg for arg in calls[0][0])
    saved = backend.history.load(chat_id)
    assert saved["turns"][0]["message"] == "Summarize this model."
    assert saved["turns"][0]["assistant"] == "Ready."
    assert saved["turns"][0]["status"] == "complete"
    assert not backend.is_chat_active(chat_id)


@pytest.mark.asyncio
async def test_stream_turn_reports_unsupported_attachment_and_does_not_spawn(tmp_path):
    spawned = False

    async def fake_factory(*_argv, **_kwargs):
        nonlocal spawned
        spawned = True
        raise AssertionError("unsupported attachment should fail before launching Codex")

    backend = ChatGPTBackend(
        tmp_path / "cache",
        MODEL_NAMESPACE,
        process_factory=fake_factory,
    )
    events = [event async for event in backend.stream_turn(
        "chat-1",
        "Analyze this file.",
        attachments=[{"type": "document", "source": {"type": "base64", "data": "abc"}}],
    )]
    assert not spawned
    assert events[0] == {"t": "chat", "chat": "chat-1"}
    assert events[1]["t"] == "error"
    assert "Document attachments are not supported" in events[1]["error"]
    assert events[2] == {"t": "done"}


@pytest.mark.asyncio
async def test_stream_turn_requires_terminal_event_and_nonempty_content(tmp_path):
    cases = {
        "missing-terminal": [{"type": "item.completed", "item": {
            "id": "msg_1", "type": "agent_message", "text": "Looks fine."}}],
        "malformed": [b"{not json}\n"],
        "empty": [],
    }
    for chat_id, raw_events in cases.items():
        lines = [
            event if isinstance(event, bytes) else json.dumps(event).encode() + b"\n"
            for event in raw_events
        ]
        if chat_id == "empty":
            lines.append(json.dumps({"type": "turn.completed"}).encode() + b"\n")
        process = FakeProcess(lines)

        async def fake_factory(*_argv, _process=process, **_kwargs):
            return _process

        backend = ChatGPTBackend(tmp_path / chat_id, MODEL_NAMESPACE, process_factory=fake_factory)
        events = [event async for event in backend.stream_turn(chat_id, "Try this.")]
        assert any(event["t"] == "error" for event in events), chat_id
        assert events[-1]["t"] == "done"
        assert backend.history.load(chat_id)["turns"] == []
        assert not backend.is_chat_active(chat_id)


@pytest.mark.asyncio
async def test_reservations_reject_duplicate_turn_and_rewind_while_active(tmp_path):
    backend = ChatGPTBackend(tmp_path, MODEL_NAMESPACE)
    assert backend.reserve_chat("chat-1")
    assert not backend.reserve_chat("chat-1")
    assert backend.rewind_chat("chat-1", "latest") == {
        "ok": False,
        "cut": False,
        "error": "A turn is still running for this chat.",
    }
    duplicate = [event async for event in backend.stream_turn("chat-1", "blocked")]
    assert [event["t"] for event in duplicate] == ["chat", "error", "done"]
    backend.release_chat("chat-1")

    process = FakeProcess([
        json.dumps({"type": "item.completed", "item": {
            "id": "msg_1", "type": "agent_message", "text": "Done."}}).encode() + b"\n",
        json.dumps({"type": "turn.completed"}).encode() + b"\n",
    ])

    async def fake_factory(*_argv, **_kwargs):
        return process

    backend.process_factory = fake_factory
    assert backend.reserve_chat("chat-1")
    events = [event async for event in backend.stream_turn("chat-1", "reserved", reserved=True)]
    assert events[-1]["t"] == "done"
    assert not backend.is_chat_active("chat-1")


class BlockingReader(FakeReader):
    async def readline(self):
        if self.lines:
            await asyncio.sleep(0)
            return self.lines.pop(0)
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_cancelled_turn_persists_partial_response_and_releases_reservation(tmp_path):
    process = FakeProcess([
        json.dumps({"type": "item.updated", "item": {
            "id": "msg_1", "type": "agent_message", "text": "Partial answer"}}).encode() + b"\n",
    ])
    process.returncode = None
    process.stdout = BlockingReader(process.stdout.lines)

    async def fake_factory(*_argv, **_kwargs):
        return process

    backend = ChatGPTBackend(tmp_path, MODEL_NAMESPACE, process_factory=fake_factory)
    text_seen = asyncio.Event()

    async def consume():
        async for event in backend.stream_turn("chat-stop", "Question"):
            if event["t"] == "text":
                text_seen.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(text_seen.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not backend.is_chat_active("chat-stop")
    turn = backend.history.load("chat-stop")["turns"][0]
    assert turn["interrupted"] is True
    assert turn["status"] == "interrupted"
    assert turn["assistant"] == "Partial answer"
    prompt = build_prompt(system_prompt="", history=[turn], message="Continue.")
    assert "Assistant (interrupted; partial response): Partial answer" in prompt


@pytest.mark.asyncio
async def test_image_paths_are_confined_to_model_upload_root(tmp_path):
    cache_root = tmp_path / "cache"
    upload_root = cache_root / MODEL_NAMESPACE / "agent" / "uploads"
    upload_root.mkdir(parents=True)
    image = upload_root / "sample.png"
    image.write_bytes(b"image")
    process = FakeProcess([
        json.dumps({"type": "item.completed", "item": {
            "id": "msg_1", "type": "agent_message", "text": "Seen."}}).encode() + b"\n",
        json.dumps({"type": "turn.completed"}).encode() + b"\n",
    ])
    calls = []

    async def fake_factory(*argv, **_kwargs):
        calls.append(argv)
        return process

    backend = ChatGPTBackend(cache_root, MODEL_NAMESPACE, process_factory=fake_factory)
    backend.history.append_turn("chat-valid", message="Prior", assistant="Seen", image_paths=[str(image)])
    events = [event async for event in backend.stream_turn("chat-valid", "Look", image_paths=[image])]
    assert events[-1]["t"] == "done"
    assert calls[0].count("--image") == 1
    assert calls[0][calls[0].index("--image") + 1] == str(image.resolve())

    outside = tmp_path / "outside.png"
    outside.write_bytes(b"image")
    spawned = False

    async def should_not_spawn(*_argv, **_kwargs):
        nonlocal spawned
        spawned = True
        raise AssertionError("outside image path must be rejected before spawning Codex")

    rejected_backend = ChatGPTBackend(cache_root, MODEL_NAMESPACE, process_factory=should_not_spawn)
    current_events = [event async for event in rejected_backend.stream_turn(
        "chat-outside-current", "Look", image_paths=[outside]
    )]
    assert not spawned
    assert "local upload cache" in current_events[1]["error"]

    rejected_backend.history.append_turn(
        "chat-outside-history", message="Prior", assistant="Seen", image_paths=[str(outside)]
    )
    history_events = [event async for event in rejected_backend.stream_turn("chat-outside-history", "Again")]
    assert not spawned
    assert "local upload cache" in history_events[1]["error"]
