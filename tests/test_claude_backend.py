"""Tests for the Claude CLI transport, using a fake `claude` executable (no network, no real login)."""

import asyncio
import json
import os
import stat
import sys
import textwrap
import uuid

import pytest

from explorer.claude_backend import (
    ChatHistoryError,
    ClaudeBackend,
    ClaudeChatHistory,
    ClaudeEventTranslator,
    build_claude_argv,
)

MODEL_NAMESPACE = "meta-llama-llama-3.2-3b-instruct-1642e64c6774"
SID = "af5ac06a-d8b1-4ad3-8f0a-a8867802b065"

FAKE_CLAUDE = textwrap.dedent('''\
    #!{python}
    import json, os, sys, time
    argv = sys.argv[1:]
    stdin = sys.stdin.read()
    with open(os.environ["FAKE_CLAUDE_LOG"], "a") as log:
        log.write(json.dumps({{"argv": argv, "stdin": stdin, "cwd": os.getcwd()}}) + "\\n")
    mode = os.environ.get("FAKE_CLAUDE_MODE", "ok")
    sid = argv[argv.index("--session-id") + 1] if "--session-id" in argv else argv[argv.index("--resume") + 1]

    def emit(obj):
        sys.stdout.write(json.dumps(obj) + "\\n")
        sys.stdout.flush()

    if mode == "missing" and "--resume" in argv:
        emit({{"type": "result", "subtype": "error_during_execution", "is_error": True,
              "result": "No conversation found with session ID: " + sid, "session_id": sid}})
        sys.exit(1)
    if mode == "notlogged":
        sys.stderr.write("Not logged in - please run /login\\n")
        sys.exit(1)
    emit({{"type": "system", "subtype": "init", "session_id": sid, "tools": []}})
    emit({{"type": "stream_event", "session_id": sid, "api_message_id": "m1",
          "event": {{"type": "content_block_delta", "index": 0, "delta": {{"type": "text_delta", "text": "Looking"}}}}}})
    if mode == "hang":
        time.sleep(60)
    emit({{"type": "assistant", "session_id": sid, "message": {{"id": "m1", "content": [
        {{"type": "text", "text": "Looking"}},
        {{"type": "tool_use", "id": "toolu_1", "name": "mcp__explorer__neuron", "input": {{"layer": 2, "n": 4}}}}]}}}})
    emit({{"type": "user", "session_id": sid, "message": {{"content": [
        {{"type": "tool_result", "tool_use_id": "toolu_1", "is_error": False,
          "content": [{{"type": "text", "text": "name api_key=sk-abcdefghijklmnopqrstuv"}}]}}]}}}})
    emit({{"type": "stream_event", "session_id": sid, "api_message_id": "m2",
          "event": {{"type": "content_block_delta", "index": 0, "delta": {{"type": "text_delta", "text": " done."}}}}}})
    emit({{"type": "result", "subtype": "success", "is_error": False, "result": "Looking done.",
          "session_id": sid, "total_cost_usd": 0.01}})
    ''')


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    script = tmp_path / "fake-claude"
    script.write_text(FAKE_CLAUDE.format(python=sys.executable))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ok")

    def calls():
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    return script, calls


def make_backend(tmp_path, script, **kwargs):
    return ClaudeBackend(
        tmp_path / "cache",
        MODEL_NAMESPACE,
        claude_bin=str(script),
        mcp_command=sys.executable,
        mcp_args=["/opt/explorer/agent_mcp.py"],
        mcp_env={"EXPLORER_URL": "http://127.0.0.1:8791"},
        **kwargs,
    )


def flag(argv, name):
    return argv[argv.index(name) + 1]


def test_argv_restricts_tools_to_explorer_mcp_and_keeps_text_out_of_argv():
    argv = build_claude_argv(
        claude_bin="/opt/claude",
        model="sonnet",
        system_prompt="Be careful.",
        session_id=SID,
        resume=False,
        mcp_command=sys.executable,
        mcp_args=["/opt/agent mcp.py"],
        mcp_env={"EXPLORER_CHAT": "chat_1", "EXPLORER_URL": "http://127.0.0.1:8791"},
    )
    assert argv[:2] == ["/opt/claude", "-p"]
    assert flag(argv, "--output-format") == "stream-json" and "--verbose" in argv
    assert flag(argv, "--input-format") == "stream-json"
    assert "--strict-mcp-config" in argv
    assert flag(argv, "--tools") == ""  # no Bash, Edit, Write, WebFetch or any other built-in
    assert flag(argv, "--allowedTools") == "mcp__explorer"
    assert flag(argv, "--disallowedTools") == "mcp__explorer__set_name"
    assert flag(argv, "--permission-mode") == "dontAsk"
    assert flag(argv, "--setting-sources") == "project"
    assert flag(argv, "--model") == "sonnet"
    assert flag(argv, "--system-prompt") == "Be careful."
    assert flag(argv, "--session-id") == SID and "--resume" not in argv
    config = json.loads(flag(argv, "--mcp-config"))
    server = config["mcpServers"]["explorer"]
    assert server["command"] == sys.executable and server["args"] == ["/opt/agent mcp.py"]
    assert server["env"] == {"EXPLORER_CHAT": "chat_1", "EXPLORER_URL": "http://127.0.0.1:8791"}
    assert "--dangerously-skip-permissions" not in argv

    resumed = build_claude_argv(
        session_id=SID, resume=True, mcp_command="python", mcp_args=[], mcp_env={},
    )
    assert flag(resumed, "--resume") == SID and "--session-id" not in resumed
    assert "--model" not in resumed and "--system-prompt" not in resumed


def test_argv_rejects_unsafe_values():
    base = {"session_id": SID, "resume": False, "mcp_command": "python", "mcp_args": [], "mcp_env": {}}
    with pytest.raises(ValueError, match="MCP server name"):
        build_claude_argv(mcp_server_name="../x", **base)
    with pytest.raises(ValueError, match="dash"):
        build_claude_argv(model="--evil", **base)
    with pytest.raises(ValueError):
        build_claude_argv(**{**base, "session_id": "not-a-uuid"})


def test_translator_streams_text_tools_results_and_terminal_events():
    translator = ClaudeEventTranslator()
    events = []
    for raw in [
        {"type": "system", "subtype": "init", "session_id": SID},
        {"type": "stream_event", "api_message_id": "m1",
         "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}}},
        {"type": "assistant", "message": {"id": "m1", "content": [
            {"type": "text", "text": "Hi"},
            {"type": "tool_use", "id": "t1", "name": "mcp__explorer__overview", "input": {}}]}},
        {"type": "assistant", "message": {"id": "m1", "content": [
            {"type": "tool_use", "id": "t1", "name": "mcp__explorer__overview", "input": {}}]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "is_error": True, "content": "boom"}]}},
        {"type": "assistant", "parent_tool_use_id": "t9", "message": {"id": "s", "content": [
            {"type": "text", "text": "subagent"}]}},
        {"type": "result", "subtype": "success", "is_error": False, "result": "Hi", "total_cost_usd": 0.5},
    ]:
        events.extend(translator.feed(raw))
    assert translator.session_id == SID and translator.done and not translator.failed
    assert events == [
        {"t": "text", "text": "Hi"},
        {"t": "tool", "id": "t1", "name": "overview", "input": {}},
        {"t": "tool_result", "id": "t1", "error": True, "output": "boom"},
        {"t": "done", "cost": 0.5},
    ]

    # no partial messages: the whole assistant text block is used, and the result text is not duplicated
    whole = ClaudeEventTranslator()
    out = whole.feed({"type": "assistant", "message": {"id": "x", "content": [{"type": "text", "text": "Whole"}]}})
    out += whole.feed({"type": "result", "subtype": "success", "result": "Whole"})
    assert [e for e in out if e["t"] == "text"] == [{"t": "text", "text": "Whole"}]

    failed = ClaudeEventTranslator()
    out = failed.feed({"type": "result", "subtype": "error_during_execution", "is_error": True,
                       "result": "bad access_token=abcd1234"})
    assert failed.failed and out[0]["t"] == "error" and "abcd1234" not in out[0]["error"]
    assert out[-1] == {"t": "done"}


async def collect(backend, *args, **kwargs):
    return [event async for event in backend.stream_turn(*args, **kwargs)]


@pytest.mark.asyncio
async def test_turn_streams_events_persists_scoped_history_and_resumes(tmp_path, fake_claude):
    script, calls = fake_claude
    backend = make_backend(tmp_path, script)
    events = await collect(backend, "chat-1", "What is L2 N4?", system_prompt="You are the explorer agent.",
                           context="L2 N4 is open.", attachments=["<file name=a.txt>note</file>"])

    assert [e["t"] for e in events] == ["chat", "text", "tool", "tool_result", "text", "done"]
    assert events[2] == {"t": "tool", "id": "toolu_1", "name": "neuron", "input": {"layer": 2, "n": 4}}
    assert "sk-abcdefghijklmnopqrstuv" not in events[3]["output"] and "[redacted]" in events[3]["output"]
    assert events[-1]["cost"] == 0.01

    first = calls()[0]
    assert first["cwd"] == str(backend.cwd.resolve()) or os.path.samefile(first["cwd"], backend.cwd)
    assert backend.cwd.parts[-2:] == (MODEL_NAMESPACE, "agent")
    session = flag(first["argv"], "--session-id")
    assert flag(first["argv"], "--system-prompt") == "You are the explorer agent."
    assert json.loads(flag(first["argv"], "--mcp-config"))["mcpServers"]["explorer"]["env"]["EXPLORER_CHAT"] == "chat-1"
    sent = json.loads(first["stdin"])
    assert sent["type"] == "user" and sent["message"]["role"] == "user"
    texts = [b["text"] for b in sent["message"]["content"]]
    assert "Explorer context: L2 N4 is open." in texts[0] and "What is L2 N4?" in texts[0]
    assert texts[1] == "<file name=a.txt>note</file>"

    stored = backend.history.load("chat-1")
    assert stored["claude_session_id"] == session
    turn = stored["turns"][0]
    assert turn["assistant"] == "Looking done." and turn["status"] == "complete"
    assert "sk-abcdefghijklmnopqrstuv" not in json.dumps(stored)
    assert backend.history.history_dir.parts[-3:] == (MODEL_NAMESPACE, "agent", "chats")

    await collect(backend, "chat-1", "And N5?")
    second = calls()[1]
    assert flag(second["argv"], "--resume") == session and "--session-id" not in second["argv"]
    body = json.loads(second["stdin"])["message"]["content"][0]["text"]
    assert body == "And N5?"  # resumed: no history replay
    assert len(backend.history.load("chat-1")["turns"]) == 2
    assert not backend.is_chat_active("chat-1")


@pytest.mark.asyncio
async def test_legacy_history_without_session_is_replayed_into_a_new_session(tmp_path, fake_claude):
    script, calls = fake_claude
    backend = make_backend(tmp_path, script)
    backend.history.append_turn("legacy", message="Earlier question", assistant="Earlier answer")
    assert backend.history.session_id("legacy") is None

    await collect(backend, "legacy", "Follow up")
    call = calls()[0]
    assert "--session-id" in call["argv"]
    text = json.loads(call["stdin"])["message"]["content"][0]["text"]
    assert "Earlier question" in text and "Earlier answer" in text and "Follow up" in text
    assert backend.history.session_id("legacy") == flag(call["argv"], "--session-id")


@pytest.mark.asyncio
async def test_rewind_drops_the_session_so_removed_turns_do_not_resurface(tmp_path, fake_claude):
    script, calls = fake_claude
    backend = make_backend(tmp_path, script)
    await collect(backend, "chat-r", "first")
    await collect(backend, "chat-r", "second")
    assert backend.rewind_chat("chat-r", "second") == {"ok": True, "cut": True}
    assert backend.history.session_id("chat-r") is None

    await collect(backend, "chat-r", "second again")
    call = calls()[-1]
    assert "--session-id" in call["argv"] and "--resume" not in call["argv"]
    text = json.loads(call["stdin"])["message"]["content"][0]["text"]
    assert "first" in text and "second again" in text and text.count("second") == 1


@pytest.mark.asyncio
async def test_missing_session_falls_back_to_a_fresh_one_seeded_from_history(tmp_path, fake_claude, monkeypatch):
    script, calls = fake_claude
    backend = make_backend(tmp_path, script)
    await collect(backend, "chat-m", "hello")
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "missing")
    events = await collect(backend, "chat-m", "again")
    # the fake still reports the session missing for --resume only; the retry uses --session-id and succeeds
    assert [e["t"] for e in events if e["t"] == "error"] == []
    assert events[-1]["t"] == "done"
    argvs = [c["argv"] for c in calls()]
    assert "--resume" in argvs[1] and "--session-id" in argvs[2]
    assert "hello" in json.loads(calls()[2]["stdin"])["message"]["content"][0]["text"]
    assert len(backend.history.load("chat-m")["turns"]) == 2


@pytest.mark.asyncio
async def test_failures_surface_errors_and_do_not_store_turns(tmp_path, fake_claude, monkeypatch):
    script, _calls = fake_claude
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "notlogged")
    backend = make_backend(tmp_path, script)
    events = await collect(backend, "chat-e", "hi")
    assert events[1]["t"] == "error" and "not signed in" in events[1]["error"]
    assert events[-1] == {"t": "done"}
    assert backend.history.load("chat-e")["turns"] == []
    assert not backend.is_chat_active("chat-e")

    missing = ClaudeBackend(tmp_path / "cache", MODEL_NAMESPACE, claude_bin=str(tmp_path / "nope"))
    events = await collect(missing, "chat-x", "hi")
    assert events[1]["t"] == "error" and events[-1] == {"t": "done"}


@pytest.mark.asyncio
async def test_stop_kills_process_group_and_keeps_partial_turn_with_session(tmp_path, fake_claude, monkeypatch):
    script, _calls = fake_claude
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "hang")
    backend = make_backend(tmp_path, script)
    seen = asyncio.Event()
    procs = []

    async def consume():
        async for event in backend.stream_turn("chat-s", "Question", on_process=procs.append):
            if event["t"] == "text":
                seen.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(seen.wait(), timeout=20)
    proc = next(p for p in procs if p is not None)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert proc.returncode is not None and procs[-1] is None
    assert not backend.is_chat_active("chat-s")
    stored = backend.history.load("chat-s")
    turn = stored["turns"][0]
    assert turn["interrupted"] is True and turn["assistant"] == "Looking"
    assert uuid.UUID(stored["claude_session_id"])


@pytest.mark.asyncio
async def test_reservation_blocks_duplicate_turns_and_rewind(tmp_path, fake_claude):
    script, calls = fake_claude
    backend = make_backend(tmp_path, script)
    assert backend.reserve_chat("busy") and not backend.reserve_chat("busy")
    assert backend.rewind_chat("busy", "x")["ok"] is False
    events = await collect(backend, "busy", "hi")
    assert events[1]["t"] == "error" and "already active" in events[1]["error"]
    assert calls() == []
    backend.release_chat("busy")
    assert not backend.is_chat_active("busy")
    with pytest.raises(ChatHistoryError):
        backend.reserve_chat("../bad")


@pytest.mark.asyncio
async def test_images_are_confined_to_upload_root_and_sent_as_base64_blocks(tmp_path, fake_claude):
    script, calls = fake_claude
    backend = make_backend(tmp_path, script)
    upload_root = backend.upload_root
    upload_root.mkdir(parents=True)
    image = upload_root / "sample.png"
    image.write_bytes(b"\x89PNGdata")
    events = await collect(backend, "chat-i", "Look", image_paths=[image])
    assert events[-1]["t"] == "done"
    blocks = json.loads(calls()[0]["stdin"])["message"]["content"]
    assert blocks[-1]["type"] == "image" and blocks[-1]["source"]["media_type"] == "image/png"

    outside = tmp_path / "outside.png"
    outside.write_bytes(b"x")
    before = len(calls())
    events = await collect(backend, "chat-o", "Look", image_paths=[outside])
    assert "local upload cache" in events[1]["error"] and len(calls()) == before
    events = await collect(backend, "chat-a", "Look", attachments=[{"type": "document", "source": {}}])
    assert events[1]["t"] == "error" and len(calls()) == before


def test_history_scoping_and_legacy_compatibility(tmp_path):
    directory = tmp_path / "chats"
    ClaudeChatHistory(directory, MODEL_NAMESPACE).append_turn("c", message="m", assistant="a", session_id=SID)
    other = ClaudeChatHistory(directory, "another-model-0000")
    with pytest.raises(ChatHistoryError, match="different model"):
        other.load("c")
    # a file written by the Codex transport (no session id) stays readable
    legacy = ClaudeChatHistory(directory, MODEL_NAMESPACE)
    (directory / "old.json").write_text(json.dumps(
        {"schema_version": 1, "model_namespace": MODEL_NAMESPACE, "turns": [{"message": "q", "assistant": "r"}]}))
    assert legacy.load("old")["turns"][0]["message"] == "q" and legacy.session_id("old") is None
    (directory / "bad.json").write_text("{not json")
    with pytest.raises(ChatHistoryError):
        legacy.load("bad")
