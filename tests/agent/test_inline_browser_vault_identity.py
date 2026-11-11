"""Inline vault overrides receive the same call identity as registry tools (#153172)."""

import json
from types import SimpleNamespace

import pytest

from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext


@pytest.fixture
def vault_registry(monkeypatch):
    import tools.browser_vault_tool  # register the built-in before installing the override
    import model_tools
    import tools.registry as registry_module

    registry = registry_module.ToolRegistry()
    monkeypatch.setattr(registry_module, "registry", registry)
    monkeypatch.setattr(model_tools, "registry", registry)
    return registry


def _agent(session):
    return SimpleNamespace(
        session_id=session, _current_turn_id=f"turn-{session}",
        _inherited_conversation_id=f"root-{session}",
        drive_preview_callback=lambda: None,
    )


def _observed(kwargs):
    from tools import approval_context

    return {
        "session": kwargs.get("session_id"),
        "conversation": kwargs.get("conversation_id"),
        "turn": approval_context._approval_turn_id.get(),
        "call": approval_context._approval_tool_call_id.get(),
        "approval_session": approval_context._approval_session_id.get(),
    }


@pytest.mark.parametrize("asynchronous", [False, True])
def test_inline_vault_has_registry_call_identity(vault_registry, asynchronous):
    from model_tools import _CallIds, _execute_tool

    def handler(args, **kwargs):
        return json.dumps(_observed(kwargs))

    async def async_handler(args, **kwargs):
        return handler(args, **kwargs)

    vault_registry.register(
        name="browser_vault", toolset="identity-test", schema={"name": "browser_vault"},
        handler=async_handler if asynchronous else handler, is_async=asynchronous, override=True,
    )
    agent = _agent("session-a")
    ctx = InlineToolContext(effective_task_id="task-a", tool_call_id="call-a")
    expected = {
        "session": "session-a", "conversation": "root-session-a",
        "turn": "turn-session-a", "call": "call-a", "approval_session": "session-a",
    }
    control = _execute_tool(
        "browser_vault", {"action": "list"}, {"action": "list"},
        _CallIds(task_id="task-a", session_id=agent.session_id, conversation_id="root-session-a",
                 turn_id=agent._current_turn_id, tool_call_id=ctx.tool_call_id),
        user_task=None, enabled_tools=None, skip_tool_execution_middleware=True,
    )
    assert json.loads(control) == expected
    result = INLINE_TOOL_EXECUTORS["browser_vault"](agent, {"action": "list"}, ctx)
    assert json.loads(result) == expected

    def narrow_handler(args, task_id, preview_callback):
        return json.dumps({"task": task_id, "callback": preview_callback is agent.drive_preview_callback,
                           "target": args.get("target")})

    vault_registry.register(
        name="browser_vault", toolset="identity-test", schema={"name": "browser_vault"},
        handler=narrow_handler, override=True,
    )
    args = {"action": "fill", "target": "preview"}
    assert json.loads(INLINE_TOOL_EXECUTORS["browser_vault"](agent, args, ctx)) == {
        "task": "task-a", "callback": True, "target": "preview",
    }
    assert args == {"action": "fill", "target": "preview"}


@pytest.mark.parametrize("inner_failure", [False, True])
def test_inline_vault_restores_nested_identity(vault_registry, inner_failure):
    from model_tools import _CallIds, _approval_observability

    observations = []

    def handler(args, **kwargs):
        observations.append(_observed(kwargs))
        if args["handle"] == "outer":
            inner = INLINE_TOOL_EXECUTORS["browser_vault"](
                _agent("session-b"), {"action": "list", "handle": "inner"},
                InlineToolContext(effective_task_id="task-b", tool_call_id="call-b"),
            )
            assert bool(json.loads(inner).get("error")) == inner_failure
            observations.append(_observed(kwargs))
        elif inner_failure:
            raise ValueError("synthetic handler refusal")
        return json.dumps({"ok": True})

    vault_registry.register(
        name="browser_vault", toolset="identity-test", schema={"name": "browser_vault"},
        handler=handler, override=True,
    )
    with _approval_observability(_CallIds(session_id="outer", turn_id="outer-turn", tool_call_id="outer-call")):
        before = _observed({})
        result = INLINE_TOOL_EXECUTORS["browser_vault"](
            _agent("session-a"), {"action": "list", "handle": "outer"},
            InlineToolContext(effective_task_id="task-a", tool_call_id="call-a"),
        )
        assert json.loads(result) == {"ok": True}
        assert _observed({}) == before
    assert [o["approval_session"] for o in observations] == ["session-a", "session-b", "session-a"]
    assert [o["call"] for o in observations] == ["call-a", "call-b", "call-a"]
