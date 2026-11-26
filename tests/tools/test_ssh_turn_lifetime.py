"""Regression for #160417: SSH shell state survives turn cleanup, not session teardown.

SSH transport is replaced with local bash; construction, snapshots, configuration,
environment lookup, terminal dispatch and lifecycle cleanup are real.
"""

import json
import shutil
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.chat_completion_helpers import cleanup_task_resources
from tools import terminal_tool
from tools.environments import ssh
from tools.environments.local import LocalEnvironment
from tools.terminal_tool_lifecycle import (
    _cleanup_inactive_envs,
    cleanup_vm,
    ensure_task_env,
    get_active_env,
)


@pytest.fixture
def local_ssh(monkeypatch, tmp_path):
    bash = shutil.which("bash")
    assert bash is not None
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.setenv("TERMINAL_SSH_HOST", "fixture.invalid")
    monkeypatch.setenv("TERMINAL_SSH_USER", "fixture")
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", True)
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(ssh, "_ensure_ssh_available", lambda: None)
    monkeypatch.setattr(ssh.SSHEnvironment, "_establish_connection", lambda self: None)
    monkeypatch.setattr(ssh.SSHEnvironment, "_detect_remote_home", lambda self: str(tmp_path))
    monkeypatch.setattr(ssh.SSHEnvironment, "_ensure_remote_dirs", lambda self: None)
    monkeypatch.setattr(ssh.SSHEnvironment, "get_temp_dir", lambda self: str(tmp_path))
    monkeypatch.setattr(ssh, "FileSyncManager", MagicMock())

    def transport(self, command, *, login=False, timeout=120, stdin_data=None):
        return ssh._popen_bash([bash, "-c", command], stdin_data)

    monkeypatch.setattr(ssh.SSHEnvironment, "_run_bash", transport)
    monkeypatch.setattr("tools.browser_session_cloud.is_headed_mode", lambda: False)
    monkeypatch.setattr("run_agent.cleanup_browser", MagicMock())
    yield
    for key in list(terminal_tool._active_environments):
        cleanup_vm(key)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("setting", [None, "true", "false"])
def test_ssh_survives_turns_only_when_persistent(local_ssh, monkeypatch, setting):
    if setting is None:
        monkeypatch.delenv("TERMINAL_SSH_PERSISTENT", raising=False)
        monkeypatch.delenv("TERMINAL_PERSISTENT_SHELL", raising=False)
    else:
        monkeypatch.setenv("TERMINAL_SSH_PERSISTENT", setting)
    task = "ssh-turn-lifetime"
    key = terminal_tool._resolve_container_task_id(task)

    def run(command):
        result = json.loads(terminal_tool.terminal_tool(command, task_id=task))
        assert result["exit_code"] == 0, result
        return result["output"].strip()

    run("export SSH_TURN_MARKER=kept; printf artifact > marker.txt")
    first = get_active_env(task)
    assert isinstance(first, ssh.SSHEnvironment)
    cleanup_task_resources(SimpleNamespace(verbose_logging=True), key)
    persistent = setting != "false"
    assert get_active_env(task) is (first if persistent else None)
    assert run('printf "%s:" "${SSH_TURN_MARKER:-none}"; cat marker.txt') == (
        "kept:artifact" if persistent else "none:artifact"
    )
    assert (get_active_env(task) is first) is persistent

    cleanup_vm(key)
    assert get_active_env(task) is None
    assert ensure_task_env(task) is not first
    terminal_tool._last_activity[key] = time.time() - 301
    _cleanup_inactive_envs(lifetime_seconds=300)
    assert get_active_env(task) is None


@pytest.mark.platforms("posix")
def test_shell_snapshot_persistence_keeps_local_per_turn_cleanup(local_ssh, monkeypatch, tmp_path):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    env = LocalEnvironment(cwd=str(tmp_path), persistent=True)
    assert env._persist_env is True
    key = terminal_tool._resolve_container_task_id("local-turn-lifetime")
    terminal_tool._active_environments[key] = env
    cleanup_task_resources(SimpleNamespace(verbose_logging=True), key)
    assert get_active_env("local-turn-lifetime") is None
