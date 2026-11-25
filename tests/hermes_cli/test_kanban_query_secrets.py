"""Kanban queries read persisted cards without resolving secret sources (#158046)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _run(args: list[str], home: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        env={
            **os.environ,
            "HOME": str(home.parent),
            "HERMES_HOME": str(home),
            "HERMES_KANBAN_HOME": str(home),
            "HERMES_DISABLE_LAZY_INSTALLS": "1",
        },
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture
def board(tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    hits = tmp_path / "source-hits"
    helper = tmp_path / "source.py"
    helper.write_text(
        "from pathlib import Path\n"
        f"with Path({str(hits)!r}).open('a') as out: out.write('called\\n')\n"
        "print('KANBAN_FIXTURE_TOKEN=synthetic-fixture')\n",
        encoding="utf-8",
    )
    command = shlex.join([sys.executable, str(helper)])
    (home / "config.yaml").write_text(
        json.dumps(
            {
                "secrets": {"command": {"enabled": True, "command": command}},
            }
        ),
        encoding="utf-8",
    )
    board_home = home / "board-home"
    selected = board_home / "kanban.db"
    (home / ".env").write_text(f"HERMES_KANBAN_HOME={board_home}\n", encoding="utf-8")
    created = _run(["kanban", "create", "query control", "--triage", "--json"], home)
    task = json.loads(created.stdout)
    assert selected.is_file(), "the ordinary dotenv layer must select the board"
    assert hits.read_text().splitlines() == ["called"]
    return home, hits, task["id"]


@pytest.mark.platforms("posix")  # the configured command source uses /bin/sh
@pytest.mark.parametrize(
    "query",
    [
        ["list"],
        ["ls"],
        ["show"],
        ["--board", "default", "list"],
        ["--board=default", "show"],
    ],
)
def test_queries_keep_card_and_dotenv_selection_without_resolving_sources(board, query):
    home, hits, task_id = board
    args = ["kanban", *query]
    if query[-1] == "show":
        args.append(task_id)
    result = _run([*args, "--json"], home)
    payload = json.loads(result.stdout)
    card = payload[0] if isinstance(payload, list) else payload["task"]
    assert card["id"] == task_id
    assert hits.read_text().splitlines() == ["called"]
    assert "applied 1 secret" not in result.stderr


@pytest.mark.platforms("posix")
def test_query_optimization_keeps_source_loading_for_writes(board):
    home, hits, _ = board
    _run(["kanban", "create", "second control", "--triage", "--json"], home)
    assert hits.read_text().splitlines() == ["called", "called"]
    _run(["--version"], home)
    assert hits.read_text().splitlines() == ["called", "called"]
