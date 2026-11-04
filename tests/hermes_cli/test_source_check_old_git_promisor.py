"""Passive downgrade checks must not trigger old Git's promisor lazy fetch."""

from types import SimpleNamespace

import pytest

from hermes_cli import source_check, update_downgrade_guard


@pytest.mark.parametrize(
    "remote_verdict,held", [(True, True), (False, False), (None, False)]
)
def test_old_git_partial_clone_uses_remote_verdict_without_git_probes(
    tmp_path, monkeypatch, remote_verdict, held
):
    co = SimpleNamespace(
        root=tmp_path, git="git", embedded=None, repository="NousResearch/hermes-agent"
    )
    home = tmp_path / "home"
    monkeypatch.setattr(
        update_downgrade_guard, "_ledgered_profiles", lambda _: [("", home)]
    )

    def git_stdout(args, **_kwargs):
        if args == ["--version"]:
            return "git version 2.39.5 (Apple Git-154)"
        if args[0] == "config":
            return "remote.origin.promisor true"
        raise AssertionError(f"unexpected Git probe: {args}")

    monkeypatch.setattr(source_check, "_git_stdout", git_stdout)
    monkeypatch.setattr(
        source_check, "_git_run", lambda *args, **kwargs: pytest.fail("lazy-fetch risk")
    )
    monkeypatch.setattr(
        source_check, "_github_predates_runtime", lambda *args: remote_verdict
    )

    assert source_check._update_refuses(co, "a" * 40, home) is held


def test_old_git_full_clone_keeps_local_forward_check(tmp_path, monkeypatch):
    co = SimpleNamespace(
        root=tmp_path, git="git", embedded=None, repository="NousResearch/hermes-agent"
    )
    home = tmp_path / "home"
    monkeypatch.setattr(
        update_downgrade_guard, "_ledgered_profiles", lambda _: [("", home)]
    )
    monkeypatch.setattr(
        source_check,
        "_git_stdout",
        lambda args, **_kwargs: "git version 2.39.5" if args == ["--version"] else None,
    )
    seen = []

    def git_run(args, **_kwargs):
        seen.append(args)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(source_check, "_git_run", git_run)
    monkeypatch.setattr(
        source_check,
        "_github_predates_runtime",
        lambda *args: pytest.fail("remote needed"),
    )

    assert source_check._update_refuses(co, "a" * 40, home) is False
    assert seen == [["merge-base", "--is-ancestor", "HEAD", "a" * 40]]
