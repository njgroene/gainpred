"""Tests for gainpred.launch's safety checks (no Modal calls)."""

import pytest

from gainpred import launch


def test_refuses_to_overwrite_an_existing_run(monkeypatch):
    monkeypatch.setattr(launch, "code_is_committed", lambda: True)
    monkeypatch.setattr(launch, "runs_on_volume", lambda: {"si-old"})
    with pytest.raises(SystemExit, match="never overwritten"):
        launch.spawn("self_improve", "si-old", {})


def test_refuses_a_name_spawned_before(monkeypatch):
    monkeypatch.setattr(launch, "code_is_committed", lambda: True)
    monkeypatch.setattr(launch, "runs_on_volume", lambda: set())
    monkeypatch.setattr(launch, "records", lambda: [{"run": "si-x", "call_id": "fc-1"}])
    with pytest.raises(SystemExit, match="never overwritten"):
        launch.spawn("self_improve", "si-x", {})


def test_refuses_uncommitted_code(monkeypatch):
    monkeypatch.setattr(launch, "code_is_committed", lambda: False)
    with pytest.raises(SystemExit, match="uncommitted"):
        launch.spawn("self_improve", "si-new", {})


def test_longest_run_launch_compares_the_longest_run_manifest(monkeypatch, tmp_path):
    """runs-main / runs-rounds check data/longest_run/manifest.json, after the other safety checks."""
    monkeypatch.setattr(launch, "code_is_committed", lambda: True)
    monkeypatch.setattr(launch, "runs_on_volume", lambda: set())
    monkeypatch.setattr(launch, "records", lambda: [])
    monkeypatch.setattr(launch, "git", lambda *a: "abc1234")
    monkeypatch.setattr(launch, "ensure_deployed", lambda commit: None)
    monkeypatch.setitem(launch.DATA, "longest_run", ("make_run_data", tmp_path / "missing.json"))
    with pytest.raises(SystemExit, match="longest_run data locally"):
        launch.spawn("train_start_runs", "runs-new", {}, task="longest_run")
