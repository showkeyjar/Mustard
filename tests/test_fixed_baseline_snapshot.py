from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from scripts.evaluate_pretraining import build_runner_from_state_dir
from scripts.evaluate_real_prompts import evaluate_isolated_prompts


def _write_controls(path: Path, sections: dict) -> None:
    path.write_text(json.dumps(sections, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# build_runner_from_state_dir: use_snapshot_controls semantics
# ---------------------------------------------------------------------------


def test_build_runner_uses_snapshot_controls_when_present():
    snapshot_controls = {
        "policy": {"_snapshot_marker": 123},
        "glance": {},
        "core": {},
    }
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        snapshot = tmp / "snapshot"
        snapshot.mkdir()
        _write_controls(snapshot / "runtime_controls.json", snapshot_controls)
        (snapshot / "policy_state.json").write_text("{}", encoding="utf-8")

        ws = tmp / "ws"
        runner = build_runner_from_state_dir(snapshot, ws)
        out = json.loads((ws / "runtime_controls.json").read_text(encoding="utf-8"))
        # Baseline reflects the PINNED policy, not the live global file.
        assert out["policy"]["_snapshot_marker"] == 123
        assert runner is not None


def test_build_runner_falls_back_to_global_without_snapshot_controls():
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        artifact = tmp / "artifact"  # no runtime_controls.json
        artifact.mkdir()
        (artifact / "policy_state.json").write_text("{}", encoding="utf-8")

        ws = tmp / "ws"
        build_runner_from_state_dir(artifact, ws)
        out = json.loads((ws / "runtime_controls.json").read_text(encoding="utf-8"))
        # No snapshot -> global file is used (no snapshot marker present).
        assert "_snapshot_marker" not in out.get("policy", {})


def test_build_runner_override_wins_over_snapshot():
    snapshot_controls = {
        "policy": {"_snapshot_marker": 1},
        "glance": {},
        "core": {},
    }
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        snapshot = tmp / "snapshot"
        snapshot.mkdir()
        _write_controls(snapshot / "runtime_controls.json", snapshot_controls)
        (snapshot / "policy_state.json").write_text("{}", encoding="utf-8")

        ws = tmp / "ws"
        build_runner_from_state_dir(
            snapshot, ws, override_controls={"policy": {"_override_marker": True}}
        )
        out = json.loads((ws / "runtime_controls.json").read_text(encoding="utf-8"))
        # Explicit override is applied; the snapshot baseline is NOT used.
        assert out["policy"].get("_override_marker") is True
        assert "_snapshot_marker" not in out["policy"]


# ---------------------------------------------------------------------------
# evaluate_isolated_prompts: baseline source switches on snapshot presence
# ---------------------------------------------------------------------------


def _fake_runner_for_workspace(workspace, baseline_tool, pretrained_tool):
    tool = pretrained_tool if "pretrained" in str(workspace) else baseline_tool
    trace = SimpleNamespace(
        steps=[SimpleNamespace(selected_tool=tool)],
        actions=[],
    )
    return SimpleNamespace(run=lambda prompt: ("answer", trace))


def test_evaluate_isolated_prompts_baseline_source_switches_on_snapshot():
    calls: list = []

    def fake_build(source_dir, workspace, override_controls=None):
        calls.append(source_dir)
        return _fake_runner_for_workspace(workspace, "search_tool", "search_tool")

    prompts = [{"id": "p1", "prompt": "q1", "expected_tool": "search_tool"}]
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        snapshot = tmp / "snapshot"
        snapshot.mkdir()
        _write_controls(
            snapshot / "runtime_controls.json",
            {"policy": {}, "glance": {}, "core": {}},
        )
        with mock.patch(
            "scripts.evaluate_real_prompts.build_runner_from_state_dir",
            side_effect=fake_build,
        ):
            with mock.patch(
                "scripts.evaluate_real_prompts.BASELINE_SNAPSHOT_DIR", snapshot
            ):
                res = evaluate_isolated_prompts(prompts, artifact_dir=tmp / "artifact")
            # First call is the baseline runner -> loads the snapshot dir.
            assert calls[0] == snapshot
            assert res["delta_tool_match_rate"] == 0.0

        # Snapshot absent -> baseline source falls back to None (legacy behavior).
        calls.clear()
        with mock.patch(
            "scripts.evaluate_real_prompts.build_runner_from_state_dir",
            side_effect=fake_build,
        ):
            with mock.patch(
                "scripts.evaluate_real_prompts.BASELINE_SNAPSHOT_DIR",
                tmp / "does_not_exist",
            ):
                evaluate_isolated_prompts(prompts, artifact_dir=tmp / "artifact")
        # baseline is the FIRST call of this second evaluation
        assert calls[0] is None


def test_evaluate_isolated_prompts_delta_reflects_baseline_vs_pretrained():
    calls: list = []

    def fake_build(source_dir, workspace, override_controls=None):
        calls.append(source_dir)
        # Baseline picks search_tool; pretrained picks calc_tool -> a real DeltaP.
        return _fake_runner_for_workspace(workspace, "search_tool", "calc_tool")

    prompts = [{"id": "p1", "prompt": "q1", "expected_tool": "search_tool"}]
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        snapshot = tmp / "snapshot"
        snapshot.mkdir()
        _write_controls(
            snapshot / "runtime_controls.json",
            {"policy": {}, "glance": {}, "core": {}},
        )
        with mock.patch(
            "scripts.evaluate_real_prompts.build_runner_from_state_dir",
            side_effect=fake_build,
        ):
            with mock.patch(
                "scripts.evaluate_real_prompts.BASELINE_SNAPSHOT_DIR", snapshot
            ):
                res = evaluate_isolated_prompts(prompts, artifact_dir=tmp / "artifact")

        # baseline search==search True, pretrained calc==search False -> delta -1.
        assert res["rows"][0]["delta"] == -1
        assert res["delta_tool_match_rate"] == -1.0
