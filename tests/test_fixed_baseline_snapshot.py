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


# ---------------------------------------------------------------------------
# End-to-end: the REAL build_runner_from_state_dir copies the snapshot controls
# into the baseline workspace, so the baseline runner genuinely reflects the
# pinned policy rather than the live global file. This is what makes the
# DeltaP signal *discriminative* (phase one's whole point).
# ---------------------------------------------------------------------------


def _real_runner_factory_that_reads_workspace(workspace, expected_tool, marker_tool):
    """Wrap the real builder so the workspace controls file is actually written,
    then return a fake runner whose tool choice depends on that file's content."""
    real_runner = build_runner_from_state_dir(
        # source_dir / workspace are injected by the patched evaluate_isolated_prompts
        # call; we capture them via closure below.
        _real_runner_factory_that_reads_workspace._current_source,
        workspace,
        _real_runner_factory_that_reads_workspace._current_override,
    )
    controls = json.loads(
        (workspace / "runtime_controls.json").read_text(encoding="utf-8")
    )
    marker = controls.get("policy", {}).get("_snapshot_marker")
    chosen = marker_tool if marker else "global_tool"
    real_runner.run = lambda prompt: (
        "answer",
        SimpleNamespace(
            steps=[SimpleNamespace(selected_tool=chosen)], actions=[]
        ),
    )
    return real_runner


def test_end_to_end_snapshot_controls_make_baseline_distinct():
    def fake_build(source_dir, workspace, override_controls=None):
        _real_runner_factory_that_reads_workspace._current_source = source_dir
        _real_runner_factory_that_reads_workspace._current_override = override_controls
        return _real_runner_factory_that_reads_workspace(workspace, "search_tool", "snapshot_tool")

    prompts = [{"id": "p1", "prompt": "q1", "expected_tool": "snapshot_tool"}]
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        snapshot = tmp / "snapshot"
        snapshot.mkdir()
        _write_controls(
            snapshot / "runtime_controls.json",
            {"policy": {"_snapshot_marker": 1}, "glance": {}, "core": {}},
        )
        (snapshot / "policy_state.json").write_text("{}", encoding="utf-8")
        artifact = tmp / "artifact"  # no runtime_controls.json -> uses global
        artifact.mkdir()

        with mock.patch(
            "scripts.evaluate_real_prompts.build_runner_from_state_dir",
            side_effect=fake_build,
        ):
            with mock.patch(
                "scripts.evaluate_real_prompts.BASELINE_SNAPSHOT_DIR", snapshot
            ):
                res = evaluate_isolated_prompts(prompts, artifact_dir=artifact)

    # Baseline (snapshot, marker present) picks snapshot_tool == expected -> match.
    # Pretrained (artifact, no controls -> global, no marker) picks global_tool -> miss.
    assert res["rows"][0]["baseline_match"] is True
    assert res["rows"][0]["pretrained_match"] is False
    assert res["rows"][0]["delta"] == -1
    assert res["delta_tool_match_rate"] == -1.0


def test_end_to_end_no_false_signal_when_snapshot_equals_global():
    global_controls = json.loads(
        Path("data/control/runtime_controls.json").read_text(encoding="utf-8")
    )

    def fake_build(source_dir, workspace, override_controls=None):
        _real_runner_factory_that_reads_workspace._current_source = source_dir
        _real_runner_factory_that_reads_workspace._current_override = override_controls
        return _real_runner_factory_that_reads_workspace(workspace, "tool_A", "tool_A")

    prompts = [{"id": "p", "prompt": "q", "expected_tool": "global_tool"}]
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        snapshot = tmp / "snapshot"
        snapshot.mkdir()
        # Snapshot is an EXACT copy of the live global controls -> should behave
        # identically to the pretrained (global) runner.
        _write_controls(snapshot / "runtime_controls.json", global_controls)
        (snapshot / "policy_state.json").write_text("{}", encoding="utf-8")
        artifact = tmp / "artifact"
        artifact.mkdir()

        with mock.patch(
            "scripts.evaluate_real_prompts.build_runner_from_state_dir",
            side_effect=fake_build,
        ):
            with mock.patch(
                "scripts.evaluate_real_prompts.BASELINE_SNAPSHOT_DIR", snapshot
            ):
                res = evaluate_isolated_prompts(prompts, artifact_dir=artifact)

    # Both baseline (snapshot==global) and pretrained (global) pick global_tool == expected.
    # DeltaP stays 0 -> no false signal (the honest guarantee from the proposal).
    assert res["rows"][0]["baseline_match"] is True
    assert res["rows"][0]["pretrained_match"] is True
    assert res["rows"][0]["delta"] == 0
    assert res["delta_tool_match_rate"] == 0.0


# ---------------------------------------------------------------------------
# Phase two: override_controls injects a *candidate* HarnessPolicy as P_after,
# so delta_tool_match_rate becomes non-zero (candidate vs pinned baseline).
# ---------------------------------------------------------------------------


def test_evaluate_isolated_prompts_override_controls_make_pretrained_distinct():
    def fake_build(source_dir, workspace, override_controls=None):
        real_runner = build_runner_from_state_dir(source_dir, workspace, override_controls)
        # Candidate (override) is written into the workspace controls file; the
        # fake runner reflects it. This proves the override actually reached the
        # pretrained runner (not silently dropped by build_runner_from_state_dir).
        controls = json.loads(
            (workspace / "runtime_controls.json").read_text(encoding="utf-8")
        )
        chosen = (
            "candidate_tool"
            if controls.get("policy", {}).get("_candidate_marker")
            else "baseline_tool"
        )
        real_runner.run = lambda prompt: (
            "answer",
            SimpleNamespace(steps=[SimpleNamespace(selected_tool=chosen)], actions=[]),
        )
        return real_runner

    prompts = [{"id": "p1", "prompt": "q1", "expected_tool": "candidate_tool"}]
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        snapshot = tmp / "snapshot"
        snapshot.mkdir()
        _write_controls(
            snapshot / "runtime_controls.json",
            {"policy": {}, "glance": {}, "core": {}},
        )
        (snapshot / "policy_state.json").write_text("{}", encoding="utf-8")
        artifact = tmp / "artifact"  # no runtime_controls.json -> global fallback
        artifact.mkdir()

        with mock.patch(
            "scripts.evaluate_real_prompts.build_runner_from_state_dir",
            side_effect=fake_build,
        ):
            with mock.patch(
                "scripts.evaluate_real_prompts.BASELINE_SNAPSHOT_DIR", snapshot
            ):
                res = evaluate_isolated_prompts(
                    prompts,
                    artifact_dir=artifact,
                    override_controls={"policy": {"_candidate_marker": 1}},
                )

    # Baseline (snapshot, no candidate marker) -> baseline_tool != expected (miss).
    # Pretrained (candidate injected via override) -> candidate_tool == expected (hit).
    # DeltaP = +1: candidate improved over the pinned baseline.
    assert res["rows"][0]["baseline_match"] is False
    assert res["rows"][0]["pretrained_match"] is True
    assert res["rows"][0]["delta"] == 1
    assert res["delta_tool_match_rate"] == 1.0
