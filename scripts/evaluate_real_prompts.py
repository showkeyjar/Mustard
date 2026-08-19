from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

from carm.training import load_training_config
from scripts.evaluate_pretraining import build_runner_from_state_dir, load_eval_prompts

# Fixed baseline snapshot for Self-Harness P_before (populated by scripts/pin_baseline.py).
# When present, real_prompt_eval's baseline runner loads the pinned policy + state
# instead of a stateless default runner, so delta_tool_match_rate (P_after - P_before)
# becomes discriminative. Missing -> fall back to None (legacy behavior, DeltaP=0).
BASELINE_SNAPSHOT_DIR = Path("data/eval/baseline_snapshot")

# Wall-clock guard per runner.run. Without this, a single prompt whose agent loops
# or blocks on an unreachable tool hangs the ENTIRE evaluation forever (observed:
# 23/63 real prompts hung indefinitely in a sandbox with partial tool reachability).
# A timeout marks that prompt as a non-match and lets the run continue.
RUNNER_TIMEOUT = float(os.environ.get("REAL_PROMPT_RUNNER_TIMEOUT", "120"))


class _EmptyTrace:
    """Placeholder trace for a prompt whose runner.run timed out."""

    steps: list = []
    actions: list = []


def _run_with_timeout(runner, prompt: str, timeout: float = RUNNER_TIMEOUT):
    """Run runner.run(prompt) in a daemon thread, returning (trace, timed_out).

    On timeout the runner thread is left to finish on its own (daemon) but we
    return an _EmptyTrace so the prompt is scored as a non-match instead of
    blocking the whole evaluation.
    """
    box: dict = {}

    def _target() -> None:
        try:
            result = runner.run(prompt)
            # runner.run returns (something, trace); the trace is the 2nd element.
            box["trace"] = result[1] if isinstance(result, tuple) and len(result) >= 2 else result
        except Exception as exc:  # noqa: BLE001 - surface as a non-match, not a crash
            box["error"] = exc

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return _EmptyTrace(), True
    if "error" in box:
        # A raised error in the runner means the prompt could not be evaluated;
        # treat identically to a timeout (non-match) so one bad prompt can't kill
        # the whole run.
        return _EmptyTrace(), True
    return box.get("trace", _EmptyTrace()), False



def evaluate_isolated_prompts(
    prompts: list[dict[str, str]],
    *,
    artifact_dir: Path,
    override_controls: dict | None = None,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []

    for item in prompts:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline_source = (
                BASELINE_SNAPSHOT_DIR if BASELINE_SNAPSHOT_DIR.exists() else None
            )
            baseline_runner = build_runner_from_state_dir(baseline_source, root / "baseline")
            # Phase two (v26): P_after tests the *candidate* HarnessPolicy when a
            # candidate is supplied, while P_before stays the pinned baseline snapshot.
            # override_controls wins over the snapshot (and the global file) inside
            # build_runner_from_state_dir, so delta_tool_match_rate then measures the
            # candidate against the pinned baseline instead of artifact-vs-snapshot.
            # When omitted (default main() run), behavior is unchanged: pretrained
            # uses artifact_dir -> global controls fallback -> DeltaP stays 0.
            pretrained_runner = build_runner_from_state_dir(
                artifact_dir, root / "pretrained", override_controls=override_controls
            )

            baseline_trace, baseline_timed_out = _run_with_timeout(
                baseline_runner, str(item.get("prompt", "")), timeout=RUNNER_TIMEOUT
            )
            pretrained_trace, pretrained_timed_out = _run_with_timeout(
                pretrained_runner, str(item.get("prompt", "")), timeout=RUNNER_TIMEOUT
            )

        expected_tool = str(item.get("expected_tool", ""))
        baseline_used_tool = next(
            (step.selected_tool for step in baseline_trace.steps if step.selected_tool),
            "",
        )
        pretrained_used_tool = next(
            (
                step.selected_tool
                for step in pretrained_trace.steps
                if step.selected_tool
            ),
            "",
        )

        baseline_match = baseline_used_tool == expected_tool if expected_tool else False
        pretrained_match = pretrained_used_tool == expected_tool if expected_tool else False

        rows.append(
            {
                "id": str(item.get("id", "")),
                "logic_skill": str(item.get("logic_skill", "")),
                "expected_tool": expected_tool,
                "baseline_used_tool": baseline_used_tool,
                "pretrained_used_tool": pretrained_used_tool,
                "baseline_actions": list(baseline_trace.actions),
                "pretrained_actions": list(pretrained_trace.actions),
                "baseline_match": baseline_match,
                "pretrained_match": pretrained_match,
                "baseline_timed_out": baseline_timed_out,
                "pretrained_timed_out": pretrained_timed_out,
                # Self-Harness pillar (P_before -> DeltaP): per-prompt movement.
                # +1 = candidate improved over baseline, -1 = regressed, 0 = unchanged.
                "delta": int(pretrained_match) - int(baseline_match),
            }
        )

    total = max(1, len(rows))
    baseline_matches = sum(1 for row in rows if row["baseline_match"])
    pretrained_matches = sum(1 for row in rows if row["pretrained_match"])
    return {
        # Self-Harness pillar DeltaP (P_after - P_before) at the aggregate level.
        # Consumed by team_conductor.py (self_harness_eval.require_non_negative_real_prompt_delta
        # and deep_cycle_policy.require_positive_real_prompt_delta) and build_daily_digest.
        "delta_tool_match_rate": round((pretrained_matches - baseline_matches) / total, 4),
        "summary": {
            "prompt_count": len(rows),
            "baseline_match_rate": round(baseline_matches / total, 4),
            "pretrained_match_rate": round(pretrained_matches / total, 4),
            "baseline_avg_steps": round(
                sum(len(row["baseline_actions"]) for row in rows) / total, 4
            ),
            "pretrained_avg_steps": round(
                sum(len(row["pretrained_actions"]) for row in rows) / total, 4
            ),
        },
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    prompt_path = Path(args[0]) if args else Path("configs/real_prompt_eval.json")
    prompts = load_eval_prompts(prompt_path)
    training = load_training_config("configs/training.yaml")
    artifact_dir = Path(
        str(
            training.get("training", {})
            .get("pretraining", {})
            .get("artifact_dir", "data/pretrain")
        )
    )

    result = evaluate_isolated_prompts(prompts, artifact_dir=artifact_dir)

    # Persist to data/eval/real_prompt_eval_latest.json so downstream
    # consumers (current_best.py, team_conductor.py, analyze_reasoning_patterns.py)
    # always read fresh results instead of stale snapshots.
    #
    # IMPORTANT: only the default (no-arg) run -- the official config
    # configs/real_prompt_eval.json -- may overwrite latest. Explicit config
    # paths are used for ad-hoc probes (recovery variants, stress evals,
    # quality-focus evals) whose small prompt counts would otherwise pollute
    # the official snapshot; those callers persist their own result files.
    if not args:
        latest_path = Path("data/eval/real_prompt_eval_latest.json")
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        latest_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
