from __future__ import annotations

import json
import shutil
from pathlib import Path

from carm.training import load_training_config

# State files copied by build_runner_from_state_dir when a snapshot is loaded as
# source_dir. Keeping this list in sync with that function is what makes the
# pinned baseline a faithful P_before (controls + trained state). episodes.jsonl
# is the learned experience and MUST be pinned too, or the baseline evaluates with
# an empty ExperienceStore (see build_runner_from_state_dir's fallback note).
SNAPSHOT_STATE_FILES = (
    "policy_state.json",
    "concept_state.json",
    "core_state.json",
    "evolution_state.json",
    "episodes.jsonl",
)


def main() -> int:
    control_src = Path("data/control/runtime_controls.json")
    training = load_training_config("configs/training.yaml")
    artifact_dir = Path(
        str(
            training.get("training", {})
            .get("pretraining", {})
            .get("artifact_dir", "data/pretrain")
        )
    )
    snapshot_dir = Path("data/eval/baseline_snapshot")
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    if control_src.exists():
        shutil.copyfile(control_src, snapshot_dir / "runtime_controls.json")
        copied.append("runtime_controls.json")
    for name in SNAPSHOT_STATE_FILES:
        src = artifact_dir / name
        if src.exists():
            shutil.copyfile(src, snapshot_dir / name)
            copied.append(name)

    summary = {
        "snapshot_dir": str(snapshot_dir),
        "control_sections": (
            list(
                json.loads(
                    (snapshot_dir / "runtime_controls.json").read_text(encoding="utf-8")
                ).keys()
            )
            if (snapshot_dir / "runtime_controls.json").exists()
            else []
        ),
        "files": copied,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
