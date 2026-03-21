from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from carm.experience import ExperienceStore
from carm.pretrain_data import infer_logic_skill, infer_task_type


def normalize_prompt(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def build_candidates(
    experience_path: str | Path,
    *,
    min_value_score: float = 0.65,
    limit: int = 50,
) -> list[dict[str, object]]:
    episodes = ExperienceStore(experience_path).load_all()
    best_by_prompt: dict[str, dict[str, object]] = {}

    for episode in episodes:
        prompt = episode.user_input.strip()
        if not prompt or not episode.success or episode.value_score < min_value_score:
            continue

        used_tool = str(episode.episode_features.get("used_tool", "")).strip()
        if not used_tool:
            continue

        task_type = infer_task_type(prompt)
        logic_skill = str(episode.episode_features.get("logic_skill", "")).strip() or infer_logic_skill(prompt, task_type)
        key = normalize_prompt(prompt)
        candidate = {
            "id": f"candidate-{abs(hash(key)) % 1000000:06d}",
            "prompt": prompt,
            "expected_tool": used_tool,
            "logic_skill": logic_skill,
            "source": "experience",
            "value_score": round(float(episode.value_score), 4),
            "action_sequence": list(episode.episode_features.get("action_sequence", [])),
        }

        existing = best_by_prompt.get(key)
        if existing is None or float(candidate["value_score"]) > float(existing["value_score"]):
            best_by_prompt[key] = candidate

    candidates = sorted(best_by_prompt.values(), key=lambda item: (float(item["value_score"]), len(str(item["prompt"]))), reverse=True)
    return candidates[:limit]


def save_candidates(path: str | Path, candidates: list[dict[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"prompts": candidates}
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    experience_path = Path(args[0]) if len(args) >= 1 else Path("data/experience/episodes.jsonl")
    output_path = Path(args[1]) if len(args) >= 2 else Path("data/eval/real_prompt_candidates.json")

    candidates = build_candidates(experience_path)
    save_candidates(output_path, candidates)
    print(f"Built real prompt candidates: count={len(candidates)} path={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
