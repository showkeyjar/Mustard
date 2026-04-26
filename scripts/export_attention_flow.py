from __future__ import annotations

import json
import sys
from pathlib import Path

from carm.attention_flow import AttentionNode, nodes_to_jsonl, project_episode_attention, project_eval_row_attention
from carm.experience import ExperienceStore


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _prompt_lookup(path: Path) -> dict[str, str]:
    payload = _read_json(path)
    prompts = payload.get("prompts", [])
    if not isinstance(prompts, list):
        return {}
    lookup: dict[str, str] = {}
    for item in prompts:
        if isinstance(item, dict):
            lookup[str(item.get("id", ""))] = str(item.get("prompt", ""))
    return lookup


def export_attention_flow(
    experience_path: Path = Path("data/experience/episodes.jsonl"),
    output_path: Path = Path("data/attention/attention_flow.jsonl"),
    eval_path: Path = Path("data/eval/real_prompt_eval_latest.json"),
    prompt_path: Path = Path("configs/real_prompt_eval.json"),
) -> list[AttentionNode]:
    episodes = ExperienceStore(experience_path).load_all()
    nodes: list[AttentionNode] = []
    for index, episode in enumerate(episodes, start=1):
        nodes.extend(project_episode_attention(episode, episode_id=f"episode-{index:06d}"))

    eval_payload = _read_json(eval_path)
    rows = eval_payload.get("rows", [])
    prompts = _prompt_lookup(prompt_path)
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                case_id = str(row.get("id", ""))
                nodes.extend(project_eval_row_attention(row, prompts.get(case_id, "")))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = nodes_to_jsonl(nodes)
    output_path.write_text((payload + "\n") if payload else "", encoding="utf-8")
    return nodes


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    experience_path = Path(args[0]) if len(args) >= 1 else Path("data/experience/episodes.jsonl")
    output_path = Path(args[1]) if len(args) >= 2 else Path("data/attention/attention_flow.jsonl")
    eval_path = Path(args[2]) if len(args) >= 3 else Path("data/eval/real_prompt_eval_latest.json")
    prompt_path = Path(args[3]) if len(args) >= 4 else Path("configs/real_prompt_eval.json")
    nodes = export_attention_flow(experience_path, output_path, eval_path, prompt_path)
    print(json.dumps({"node_count": len(nodes), "output_path": str(output_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
