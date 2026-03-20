from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from carm.review import ReviewStore


def build_payload(reviews: list) -> dict[str, object]:
    tag_counter: Counter[str] = Counter()
    module_counter: Counter[str] = Counter()
    recommendation_counter: Counter[str] = Counter()
    success_by_tag: defaultdict[str, list[bool]] = defaultdict(list)
    trigger_counter: Counter[str] = Counter()
    glance_help_rates: list[float] = []

    for review in reviews:
        for tag in review.issue_tags:
            tag_counter[tag] += 1
            success_by_tag[tag].append(review.success)
        for module in review.target_modules:
            module_counter[module] += 1
        for recommendation in review.recommendations:
            recommendation_counter[recommendation] += 1

        evidence = review.evidence or {}
        for trigger in evidence.get("glance_triggers", []):
            if trigger:
                trigger_counter[trigger] += 1
        if "glance_help_rate" in evidence:
            glance_help_rates.append(float(evidence["glance_help_rate"]))

    avg_glance_help = round(sum(glance_help_rates) / len(glance_help_rates), 4) if glance_help_rates else 0.0

    payload = {
        "review_count": len(reviews),
        "top_issue_tags": tag_counter.most_common(10),
        "top_target_modules": module_counter.most_common(10),
        "top_recommendations": recommendation_counter.most_common(10),
        "tag_success_rates": {
            tag: round(sum(1 for value in outcomes if value) / len(outcomes), 4)
            for tag, outcomes in success_by_tag.items()
            if outcomes
        },
        "glance_summary": {
            "trigger_counts": trigger_counter.most_common(10),
            "average_help_rate": avg_glance_help,
        },
        "slow_path_actions": build_slow_path_actions(
            tag_counter,
            module_counter,
            recommendation_counter,
            avg_glance_help,
            trigger_counter,
        ),
    }
    return payload


def build_slow_path_actions(
    tag_counter: Counter[str],
    module_counter: Counter[str],
    recommendation_counter: Counter[str],
    avg_glance_help: float,
    trigger_counter: Counter[str],
) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []

    if tag_counter.get("idle_drift", 0) >= 2:
        actions.append(
            {
                "type": "tighten_constraint",
                "target_module": "policy",
                "reason": "Repeated idle drift suggests THINK remains too cheap.",
                "proposal": "Increase THINK penalty or raise VERIFY/CALL_TOOL priority under uncertainty.",
            }
        )

    if tag_counter.get("tool_underuse", 0) >= 2:
        actions.append(
            {
                "type": "raise_tool_bias",
                "target_module": "policy",
                "reason": "Tool-eligible tasks are avoiding tool calls.",
                "proposal": "Raise CALL_TOOL prior for comparison, numeric, and code-oriented tasks.",
            }
        )

    if tag_counter.get("weak_grounding", 0) >= 2:
        actions.append(
            {
                "type": "promote_draft_path",
                "target_module": "core",
                "reason": "External results are not converting into high-confidence drafts consistently.",
                "proposal": "Strengthen RESULT->DRAFT transition and support-item generation.",
            }
        )

    if trigger_counter.get("high_uncertainty", 0) >= 3 and avg_glance_help < 0.5:
        actions.append(
            {
                "type": "reduce_glance_trigger",
                "target_module": "glance",
                "reason": "High-uncertainty glance is common but not helping enough.",
                "proposal": "Require additional conditions before activating high_uncertainty glance.",
            }
        )

    if avg_glance_help >= 0.7 and trigger_counter:
        actions.append(
            {
                "type": "retain_glance_policy",
                "target_module": "glance",
                "reason": "Bounded internal glance is helping more often than not.",
                "proposal": "Keep current budget and triggers; continue collecting evidence.",
            }
        )

    if not actions and recommendation_counter:
        most_common_recommendation, _ = recommendation_counter.most_common(1)[0]
        target_module = module_counter.most_common(1)[0][0] if module_counter else "policy"
        actions.append(
            {
                "type": "observe",
                "target_module": target_module,
                "reason": "Current reviews are stable; no structural change justified yet.",
                "proposal": most_common_recommendation,
            }
        )

    return actions


def main() -> int:
    review_path = Path(os.environ.get("CARM_REVIEW_PATH", "data/review/reviews.jsonl"))
    output_path = Path(os.environ.get("CARM_REVIEW_OUTPUT", "data/review/consolidated_recommendations.json"))
    store = ReviewStore(review_path)
    reviews = store.load_all()

    payload = build_payload(reviews)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Consolidated {len(reviews)} review(s) into {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
