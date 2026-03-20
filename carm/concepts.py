from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from carm.actions import Action
from carm.schemas import StepRecord


class AdaptiveConceptModel:
    """Learns token-to-action and token-to-tool preferences from episodes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.token_action_weights: dict[str, dict[str, float]] = {}
        self.token_tool_weights: dict[str, dict[str, float]] = {}
        self._load()
        if not self.token_action_weights:
            self._bootstrap()

    def action_priors(self, user_input: str) -> dict[str, float]:
        priors = {action.value: 0.0 for action in Action}
        tokens = self.tokenize(user_input)
        for token in tokens:
            for action_name, weight in self.token_action_weights.get(token, {}).items():
                priors[action_name] = priors.get(action_name, 0.0) + weight
        return priors

    def preferred_tool(self, user_input: str) -> str | None:
        scores: dict[str, float] = {}
        for token in self.tokenize(user_input):
            for tool_name, weight in self.token_tool_weights.get(token, {}).items():
                scores[tool_name] = scores.get(tool_name, 0.0) + weight
        if not scores:
            return None
        return max(scores, key=scores.get)

    def learn(self, steps: list[StepRecord], learning_rate: float) -> None:
        for step in steps:
            if not step.high_value or not step.user_input:
                continue
            tokens = self.tokenize(step.user_input)
            if not tokens:
                continue

            token_counts = Counter(tokens)
            for token, count in token_counts.items():
                weight_delta = learning_rate * step.reward * min(count, 3)
                action_bucket = self.token_action_weights.setdefault(token, {})
                action_bucket[step.action] = action_bucket.get(step.action, 0.0) + weight_delta

                if step.selected_tool:
                    tool_bucket = self.token_tool_weights.setdefault(token, {})
                    tool_bucket[step.selected_tool] = tool_bucket.get(step.selected_tool, 0.0) + weight_delta

        self._save()

    def export_state(self) -> dict[str, object]:
        return {
            "token_action_weights": self.token_action_weights,
            "token_tool_weights": self.token_tool_weights,
        }

    def tokenize(self, text: str) -> list[str]:
        lowered = text.lower()
        ascii_tokens = re.findall(r"[a-z0-9_]{2,}", lowered)
        chinese_parts = re.findall(r"[\u4e00-\u9fff]{2,}", text)
        chinese_tokens: list[str] = []
        for part in chinese_parts:
            chinese_tokens.append(part)
            for size in (2, 3):
                for idx in range(0, max(0, len(part) - size + 1)):
                    chinese_tokens.append(part[idx : idx + size])
        return list(dict.fromkeys(ascii_tokens + chinese_tokens))

    def _bootstrap(self) -> None:
        seeds = {
            "比较": ("CALL_TOOL", "search", 0.45),
            "区别": ("CALL_TOOL", "search", 0.45),
            "对比": ("CALL_TOOL", "search", 0.45),
            "优缺点": ("CALL_TOOL", "search", 0.45),
            "计算": ("CALL_TOOL", "calculator", 0.55),
            "数字": ("CALL_TOOL", "calculator", 0.45),
            "代码": ("CALL_TOOL", "code_executor", 0.5),
            "python": ("CALL_TOOL", "code_executor", 0.5),
        }
        for token, (action_name, tool_name, weight) in seeds.items():
            self.token_action_weights[token] = {action_name: weight}
            self.token_tool_weights[token] = {tool_name: weight}
        self._save()

    def _load(self) -> None:
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.token_action_weights = payload.get("token_action_weights", {})
        self.token_tool_weights = payload.get("token_tool_weights", {})

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.export_state(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
