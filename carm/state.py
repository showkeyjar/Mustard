from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentState:
    hidden: dict[str, str] = field(default_factory=dict)
    latent: list[float] = field(default_factory=lambda: [0.0] * 6)
    phase: str = "INIT"
    step_idx: int = 0
    last_action: str = "NONE"
    uncertainty: float = 1.0
    answer_ready: float = 0.0
    glance_budget: int = 1

    def snapshot(self) -> "AgentState":
        return AgentState(
            hidden=dict(self.hidden),
            latent=list(self.latent),
            phase=self.phase,
            step_idx=self.step_idx,
            last_action=self.last_action,
            uncertainty=self.uncertainty,
            answer_ready=self.answer_ready,
            glance_budget=self.glance_budget,
        )
