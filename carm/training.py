from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from carm.attention_flow import AttentionTrainingView, nodes_from_payloads
from carm.actions import Action
from carm.core import AdaptiveReasoningCore
from carm.evolution import EvolutionSignal, OnlineEvolutionManager
from carm.experience import ExperienceStore
from carm.policy import OnlinePolicy
from carm.pretrain_data import load_pretrain_samples, sample_to_episode
from carm.review import ReviewStore
from carm.schemas import StepRecord


DEFAULT_TRAINING_CONFIG: dict[str, object] = {
    "training": {
        "mode": "two_stage",
        "pretraining": {
            "enabled": True,
            "artifact_dir": "data/pretrain",
            "replay_success_only": True,
            "max_episodes": 500,
            "dataset_path": "data/pretrain/pretrain_corpus.jsonl",
            "max_synthetic_samples": 2000,
            "reset_artifacts": True,
        },
        "online_evolution": {
            "enabled": True,
            "allow_episode_learning": True,
            "allow_user_signals": True,
            "signal_state_path": "data/evolution/state.json",
            "signal_log_path": "data/evolution/signals.jsonl",
        },
    }
}


def load_training_config(path: str | Path | None) -> dict[str, object]:
    config = json.loads(json.dumps(DEFAULT_TRAINING_CONFIG))
    if path is None:
        return config

    candidate = Path(path)
    if not candidate.exists():
        return config

    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return config

    training = payload.get("training", {})
    if isinstance(training, dict):
        config["training"].update(training)
        for key in ("pretraining", "online_evolution"):
            if isinstance(training.get(key), dict):
                section = dict(config["training"].get(key, {}))
                section.update(training[key])
                config["training"][key] = section
    return config


@dataclass
class PretrainResult:
    artifact_dir: Path
    episode_count: int
    synthetic_sample_count: int
    replayed_step_count: int
    signal_count: int
    attention_view_count: int = 0


class OfflinePretrainer:
    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)

    def run(
        self,
        experience_path: str | Path,
        review_path: str | Path,
        signal_log_path: str | Path | None = None,
        dataset_path: str | Path | None = None,
        attention_flow_path: str | Path | None = None,
        attention_views_path: str | Path | None = None,
        *,
        max_episodes: int = 500,
        max_synthetic_samples: int = 2000,
        replay_success_only: bool = True,
        reset_artifacts: bool = True,
    ) -> PretrainResult:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        if reset_artifacts:
            self._reset_artifacts()

        policy = OnlinePolicy(
            self.artifact_dir / "policy_state.json",
            self.artifact_dir / "concept_state.json",
        )
        core = AdaptiveReasoningCore(self.artifact_dir / "core_state.json")
        evolution = OnlineEvolutionManager(
            self.artifact_dir / "evolution_state.json",
            self.artifact_dir / "signals.jsonl",
        )

        episodes = ExperienceStore(experience_path).load_all()
        if replay_success_only:
            episodes = [episode for episode in episodes if episode.success]
        episodes = episodes[:max_episodes]

        replayed_step_count = 0
        for episode in episodes:
            policy.learn(episode.steps)
            core.learn(episode.user_input, episode.steps, episode.success)
            replayed_step_count += len(episode.steps)

        synthetic_sample_count = 0
        dataset_candidate = Path(dataset_path) if dataset_path is not None else None
        if dataset_candidate is not None and dataset_candidate.exists():
            for sample in load_pretrain_samples(dataset_candidate)[:max_synthetic_samples]:
                episode = sample_to_episode(sample)
                policy.learn(episode.steps)
                core.learn(episode.user_input, episode.steps, episode.success)
                replayed_step_count += len(episode.steps)
                synthetic_sample_count += 1

        signal_count = 0
        signal_path = Path(signal_log_path) if signal_log_path is not None else None
        if signal_path is not None and signal_path.exists():
            for line in signal_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                signal = EvolutionSignal(**json.loads(line))
                synthetic_steps = evolution.apply_signal(signal)
                if synthetic_steps:
                    policy.learn(synthetic_steps)
                    core.learn(signal.query or signal.goal or signal.note, synthetic_steps, success=signal.reward >= 0.0)
                signal_count += 1

        attention_view_count = self._replay_attention_views(
            policy, core, attention_flow_path, attention_views_path
        )

        reviews = ReviewStore(review_path).load_all()
        manifest = {
            "mode": "two_stage",
            "episode_count": len(episodes),
            "synthetic_sample_count": synthetic_sample_count,
            "replayed_step_count": replayed_step_count,
            "review_count": len(reviews),
            "signal_count": signal_count,
            "attention_view_count": attention_view_count,
            "artifacts": {
                "policy_state": str(self.artifact_dir / "policy_state.json"),
                "concept_state": str(self.artifact_dir / "concept_state.json"),
                "core_state": str(self.artifact_dir / "core_state.json"),
                "evolution_state": str(self.artifact_dir / "evolution_state.json"),
            },
        }
        (self.artifact_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return PretrainResult(
            artifact_dir=self.artifact_dir,
            episode_count=len(episodes),
            synthetic_sample_count=synthetic_sample_count,
            replayed_step_count=replayed_step_count,
            signal_count=signal_count,
            attention_view_count=attention_view_count,
        )

    def _replay_attention_views(
        self,
        policy: OnlinePolicy,
        core: AdaptiveReasoningCore,
        attention_flow_path: Path | str | None,
        attention_views_path: Path | str | None,
    ) -> int:
        views: list[AttentionTrainingView] = []

        views_candidate = Path(attention_views_path) if attention_views_path is not None else None
        if views_candidate is not None and views_candidate.exists():
            payload = json.loads(views_candidate.read_text(encoding="utf-8"))
            items = payload.get("views", []) if isinstance(payload, dict) else []
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        views.append(AttentionTrainingView(**item))

        flow_candidate = Path(attention_flow_path) if attention_flow_path is not None else None
        if flow_candidate is not None and flow_candidate.exists() and not views:
            payload = json.loads(flow_candidate.read_text(encoding="utf-8"))
            nodes_data = payload.get("nodes", []) if isinstance(payload, dict) else []
            if isinstance(nodes_data, list) and nodes_data:
                from carm.attention_flow import build_training_views
                nodes = nodes_from_payloads(nodes_data)
                views = build_training_views(nodes)

        for view in views:
            synthetic_step = self._view_to_step(view)
            if synthetic_step is not None:
                policy.learn([synthetic_step])
                core.learn(
                    view.episode_id,
                    [synthetic_step],
                    success=view.release_allowed,
                )

        return len(views)

    def _view_to_step(self, view: AttentionTrainingView) -> StepRecord | None:
        reward = 1.0 if view.release_allowed else -0.8
        if view.recommended_action == "ANSWER" and not view.release_allowed:
            reward = -1.0
        if view.recommended_action == "VERIFY" and view.release_allowed:
            reward = 0.5

        feature_snapshot: dict[str, float] = {
            "bias": 1.0,
            "uncertainty": 0.4 if "conflict_unresolved" in view.residual_pressure else 0.2,
            "has_conflict": 1.0 if "conflict_unresolved" in view.residual_pressure else 0.0,
            "has_result": 0.0,
            "has_draft": 1.0,
        }

        target_slot = ""
        if view.recommended_action == "WRITE_MEM":
            target_slot = "DRAFT" if view.release_allowed else "HYP"
        if view.recommended_action == "VERIFY":
            target_slot = ""

        return StepRecord(
            step_idx=view.step_idx,
            action=view.recommended_action,
            reason=view.supervision_note,
            score=abs(reward),
            feature_snapshot=feature_snapshot,
            user_input=view.episode_id,
            target_slot=target_slot,
            reward=reward,
            reward_reason=f"attention_view:{view.residual_pressure}",
            high_value=abs(reward) >= 0.5,
        )

    def _reset_artifacts(self) -> None:
        for name in (
            "policy_state.json",
            "concept_state.json",
            "core_state.json",
            "evolution_state.json",
            "signals.jsonl",
            "manifest.json",
        ):
            path = self.artifact_dir / name
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
