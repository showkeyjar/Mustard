# Proposal: VQR-inspired reasoning pattern codec

- problem: 当前 CARM 的中间表示主要是固定槽位 `PLAN/HYP/DRAFT` 与 6 维 latent，能支持基本流程，但难以证明它对高信息任务形成了更强的信息压缩表示；现有自动研发流程也无法保证模型能力进化。
- evidence:
  - `docs/plans/2026-04-12-codex-human-loop-reset.md` 已建议从 `Claw Team` 主导切回 `Human + Codex` 主导。
  - `memory/failure_patterns.md` 当前仍记录 `sampling_blind_spot`、`repeated_conflict_detection_gap`、`comparison_under_conflicting_sources`、`tool_boundary_sampling_gap`。
  - `carm/core.py` 当前核心表示为少量手写 feature、slot weights 与 `LATENT_DIM = 6`，缺少“典型模式 + 残差信息”的显式结构。
  - `D:\codes\vqr` 的压缩链路已验证“主模式码本 + 残差编码 + 质量门控 + 背景场扣除”可把高维时空信息压成可重建表示。
- from_failure_pattern: sampling_blind_spot
- from_top_gap: 前沿对标不足 / 用户纠偏闭环缺失
- change_type: offline_experiment
- proposed_change: 新增一个离线 `reasoning pattern codec` 实验，先从现有 eval rows、pretrain samples 与成功 episode 中学习/归纳 reasoning patterns，再为每条任务输出 `pattern_id`、`residual_features`、`fit_score`、`reconstruction_notes`，用于解释冲突、多源整合、工具边界与终止判断任务的失败或险过原因。
- expected_metric_delta:
  - `residual_explanation_rate`: 从无指标提升到可统计
  - `hard_logic_pass_rate`: 建立基线后再要求正向提升
  - `critical_failure_count`: 后续实验应下降
- risk_level: low
- evaluation_plan:
  - 先不接默认运行时，只生成离线报告。
  - 用 `python -m unittest discover -s tests -v` 保证现有行为不退化。
  - 对 hard logic eval pack 统计 pattern 命中、residual 覆盖与错误解释率。
- rollback_plan: 删除新增离线实验文件与报告产物；不涉及默认权重、运行时策略或历史数据迁移。
- needs_human_approval: false for offline prototype; true before default runtime integration
- relative_to_last_round: 相比上一轮只扩 real prompts，本提案把盲区从“样本不足”推进到“中间表示无法解释高信息残差”的可验证问题。
- scenario_fit: 用户希望人工指挥 Codex 推动核心模型能力，并参考 VQR 的有效压缩思路改善当前模型结构。

