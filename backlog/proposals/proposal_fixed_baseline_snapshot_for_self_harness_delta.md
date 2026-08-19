# 提案：固定基线快照，让 Self-Harness ΔP 真正有判别力

- 提案编号：v26（接续 v25 Self-Harness 协议替代 0.9 门槛）
- 日期：2026-08-19
- 关联文档：
  - `docs/Mustard_修改与优化建议_2026-08-18.md` §8（Self-Harness 协议：P_before / ΔP / Transfer / Regression / Cost）
  - `backlog/proposals/proposal_self_harness_eval_protocol_replacing_0.9_gate.md`（v25，已落地 0.9 门槛起草）
  - 提交 `25c47db`（v25 闭环：real_prompt_eval 已产出 `delta_tool_match_rate` 字段）
- 变更类型：**Human Gate**（改 `scripts/evaluate_pretraining.py` 的 `build_runner_from_state_dir` controls 加载语义 + 改 `scripts/evaluate_real_prompts.py` 的 baseline 来源，均影响评测基线与 Arbiter 的 ΔP 绑定判断，属 `runtime_control` 治理层变更）
- 状态：待用户审批（仅起草，未落地代码）

## 1. 问题（已实测确认）

v25 已让 `delta_tool_match_rate`（P_after − P_before）字段真正产出并被 `team_conductor` 的 `self_harness_eval.require_non_negative_real_prompt_delta` 消费。但实测回填发现：当前 `baseline_match_rate == pretrained_match_rate == 0.7302`，**聚合 ΔP 结构性退化为 0**，逐行 churn 是 `+1/−1` 相互抵消（子集内 1 条改善、子集外 1 条回归）。

**根因（代码级事实，非推测）**：
- `scripts/evaluate_real_prompts.py:22`：`baseline_runner = build_runner_from_state_dir(None, root/"baseline")`。
- `scripts/evaluate_pretraining.py:96-99`：`build_runner_from_state_dir` **无条件用全局 `data/control/runtime_controls.json` 覆盖** workspace 的 controls（不论 baseline 还是 pretrained）。
- 因此 baseline 与 pretrained 的 HarnessPolicy（controls：policy/glance/core）完全相同，且当前训练 state 在 `allow_episode_learning=False` 下不影响工具选择 → 二者给出相同 `0.7302` → ΔP 恒为 0。
- 结论：ΔP 绑定虽已激活（delta<0 会硬卡），但**当前无法区分真实改善**——baseline 是「无 state 的默认 runner」，不是「上次已知良好的策略快照」。这与 Self-Harness 协议的 `P_before` 语义不符。

## 2. 核心主张

让 `P_before` = **固化的上次已知良好策略快照**（controls + 训练 state），而非「无 state 的默认 runner」。这样 ΔP = 当前候选 − 上次快照，才真正反映「这次改动带来了多少成长」，并与 v25 的 ΔP 硬闸闭环。

## 3. 具体落地建议（分两阶段；阶段一为本次执行范围）

### 阶段一：固定基线快照（本次提案范围）

1. **新增基线快照目录** `data/eval/baseline_snapshot/`（不进 git，首次需显式 pin；加入 `.gitignore`）。
2. **新增 `scripts/pin_baseline.py`**：将当前「已知良好」配置固化到快照——
   - `data/control/runtime_controls.json` → `data/eval/baseline_snapshot/runtime_controls.json`
   - `data/experience/{policy_state,concept_state,core_state,evolution_state}.json`（存在的）→ 快照
   - 打印快照内容摘要（sections / 文件列表 / hash）供 Guardian 审查。
3. **增强 `build_runner_from_state_dir`（必要修复）**：新增 `use_snapshot_controls` 语义——
   - 若 `source_dir` 含 `runtime_controls.json` 且未显式传 `override_controls`，则**优先用快照的 controls**，不再用全局文件覆盖。
   - 向后兼容：`override_controls` 显式传入时优先 override；`source_dir` 无 controls 时回退全局（今日行为）。
   - 说明：此修复是让 baseline 真正区别于 pretrained 的前提——否则即便 baseline 有快照 controls，也会被 L96-99 全局覆盖，ΔP 仍恒 0。
4. **改 `evaluate_isolated_prompts` 的 baseline 来源**：`build_runner_from_state_dir(None, ...)` → `build_runner_from_state_dir(BASELINE_SNAPSHOT_DIR if BASELINE_SNAPSHOT_DIR.exists() else None, ...)`。快照目录含 `runtime_controls.json` → 触发 `use_snapshot_controls`。
5. **向后兼容**：快照目录不存在时回退 `None` → 全局覆盖 → 与今日行为完全一致（ΔP=0），不破坏现有评测与回填。

### 阶段二（后续，可能也属 Human Gate）：real_prompt_eval 支持候选注入

- `evaluate_isolated_prompts` 增加 `override_controls` 参数，使 P_after 能测**候选 HarnessPolicy**（而非仅默认 controls 的 artifact）。
- 当前 `CANDIDATE_POLICY` 只在 `evaluate_combined_tool_policy_candidate` 经 `override_controls` 生效，`real_prompt_eval` 永不注入候选 → ΔP 始终只反映 artifact 训练变化。
- 阶段二是让 ΔP 直接反映候选 policy 改善的主要杠杆，超出「固定 baseline」范围，单列提案。

## 4. 影响与风险

- **不承诺分数提升**：固定 baseline 治的是「信号缺失」，不是「刷分」。首次 pin 基线 = 当前 `0.7302` 快照，ΔP 在训练无进展时仍 = 0（如实，无误信号）。
- **真正让 ΔP 非零的杠杆**：阶段二（候选注入 real_prompt_eval）+ 训练/候选真正带来工具选择差异。固定 baseline 是必要不充分条件。
- **0.7302 上限不变**：见 `memory/MEMORY.md`——`0.7302`（46/63）是外部工具不可达沙箱外的可达上限；固定同一上限与事实不矛盾。
- **误 pin 风险**：若误 pin 坏基线会屏蔽回归 → `pin_baseline` 动作须纳入 Guardian 审查（确认快照来自一次 deep_cycle 通过态）。
- **调用方安全**：`auto_train.py`、`evaluate_conflict_verify_candidate.py`、`evaluate_comparison_search_candidate.py`、`evaluate_tool_boundary_candidate.py`、`evaluate_combined_tool_policy_candidate.py` 均传 `artifact_dir`（= `data/experience`，不含 `runtime_controls.json`）作 source_dir → 不触发 `use_snapshot_controls`；候选脚本显式传 `override_controls` 优先于快照 → 行为全部不变。
- **Human Gate**：改 `build_runner_from_state_dir` 的 controls 加载语义影响评测基线，进而改变 Arbiter 的 ΔP 绑定判断，属 `runtime_control` 治理层变更，须用户批准。

## 5. 验证计划

- **单元驱动**：构造临时 baseline 快照（故意让 1 条 prompt 的 controls 不同），验证 `evaluate_isolated_prompts` 产出 `delta=±1/0`，且 baseline 加载用的是快照 controls（不被全局覆盖）。
- **端到端（外部可达 CI）**：`pin_baseline` 固化当前 `0.7302` 基线后重测，确认 ΔP 仍 = 0（无假信号）+ `0.7302` 稳定。
- **team_conductor 三态回归**：固定 baseline 后 ΔP 绑定路径不变——`delta=0.0 → direction_adjust`（解卡）、`delta<0 → uncertain_needs_human`（ΔP 硬卡）、legacy `enforce=true → uncertain_needs_human`。
- **回归**：`pytest tests/test_evaluate_pretraining.py tests/test_team_conductor.py`（已知 `test_build_proposals_prioritizes_new_failure_patterns` 为既有/环境性失败，与本次无关）。

## 6. 待办 / 解锁顺序

1. 用户审批本提案（Human Gate）。
2. 落地阶段一 + 单测 + 提交（不推送，GitHub 断开）。
3. 外部可达 CI 重测固化 `0.7302` 基线，确认 ΔP 信号路径无假阳。
4. 视训练/候选进展，再起草阶段二（候选注入 real_prompt_eval）。
