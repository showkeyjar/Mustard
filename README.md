# CARM 最小原型

Compact Agentic Reasoning Model（CARM）的在线学习原型。

这个仓库提供了一套可运行的小型智能体脚手架，核心由“推理内核 + 工作记忆 + 工具总线”组成。当前版本已经升级为两段式方案：

- 预训练环节负责把历史 episode / review / evolution signal 回放成稳定初始能力
- 在线进化环节负责把用户显式目标、纠偏和偏好信号精确传给模型
- 运行时仍保留轻量在线更新，但现在受结构化用户信号约束

项目目标是 `CARM` 本身；`Claw Team` 仅作为持续自动优化该项目的工程化手段。

为避免主 README 混淆目标与手段，`Claw Team / GitHub 自动交付 / 人机协作` 的开发运维说明已拆分到：
- [docs/dev/claw_team.md](docs/dev/claw_team.md)

## 当前包含

- 结构化智能体状态与动作协议
- 固定槽位的工作记忆板
- 带持久化权重的在线策略控制环
- 从历史对话中学习动作与工具偏好的概念层
- 学习何时形成 `PLAN`、`HYP`、`DRAFT` 的自适应推理核
- 带 `action_items / unknowns / evidence_targets / confidence_band` 的结构化中间表示
- **真实可用的4种工具**：
  - **搜索**（SearchTool）：DuckDuckGo 真实检索 → Wikipedia 中文/英文回退 → 结构化 fallback
  - **计算器**（CalculatorTool）：递归下降解析器，支持 `+-*/**()`，中文自然语言算式提取（"N的M次方"、"每席位M元按年预算"、"平方根"等）
  - **代码执行**（CodeExecutorTool）：subprocess 沙箱执行，超时保护，6种常见算法模板（快速排序/冒泡排序/二分查找/斐波那契/归并排序/链表），print 输出正确捕获
  - **大模型代理**（BigModelProxyTool）：直连 Gemini API 或内置代理回退
- **语义意图编码器**（SemanticEncoder）：零依赖 Tier 1 同义词模式 + 可选 Tier 2 sentence-transformers 嵌入，覆盖搜索/计算/代码/大模型四类意图，带 LRU 缓存
- **统一信号模块**（signals.py）：冲突检测、搜索/计算/代码/正式/比较 6 类意图信号，跨模块复用无重复
- **语义优先工具路由**：policy.py 中 CALL_TOOL 分支基于语义编码器的 intent_scores 做工具选择，硬规则（算式→calculator、冲突→search、代码动作→code_executor）作为高优先级覆盖
- 自然语言风格的 Decoder 输出："关于「...」"开头，分段展示结论/依据/可信度/风险
- 用于动作奖励和对话摘要的经验回放存储
- 离线预训练脚本与预训练产物清单，支持增量训练（`reset_artifacts=False`）
- 增强评测维度：tool_match_rate + answer_completeness_rate + reasonable_confidence_rate + support_items_rate + conflict_risk_marked_rate
- 结构化在线进化信号管理器
- 可直接运行的本地入口
- 覆盖记忆流、代理主循环和桌面桥梁的测试

## 快速开始

### 0. 环境要求

- Python `3.10+`
- 已把仓库完整拉取到本地

```powershell
python --version
```

### 1. 安装

```powershell
# 基础安装（含搜索+计算器+代码执行+大模型代理）
pip install -e .

# 可选：语义增强（sentence-transformers 嵌入）
pip install -e ".[semantic]"

# 可选：Windows 桌面代理（OCR + 截屏）
pip install -e ".[desktop]"
```

### 2. 配置 LLM 后端（可选但推荐）

CARM 的 bigmodel_proxy 工具支持三种后端，优先级从高到低：

1. **Gemini API**（云上，质量最高）：设置环境变量 `GEMINI_API_KEY`
2. **Ollama 本地模型**（零配置，默认 `localhost:11434`）：启动 `ollama serve` 并拉取模型
3. **Distill 模式**（无 LLM 时的结构化回退）

```powershell
# 方式一：Gemini
set GEMINI_API_KEY=your-key-here

# 方式二：Ollama（需先安装 ollama 并拉取模型）
ollama pull qwen3-coder
ollama serve
# 可选：自定义 Ollama 地址和模型
set OLLAMA_BASE_URL=http://localhost:11434
set OLLAMA_MODEL=qwen3-coder
```

### 3. 运行

```powershell
python -m scripts.desktop_agent_control launch
```

如果你必须从批处理入口启动，也可以使用 [start_carm.cmd](d:/codes/Mustard/start_carm.cmd)，但它会短暂经过命令行窗口，不如 `start_carm.vbs` 干净。

### 1. 开发协作与自动优化（Claw Team）

`Claw Team` 相关的跨机器启动、自动交付、PR 评审/合并、人机协作与 GitHub 配置，已统一迁移到开发文档：

- [docs/dev/claw_team.md](docs/dev/claw_team.md)

以下 README 继续聚焦 CARM 本体能力与训练/评测/运行方式。

## 常用命令

### 本地推理

```powershell
python -m scripts.run_local_agent "比较 PostgreSQL 和 MySQL 在中小团队里的适用性"
python -m scripts.interactive_agent
python -m scripts.build_pretrain_dataset
python -m scripts.apply_pretrain_review_feedback
python -m scripts.pretrain_carm
python -m scripts.evaluate_pretraining
python -m scripts.evaluate_real_prompts
python -m scripts.analyze_reasoning_patterns
python -m scripts.evaluate_conflict_verify_candidate
python -m scripts.evaluate_tool_boundary_candidate
python -m scripts.evaluate_comparison_search_candidate
python -m scripts.evaluate_combined_tool_policy_candidate
python -m scripts.build_real_prompt_candidates
python -m scripts.auto_train
```

如果你手头已经有公开任务语料的本地文件，也可以直接导入：

```powershell
$env:CARM_PRETRAIN_IMPORT_PATHS="data/raw/public_tasks.jsonl;data/raw/prompts.txt"
python -m scripts.build_pretrain_dataset
```

支持本地 `jsonl / json / txt`。构建时会自动做任务类型推断、结构化标注、去重、质量过滤，并额外导出一份人工抽检包 `data/pretrain/review_pack.jsonl`。
默认预训练会先重置 `data/pretrain/` 下的权重产物，再从当前数据集重新训练，避免旧状态把新评估结果污染。
当前默认也会自动从 `data/experience/episodes.jsonl` 抽取高价值成功 episode，转成一批 `experience_auto` 训练样本，并顺手导出 `data/eval/real_prompt_candidates.json` 作为真实回归候选集。
同时也会启用 `teacher_distill`，借助已有 `bigmodel_proxy` 把当前 prompt 池蒸馏成一批结构化 teacher 样本，输出到 `data/pretrain/teacher_distill.jsonl`，再并入主预训练集。
如果你配置了真实 Gemini API，`bigmodel_proxy` 会优先直连真实大模型；未配置时才回退到仓库内置代理。

接入大模型有三种方式（优先级从高到低）：

1. **Gemini API**（云上，质量最高）
2. **Ollama 本地模型**（零配置，默认 `localhost:11434`）
3. **Distill 回退**（无 LLM 时的结构化模板）

```powershell
# 方式一：Gemini
$env:GEMINI_API_KEY="你的密钥"
$env:GEMINI_MODEL="gemini-2.5-flash"

# 方式二：Ollama（需先 ollama pull qwen3-coder）
$env:OLLAMA_BASE_URL="http://localhost:11434"
$env:OLLAMA_MODEL="qwen3-coder"
```

可选环境变量：

- `GEMINI_API_KEY` / `GEMINI_MODEL` / `GEMINI_TIMEOUT_S`
- `OLLAMA_BASE_URL` / `OLLAMA_MODEL` / `OLLAMA_TIMEOUT_S`

未配置任何 LLM 时，bigmodel_proxy 会给出明确提示而非返回硬编码废话，训练流水线仍可跑通（distill 模式回退）。

如果你已经人工修改了 `review_pack.jsonl` 里的字段，可以把修订结果回流到主训练集：

```powershell
python -m scripts.apply_pretrain_review_feedback
```

如果你希望把建样本、反馈回流、离线预训练、标准评估、真实 prompt 评估一次跑完，直接使用：

```powershell
python -m scripts.auto_train
```

它会输出一份完整训练报告到 `data/train_runs/auto_train_latest.json`，并按时间戳保留历史运行记录。

### 训练前清单

如果你之后想回来直接开始训练，最短路径就是：

1. 准备 Gemini API Key，并在当前终端设置 `GEMINI_API_KEY`
2. 可选设置 `GEMINI_MODEL=gemini-2.5-flash`
3. 运行 `python -m scripts.auto_train`
4. 查看 `data/train_runs/auto_train_latest.json`

这份报告会至少包含：

- 当前训练集样本数
- teacher distill 样本数
- 标准逻辑基准评估结果
- 真实 prompt 隔离评估结果
- 预训练产物目录位置

### 人工指挥推进指南

当前更推荐“人类定目标，Codex 做小步候选，评测通过后再进入慢速控制”，而不是长期依赖自动研发策略自行进化。每一轮可以按下面顺序推进：

1. 先看当前能力状态：

```powershell
python -m scripts.evaluate_real_prompts
python -m scripts.analyze_reasoning_patterns
python -m scripts.current_best
```

重点看 [current_best.json](d:/codes/Mustard/artifacts/current_best.json) 里的：

- `status`
- `real_prompt_match_rate`
- `hard_eval_pass_rate`
- `hard_eval_failures`
- `residual_explanation_rate`

2. 根据失败类型选择候选实验：

```powershell
python -m scripts.evaluate_conflict_verify_candidate
python -m scripts.evaluate_tool_boundary_candidate
python -m scripts.evaluate_comparison_search_candidate
python -m scripts.evaluate_combined_tool_policy_candidate
```

这些脚本只用临时 runner 打开候选控制项，不会改默认运行时。它们适合回答一个问题：这个小改动是否真的能修复当前失败，并且没有明显误伤 guard 场景。

3. 只有候选评测通过后，才写提案进入 Human Gate：

```powershell
Get-ChildItem backlog/proposals
```

提案里应该至少写清楚：

- 修复哪个失败样本或能力缺口
- 证据来自哪个 artifact
- 预期指标变化
- 风险等级
- 回滚方式
- 是否需要人工批准

涉及默认运行时策略、工具选择、桌面采样、训练数据准入、默认模型或数据迁移的变更，都必须先经 Human Gate。

4. 经人工批准后，再走慢速控制候选，而不是直接改稳定默认：

```powershell
python -m scripts.apply_slow_path_actions
python -m scripts.evaluate_control_versions
python -m scripts.judge_control_rollout
python -m scripts.system_status
```

例如组合工具策略候选的批准动作是：

```json
{"type":"enable_combined_tool_policy_candidate","target_module":"policy"}
```

它会同时打开：

- `policy.prefer_calculator_for_mixed_numeric_code = 1`
- `policy.prefer_search_for_comparison_evidence = 1`

但路径仍是 candidate rollout，可由 `judge_control_rollout` 决定保留、继续观察或回滚。

5. 候选稳定后，再考虑训练数据与结构改进：

```powershell
python -m scripts.build_real_prompt_candidates
python -m scripts.build_pretrain_dataset
python -m scripts.apply_pretrain_review_feedback
python -m scripts.auto_train
```

如果目标是吸收 `D:\codes\vqr` 的压缩思路，优先把它转成“推理模式码本 + 残差信号 + 质量门控”的可评测小实验。当前对应入口是：

```powershell
python -m scripts.analyze_reasoning_patterns
```

每轮结束前建议至少跑：

```powershell
python -m unittest discover -s tests -p "test_current_best.py" -v
python -m unittest discover -s tests -p "test_combined_tool_policy_candidate.py" -v
python -m unittest discover -s tests -p "test_review.py" -v
```

这条路径的原则是：默认行为不靠直觉改，先让候选在隔离评测里证明自己，再让慢速控制系统接管风险。

`review_pack.jsonl` 当前支持这些人工反馈字段：

- `review_status`: `pending / accept / reject / edit`
- `review_note`
- `override_task_type`
- `override_expected_tool`
- `override_target_slot`
- `override_plan_summary`
- `override_action_items`
- `override_unknowns`
- `override_evidence_targets`
- `override_draft_summary`

`interactive_agent` 现在支持三类显式进化命令：

```text
/goal 修复桌面桥梁里的误判链路
/prefer search 数据库选型
/evolve {"source":"interactive_cli","query":"数据库选型","preferred_tool":"search","reward":1.0}
```

如果你想先看这套闭环有没有效果，再决定是否继续扩展，直接运行：

```powershell
python -m scripts.evaluate_pretraining
python -m scripts.evaluate_real_prompts
```

它会用一组小规模基准任务，对比 `baseline` 和 `pretrained` 两个状态的：

- `tool_match_rate`
- `structured_write_rate`
- `avg_step_count`
- 每题的动作轨迹和工具选择差异

其中 `evaluate_real_prompts` 会按“每条 prompt 单独起隔离 runner”的方式做回归，更适合评估真实 prompt，避免会话内经验记忆把后一题带偏。默认基准在 [real_prompt_eval.json](d:/codes/Mustard/configs/real_prompt_eval.json)。

如果你想把实际使用过程中表现较好的 prompt 沉淀成下一批真实回归候选集，可以运行：

```powershell
python -m scripts.build_real_prompt_candidates
```

它会从 `data/experience/episodes.jsonl` 里抽取高价值、成功、且已显式使用工具的真实 episode，自动推断 `logic_skill`，并导出到 `data/eval/real_prompt_candidates.json`，方便你挑选后并入 [real_prompt_eval.json](d:/codes/Mustard/configs/real_prompt_eval.json)。

如果你想检查“工具命中之外，中间推理表示是否覆盖了高信息残差”，运行：

```powershell
python -m scripts.analyze_reasoning_patterns
python -m scripts.current_best
```

它会读取 [hard_logic_eval.json](d:/codes/Mustard/configs/hard_logic_eval.json)，输出 `pattern_id / residual_features / hard_eval_pass_rate`，并把结果并入 [current_best.json](d:/codes/Mustard/artifacts/current_best.json)。

如果 `hard_eval_failures` 指向冲突任务，可以先跑候选验证门控实验：

```powershell
python -m scripts.evaluate_conflict_verify_candidate
```

该脚本只在临时 runner 中打开 `policy.require_conflict_verify_before_answer=1`，用于验证候选策略，不会修改默认运行时控制文件。

如果失败项指向“代码与数值边界”类工具选择，可以跑：

```powershell
python -m scripts.evaluate_tool_boundary_candidate
```

该脚本只在临时 runner 中打开 `policy.prefer_calculator_for_mixed_numeric_code=1`，用于验证候选策略，不会修改默认运行时控制文件。

如果失败项指向“比较/证据任务被过早交给大模型生成”，可以跑：

```powershell
python -m scripts.evaluate_comparison_search_candidate
```

该脚本只在临时 runner 中打开 `policy.prefer_search_for_comparison_evidence=1`，并用管理层摘要任务做 guard，避免误伤正式总结场景。

如果两个工具边界候选都通过，可以跑组合候选：

```powershell
python -m scripts.evaluate_combined_tool_policy_candidate
```

该脚本在临时 runner 中同时打开两个候选控制项，用完整真实 prompt 与 hard eval 验证组合效果。

### 桌面常驻

```powershell
python -m scripts.desktop_agent_control launch
python -m scripts.desktop_agent_control start
python -m scripts.desktop_agent_control stop
python -m scripts.desktop_agent_control status --json
python -m scripts.desktop_agent_control snapshot
python -m scripts.desktop_agent_control install-startup
python -m scripts.desktop_bridge_chat
```

### 控制周期与慢速调优

```powershell
python -m scripts.run_control_cycle
python -m scripts.consolidate_reviews
python -m scripts.apply_slow_path_actions
python -m scripts.evaluate_control_versions
python -m scripts.judge_control_rollout
python -m scripts.rollback_runtime_controls
python -m scripts.system_status
```

### 数据迁移与测试

```powershell
python -m scripts.migrate_experience_schema
python -m unittest tests.test_team_conductor -v
python -m unittest discover -s tests -v
```

`python -m scripts.run_control_cycle` 默认会从 [control_cycle.json](d:/codes/Mustard/configs/control_cycle.json) 读取采样任务集。任务支持结构化字段：`id / prompt / tag / expected_tool`。`judge_control_rollout` 会把采样结果里的工具命中率作为辅助闸门，并支持按 `tag` 设置不同阈值，避免总体分数正常但某类任务已经退化。也可以用环境变量 `CARM_CONTROL_PROMPT_SET` 选择其他任务集，或直接在命令行后面传 prompt 覆盖配置。

`python -m scripts.run_desktop_agent` 会启动一个 Windows 常驻观察器，默认按 [desktop_agent.json](d:/codes/Mustard/configs/desktop_agent.json) 采样前台窗口、剪贴板变化和输入活动摘要，并把事件写到 `data/desktop/` 下。第一版只记录摘要，不记录原始按键内容。

当前默认也会按 `multimodal.screen_enabled=true` 抓取低频屏幕截图，写到 `data/desktop/screens/`。这部分不会直接把原始图像喂给 MVP，而是先压成多模态摘要再进入桥梁层。如果你后面要接入外部视觉工具，可以把 `multimodal.image_describer_command` 配成一个命令列表；仓库里附带了一个最小占位脚本 [describe_screen_stub.py](d:/codes/Mustard/scripts/describe_screen_stub.py) 作为接口示例。

如果要作为后台常驻服务使用，优先用 [desktop_agent_control.py](d:/codes/Mustard/scripts/desktop_agent_control.py)：
- `launch` 一键启动桌面代理、托盘和桥梁弹窗
- `start` 后台启动桌面代理
- `stop` 停止后台代理
- `status --json` 查看运行状态
- `snapshot` 查看更适合人读的状态快照
- `install-startup` 安装开机自启快捷方式
- `remove-startup` 移除开机自启

托盘入口在 [desktop_agent_tray.py](d:/codes/Mustard/scripts/desktop_agent_tray.py)。它会在系统托盘里提供启动、停止、状态查看、状态快照、打开数据目录和安装/移除开机自启几个动作。
当前托盘状态提示也会带上桥梁层的关键信号，包括当前目标、主动追问预算和最近一次主动状态，便于不打开弹窗时快速判断系统是否处于可打扰状态。

如果要做人机互动，使用 [desktop_bridge_chat.py](d:/codes/Mustard/scripts/desktop_bridge_chat.py)。它会读取桌面摘要生成的 bridge events，让你：
- 查看最近值得确认的桌面观察事件
- 直接和 CARM 对话
- 对事件标记 `有价值 / 误判 / 不要学习`
- 把系统推测的候选任务一键确认为当前目标
- 查看桥梁层基于 `current_goal` 生成的主动追问
- 一键把主动追问送入输入框继续协作
- 直接看到当前剩余追问预算和最近一次被压制或触发的原因

主动追问默认按 [desktop_agent.json](d:/codes/Mustard/configs/desktop_agent.json) 里的 `proactive` 配置运行，目前只做保守版触发：
- 只有存在 `current_goal` 时才考虑追问
- 只看桌面摘要、应用名、事件类型这些低维结构化信号
- 命中冷却时间或相关性不足时不会打扰
- 会消耗有限的主动追问预算，近期 `dismiss / misread` 过多时会自动收缩
- 用户给出 `useful` 反馈后，预算会小幅恢复

这些反馈会进入 `data/desktop/bridge_*.jsonl`，并回喂到当前学习链路。

桌面桥梁现在会优先消费“净化后的语义摘要”而不是原始窗口名，并额外带上：
- `semantic_tags`
- `semantic_confidence`
- `modality_hints`
- `multimodal_summary`

这样当前 MVP 接到的是低维语义信号，后续再接图像、音频、视频时也能继续沿用同一条适配链路。

## 当前状态

CARM 已进入可实用阶段。核心推理 + 工具路由 + 评测闭环均已验证通过：

**已验证能力**：

- 4 种真实工具（搜索/计算器/代码执行/大模型代理）端到端可用，10 条真实场景评测 10/10 通过
- 语义意图编码器驱动工具路由，中文自然语言算式（"N的M次方"、"每席位M元按年预算"、"平方根"等）自动提取并路由到计算器
- 统一信号模块消除跨模块重复，6 类意图信号 + 冲突检测复用同一份 token/规则定义
- 自然语言 Decoder 输出，分段展示结论/依据/可信度/风险
- 增量训练支持（`reset_artifacts=False`），5 维质量评测（tool_match / completeness / confidence / support_items / conflict_risk）
- 离线预训练 + 在线进化双环闭环，teacher distillation 可选接入真实大模型

**工具路由示例**：

| 用户输入 | 路由工具 | 输出摘要 |
|---------|---------|---------|
| 100个席位每席位129元按年预算多少 | CalculatorTool | 100×129×12 = 154,800 |
| 100的平方根 | CalculatorTool | 100^0.5 = 10 |
| 用快速排序排 5,3,8,1,9,2 | CodeExecutorTool | [1, 2, 3, 5, 8, 9] |
| 帮我写一个求阶乘的函数 | CodeExecutorTool | 120 (5!) |
| 什么是机器学习 | SearchTool→LLM | "基于大模型分析：机器学习是…" |
| 解释一下什么是递归 | SearchTool→LLM | "基于大模型分析：递归是一种…" |
| Python是什么语言 | SearchTool→LLM | LLM 知识回答 |
| 对比Python和Go的性能 | SearchTool→LLM | LLM 对比分析 |

> 当 DDGS/Wikipedia 均不可用时，搜索自动升级到 Ollama 本地 LLM 回答知识性问题。

**已知局限**：

- 核心推理核是手工权重的 tanh RNN，没有经过梯度训练，智能上限受限于规则引擎
- 搜索在部分网络环境下可能降级到 Wikipedia 或 LLM 兜底（DDGS 5s 超时保护）
- 代码执行模板覆盖有限（7 种算法），超出模板范围需要用户显式提供代码
- LLM 回答的语言/质量取决于 Ollama 模型选择，小模型可能出现非中文回答
- 连续长会话中 experience 回放可能让 FACT slot 内容膨胀

下一步的重点仍然是两条：
- 用真正的递归核或状态空间模块替换当前轻量推理核
- 把当前离线回放式预训练升级成可重复的数据集导出 + 批训练流程
