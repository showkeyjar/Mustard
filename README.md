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
- 搜索、计算器、代码执行和大模型代理的模拟工具
- 用于动作奖励和对话摘要的经验回放存储
- 离线预训练脚本与预训练产物清单
- 结构化在线进化信号管理器
- 可直接运行的本地入口
- 覆盖记忆流、代理主循环和桌面桥梁的测试

## 快速开始

### 0. 环境要求

想在任意机器上正确启动本项目，先满足这几个条件：

- 安装 Python `3.10+`
- 能在命令行里直接运行 `python`
- 已把仓库完整拉取到本地
- 当前终端工作目录位于仓库根目录

检查 Python 版本：

```powershell
python --version
```

如果你要运行 `Claw Team`，上面这些条件就够了。
如果你还要运行桌面常驻代理，请额外注意：桌面代理当前是 Windows 优先路径，`Claw Team` 本身则是跨平台的纯 Python 入口。

第一次使用只看这一条就够了：

双击仓库根目录下的 [start_carm.vbs](d:/codes/Mustard/start_carm.vbs)。

它会一次性完成这三件事：
- 启动桌面常驻代理
- 拉起系统托盘入口
- 打开桌面桥梁弹窗

如果你更习惯命令行，等价命令是：

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

接入真实大模型最简单的方式是设置：

```powershell
$env:GEMINI_API_KEY="你的密钥"
$env:GEMINI_MODEL="gemini-2.5-flash"
python -m scripts.auto_train
```

可选环境变量：

- `GEMINI_API_KEY`
- `GEMINI_MODEL`
- `GEMINI_TIMEOUT_S`

如果你现在还不准备接真实大模型，也可以先不配这些环境变量。此时 `bigmodel_proxy` 会自动回退到仓库内置代理，整条训练流水线仍然可以跑通，只是 teacher distillation 的上限会低一些。

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

这是一套两段式学习脚手架。当前版本已经可以：

- 把历史成功轨迹回放成预训练产物
- 用低成本模板生成 + 程序化标注自动产出通用任务理解/规划预训练集
- 从成功对话里学习概念层的动作/工具偏好
- 用轻量自适应推理核学习 `PLAN`、`HYP`、`DRAFT` 的槽位形成倾向
- 用结构化在线信号把 `goal / preferred_tool / preferred_slot / reward / learn` 精确回喂到运行中的模型
- 把这些状态收敛到更适合训练和审计的结构化 schema 里

低成本预训练数据方案现在默认会生成一份 `data/pretrain/pretrain_corpus.jsonl`，覆盖 `compare / calculate / coding / planning / summarize / fact_check` 六类通用任务。每条样本都会自动带上：

- `expected_tool`
- `target_slot`
- `plan_action_items`
- `plan_unknowns`
- `evidence_targets`
- `draft_summary`
- `quality_score`

你也可以把公开语料先放到本地，再通过导入器并入同一份数据集。当前导入器会优先读取：

- `prompt`
- `question`
- `instruction`
- `input`

然后自动推断任务类型并转成统一的 CARM 训练结构。

下一步的重点仍然是两条：
- 用真正的递归核或状态空间模块替换当前轻量推理核
- 把当前离线回放式预训练升级成可重复的数据集导出 + 批训练流程
