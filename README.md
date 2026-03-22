# CARM 最小原型

Compact Agentic Reasoning Model（CARM）的在线学习原型。

这个仓库提供了一套可运行的小型智能体脚手架，核心由“推理内核 + 工作记忆 + 工具总线”组成。当前版本已经升级为两段式方案：

- 预训练环节负责把历史 episode / review / evolution signal 回放成稳定初始能力
- 在线进化环节负责把用户显式目标、纠偏和偏好信号精确传给模型
- 运行时仍保留轻量在线更新，但现在受结构化用户信号约束

当前仓库也已经包含一套最小可运行的 `Claw Team` 骨架，用来持续发现问题、整理证据、生成改进提案，并把高风险决策交给人类审批。
现在也额外包含一条 GitHub 交付通道，用来自动提交代码、创建 Pull Request，以及自动审核 Pull Request。
如果你希望更自动，当前还支持直接通过 `Claw Team` 总控命令完成“跑团队周期 + 自动提交 PR”。

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

### 1. 跨机器启动 Claw Team

如果你的目标是“在任意机器上启动 Claw Team 来持续完善项目”，推荐只使用下面这条跨平台命令：

```powershell
python -m scripts.claw_team_control run
```

### 1.1 让 Claw Team 具备 GitHub 自动交付能力

如果你希望启动后的 `Claw Team` 不只是出提案，而是还能自动把代码提交到 GitHub、自动创建 PR、自动审核 PR，请额外准备一个 GitHub token。

推荐环境变量任选其一：

```powershell
$env:GH_TOKEN="你的 token"
```

或：

```powershell
$env:GITHUB_TOKEN="你的 token"
```

这个 token 至少需要覆盖两类能力：

- 推送代码到仓库分支
- 创建和审核 Pull Request

完成后先运行体检：

```powershell
python -m scripts.claw_team_github doctor
```

如果返回结果里：

- `git_ok` 为 `true`
- `remote_ok` 为 `true`
- `token_present` 为 `true`

就说明这台机器已经具备 GitHub 自动交付能力。

如果你还启用了自动评审和自动合并，`github-doctor` 最好同时满足这些附加条件：

- `required_label_present` 为 `true`
- `actions_enabled` 不是 `false`
- `can_approve_pull_request_reviews` 不是 `false`

### 1.2 自动提交代码到 GitHub

当 `Claw Team` 在本地完成一轮改动后，可以直接让它自动建分支、提交、推送并创建 PR：

```powershell
python -m scripts.claw_team_github submit-pr --title "Claw Team: improve control regression lane"
```

如果你想把这一步直接并入 `Claw Team` 主流程，使用：

```powershell
python -m scripts.claw_team_control deliver --title "Claw Team: improve control regression lane"
```

这条命令会顺序执行：

1. 跑一轮 `Claw Team`
2. 检查 GitHub 交付条件
3. 自动建分支
4. 自动提交当前改动
5. 自动推送
6. 自动创建 Draft Pull Request

这条命令默认会：

- 如果当前在 `main`，自动切一个 `claw/<timestamp>` 分支
- 自动 `git add -A`
- 自动提交当前改动
- 自动推送到 `origin`
- 自动创建一个 Draft Pull Request

也可以显式指定：

```powershell
python -m scripts.claw_team_github submit-pr `
  --title "Claw Team: improve control regression lane" `
  --commit-message "Claw Team: improve control regression lane" `
  --reviewer your-github-id
```

### 1.3 自动审核 Pull Request

本地可以直接让团队自动审一个 PR：

```powershell
python -m scripts.claw_team_github review-pr --pr 123 --event auto
```

也可以直接走团队总控入口：

```powershell
python -m scripts.claw_team_control review --pr 123 --event auto
```

### 1.5 低风险 PR 自动合并

现在也支持低风险 PR 的自动合并：

```powershell
python -m scripts.claw_team_github merge-pr --pr 123
```

或：

```powershell
python -m scripts.claw_team_control merge --pr 123
```

当前自动合并是保守策略，默认只有同时满足下面条件才会执行：

1. PR 不是 Draft
2. PR 可合并
3. PR 来自同一个仓库
4. PR 带有 `claw-automerge` 标签
5. 至少有一个 `APPROVED` review
6. 该 PR 的提交状态为成功

默认合并方式来自 [team_github.json](/d:/codes/Mustard/configs/team_github.json)，当前是 `squash`。

默认策略是：

- 本地校验失败：`REQUEST_CHANGES`
- Draft PR：`COMMENT`
- 校验通过且可合并：`APPROVE`

默认校验命令来自 [team_github.json](/d:/codes/Mustard/configs/team_github.json)，目前是：

```powershell
python -m unittest discover -s tests -v
```

你也可以临时覆盖：

```powershell
python -m scripts.claw_team_github review-pr `
  --pr 123 `
  --event auto `
  --check-command "python -m unittest tests.test_team_conductor -v"
```

### 1.4 GitHub 上的自动 PR 审核

仓库现在也附带了一个 workflow：

- [.github/workflows/claw-team-pr-review.yml](/d:/codes/Mustard/.github/workflows/claw-team-pr-review.yml)
- [.github/workflows/claw-team-pr-merge.yml](/d:/codes/Mustard/.github/workflows/claw-team-pr-merge.yml)

它们会在同仓库 PR 打开、更新、重新打开、转为 ready for review、提交 review 或打标签后，自动运行 Claw Team 的 review / merge 通道。

要让这个 workflow 真正能自动提交 review，需要在 GitHub 仓库设置里确认两件事：

1. `GITHUB_TOKEN` 具有足够权限
2. 仓库允许 GitHub Actions 创建和批准 Pull Request reviews

根据 GitHub 官方文档，这个选项默认可能是关闭的，需要在仓库的 `Actions -> General -> Workflow permissions` 里打开 “Allow GitHub Actions to create and approve pull requests”。

如果你还想让自动合并真正可用，建议再确认：

1. `GITHUB_TOKEN` 对 `contents` 和 `pull-requests` 具备写权限
2. 分支保护规则允许满足条件后的自动合并
3. 团队只给低风险 PR 打上 `claw-automerge` 标签

这条命令会自动完成：

- 初始化 `team/`、`memory/`、`backlog/` 工作区
- 读取当前训练、评测、控制和桥梁信号
- 生成当天的团队日报到 `memory/daily/`
- 在命中条件时生成改进提案到 `backlog/proposals/`

如果你想先检查当前机器是否已经具备正确启动条件，先运行：

```powershell
python -m scripts.claw_team_control doctor
```

如果你想只初始化团队目录，不跑周期：

```powershell
python -m scripts.claw_team_control bootstrap
```

如果你想查看当前团队状态：

```powershell
python -m scripts.claw_team_control status
```

Windows 下也可以直接双击：

- [start_claw_team.cmd](d:/codes/Mustard/start_claw_team.cmd)
- [start_claw_team.vbs](d:/codes/Mustard/start_claw_team.vbs)

但为了确保在任意机器上都能复用，仍然建议优先记住统一命令：

```powershell
python -m scripts.claw_team_control run
```

### 2. 判断 Claw Team 是否启动成功

一次成功启动至少应满足下面 4 个结果：

1. 命令正常退出，没有 Python traceback
2. 生成或更新 `memory/daily/YYYY-MM-DD.md`
3. `team/`、`memory/`、`backlog/` 目录存在
4. `doctor` 返回的 `missing_files` 为空

最短验证路径：

```powershell
python -m scripts.claw_team_control doctor
python -m scripts.claw_team_control run
python -m scripts.claw_team_control status
```

### 3. Claw Team 在做什么

当前第一版 `Claw Team` 不会自动改主分支，也不会自动修改默认运行时行为。它会负责：

- 汇总系统状态
- 观察回归与风险信号
- 生成日报
- 生成候选提案
- 自动提交本地改动为 GitHub PR
- 自动对 PR 做保守评审
- 在低风险条件满足时自动合并 PR
- 标记哪些提案必须经过人类审批

这意味着你可以安全地在新机器上把它跑起来，同时又让它具备真正的交付能力；但默认主分支上线和高风险行为变更仍然保留给人类把关。

### 4. 推荐的跨机器启动顺序

在一台全新的机器上，推荐按这个顺序启动：

1. `git clone <repo>`
2. `cd Mustard`
3. `python --version`
4. `python -m unittest tests.test_team_conductor -v`
5. `python -m scripts.claw_team_control doctor`
6. `python -m scripts.claw_team_control run`
7. `python -m scripts.claw_team_control status`
8. `python -m scripts.claw_team_github doctor`

如果第 4 步和第 5 步都通过，基本就说明这台机器已经具备运行 Claw Team 的条件。

### 5. 推荐的人机协作方式

为了让 Claw Team 真正持续完善项目，建议固定这个节奏：

1. 先运行 `python -m scripts.claw_team_control run`
2. 查看 `memory/daily/` 和 `backlog/proposals/`
3. 团队完成改动后运行 `python -m scripts.claw_team_github submit-pr --title "..."`
4. 或者直接运行 `python -m scripts.claw_team_control deliver --title "..."`
5. 自动或手动触发 PR review
6. 对低风险 PR 打上 `claw-automerge` 标签，让团队尝试自动合并
7. 人类只挑高价值或高风险提案做审批
8. 再针对被批准的提案运行训练、评测或控制脚本

对应的高频命令通常是：

```powershell
python -m scripts.claw_team_control run
python -m scripts.claw_team_control deliver --title "Claw Team: your change"
python -m scripts.claw_team_github submit-pr --title "Claw Team: your change"
python -m scripts.claw_team_control review --pr 123 --event auto
python -m scripts.claw_team_github review-pr --pr 123 --event auto
python -m scripts.claw_team_control merge --pr 123
python -m scripts.claw_team_github merge-pr --pr 123
python -m scripts.auto_train
python -m scripts.evaluate_real_prompts
python -m scripts.run_control_cycle
python -m scripts.judge_control_rollout
```

### 6. GitHub 仓库配置 Checklist

想让任何机器上的 `Claw Team` 都具备完整 GitHub 自动化能力，仓库侧建议按这份 checklist 配置：

1. 安装并启用 GitHub Actions
2. 在 `Actions -> General -> Workflow permissions` 中允许工作流写入仓库内容
3. 在同一页面启用 “Allow GitHub Actions to create and approve pull requests”
4. 为默认分支设置保护规则
5. 要求关键测试通过后才能合并
6. 只对低风险 PR 使用 `claw-automerge` 标签
7. 在使用本地交付命令的机器上配置 `GH_TOKEN` 或 `GITHUB_TOKEN`

推荐把 `claw-automerge` 视为“明确允许自动合并”的授权标签，而不是默认标签。

如果你想最快确认仓库侧是否已经配好，直接运行：

```powershell
python -m scripts.claw_team_control github-doctor
```

重点看这几项：

- `required_label_present`
- `actions_enabled`
- `workflow_default_permissions`
- `can_approve_pull_request_reviews`
- `diagnostics`

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
