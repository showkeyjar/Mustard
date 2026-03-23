# Claw Team 开发协作与自动交付说明

> 说明：CARM 是项目目标，Claw Team 是自动优化该项目的工程手段。
> 团队角色编排见：[docs/dev/llm_team_topology.md](llm_team_topology.md)

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
