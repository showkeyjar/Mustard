# GitHub Automation

## Goal

让 Claw Team 具备两类 GitHub 能力：

1. 自动把本地变更提交到远端并创建 Pull Request
2. 自动对 Pull Request 执行保守评审
3. 在满足低风险规则时自动合并 Pull Request

## Required Environment

- 仓库已配置 `origin`
- 当前目录在仓库根目录
- 已设置 `GH_TOKEN`、`GITHUB_TOKEN` 或 `MUSTARD_GITHUB_TOKEN`

## Required Access

- 能推送分支到远端仓库
- 能创建 Pull Request
- 能提交 Pull Request review

## Commands

### Doctor

```powershell
python -m scripts.claw_team_github doctor
```

医生检查现在会尽量覆盖：

- 本地 git 与远端 `origin`
- GitHub token 是否存在
- 仓库默认分支
- `claw-automerge` 标签是否存在
- GitHub Actions 是否启用
- workflow 默认权限
- 是否允许 workflow 自动批准 PR reviews

### Submit PR

```powershell
python -m scripts.claw_team_github submit-pr --title "Claw Team: improve regression lane"
```

### Review PR

```powershell
python -m scripts.claw_team_github review-pr --pr 123 --event auto
```

### Merge PR

```powershell
python -m scripts.claw_team_github merge-pr --pr 123
```

## Review Policy

- 本地校验失败时默认 `REQUEST_CHANGES`
- Draft PR 默认只做 `COMMENT`
- 校验通过且 PR 可合并时默认 `APPROVE`
- 高风险或行为变化较大的 PR 仍需人类最终确认

## Auto-Merge Policy

- 默认只允许同仓库 PR 参与自动合并
- 必须带有 `claw-automerge` 标签
- 必须至少存在一个 `APPROVED` review
- 必须通过提交状态检查
- 默认使用 `squash` 合并
