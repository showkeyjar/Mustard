# GitHub Automation

## Goal

让 Claw Team 具备两类 GitHub 能力：

1. 自动把本地变更提交到远端并创建 Pull Request
2. 自动对 Pull Request 执行保守评审
3. 在满足低风险规则时自动合并 Pull Request

## Required Environment

### Git Sync Lane

- 仓库已配置 `origin`
- 当前目录在仓库根目录
- 本地 git push 已配置可用凭据（Credential Manager / SSH / 系统登录态均可）

### PR Delivery Lane

- 满足 Git Sync Lane 的前提
- 已设置 `GH_TOKEN`、`GITHUB_TOKEN` 或 `MUSTARD_GITHUB_TOKEN`

## Required Access

- 能推送分支到远端仓库
- 能创建 Pull Request
- 能提交 Pull Request review

## Lanes

### Git Sync Lane

- 目标：自动 commit / push 选中的核心改动，不创建 PR
- 入口：`python -m scripts.claw_team_control run --auto-sync-git`
- 特点：不依赖 GitHub token；默认不使用 `git add -A`

### PR Delivery Lane

- 目标：创建 Pull Request 并进入 review / merge 流程
- 入口：`python -m scripts.claw_team_control deliver`
- 强制入口：`python -m scripts.claw_team_control deliver --force-pr`
- 特点：依赖 GitHub token 与 GitHub API

## Recommended Rollout Config

推荐先用这个保守配置上线 PR lane：

```json
{
  "github_delivery": {
    "enabled": true,
    "require_direction_correct": true,
    "require_clean_alerts": true
  },
  "auto_review": {
    "enabled": true,
    "same_repo_only": true,
    "check_commands": [
      "python -m unittest discover -s tests -v"
    ]
  },
  "auto_merge": {
    "enabled": false,
    "same_repo_only": true,
    "required_label": "claw-automerge",
    "require_approved_review": true,
    "require_clean_status": true,
    "merge_method": "squash"
  }
}
```

推荐启用顺序：

1. 先只开 `github_delivery.enabled=true`，验证自动 PR 创建是否符合预期
2. 再开 `auto_review.enabled=true`，让 PR 自动走保守评审
3. 最后再考虑 `auto_merge.enabled=true`，并且保留 `required_label`

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
