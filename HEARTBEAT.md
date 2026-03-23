# HEARTBEAT.md

每次心跳只做以下检查：

1. 运行 `python -m scripts.claw_team_control status`
2. 必要时运行 `python -m scripts.claw_team_control doctor`
3. 如果仓库存在新的低风险改进空间，整理到 `backlog/proposals/`
4. 如果缺少证据，不要编造结论，记录待验证项

如果当前没有需要处理的项目事项，回复 `HEARTBEAT_OK`。
