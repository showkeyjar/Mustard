# CARM 最小原型

Compact Agentic Reasoning Model（CARM）的在线学习原型。

这个仓库提供了一套可运行的小型智能体脚手架，核心由“推理内核 + 工作记忆 + 工具总线”组成。当前版本是混合式方案：

- 用启发式先验提供冷启动行为
- 用轻量在线动作头在每轮对话后更新策略
- 用可持久化轨迹把对话沉淀成经验记忆

## 当前包含

- 结构化智能体状态与动作协议
- 固定槽位的工作记忆板
- 带持久化权重的在线策略控制环
- 从历史对话中学习动作与工具偏好的概念层
- 学习何时形成 `PLAN`、`HYP`、`DRAFT` 的自适应推理核
- 带 `action_items / unknowns / evidence_targets / confidence_band` 的结构化中间表示
- 搜索、计算器、代码执行和大模型代理的模拟工具
- 用于动作奖励和对话摘要的经验回放存储
- 可直接运行的本地入口
- 覆盖记忆流、代理主循环和桌面桥梁的测试

## 快速开始

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

## 常用命令

### 本地推理

```powershell
python -m scripts.run_local_agent "比较 PostgreSQL 和 MySQL 在中小团队里的适用性"
python -m scripts.interactive_agent
```

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

这是一套在线学习脚手架。当前版本已经可以在推理时更新动作头，从成功对话里学习概念层的动作/工具偏好，用轻量自适应推理核学习 `PLAN`、`HYP`、`DRAFT` 的槽位形成倾向，并把这些状态收敛到更适合训练的结构化 schema 里。同时，运行时控制项已经支持版本化、审计和回滚。

下一步的重点仍然是两条：
- 用真正的递归核或状态空间模块替换当前轻量推理核
- 增加轨迹监督训练，让中间状态和动作决策更少依赖启发式
