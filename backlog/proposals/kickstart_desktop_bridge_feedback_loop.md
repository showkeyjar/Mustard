# Kickstart desktop-bridge feedback loop

- problem: 桌面桥梁暂无反馈样本，创新闭环缺少用户纠偏信号。
- change_type: desktop_behavior
- risk_level: medium
- needs_human_approval: True
- proposed_change: 优先采集并整理 bridge useful/misread 反馈，形成可训练的反馈样本包。
- rollback_plan: 仅进行反馈采集与标注，不调整桌面采样或主动追问默认策略。
- evidence:
  - bridge_feedback=0
- evaluation_plan:
  - python -m scripts.desktop_agent_control snapshot
  - python -m scripts.desktop_bridge_chat
