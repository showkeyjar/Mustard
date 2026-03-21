from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from carm.actions import Action
from carm.experience import ExperienceStore
from carm.schemas import EpisodeRecord, StepRecord


TASK_TYPES = (
    "compare",
    "calculate",
    "coding",
    "planning",
    "summarize",
    "fact_check",
)

LOGIC_SKILLS = (
    "classification",
    "comparison",
    "constraint_planning",
    "evidence_judgment",
    "conflict_detection",
    "tool_selection",
    "result_integration",
    "termination_judgment",
    "step_planning",
)


@dataclass
class PretrainSample:
    user_input: str
    task_type: str
    source_type: str
    expected_tool: str
    target_slot: str
    logic_skill: str = ""
    plan_summary: str = ""
    plan_action_items: list[str] = field(default_factory=list)
    plan_unknowns: list[str] = field(default_factory=list)
    evidence_targets: list[str] = field(default_factory=list)
    draft_summary: str = ""
    quality_score: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)


COMPARE_ITEMS = [
    ("PostgreSQL", "MySQL", "中小团队"),
    ("Redis", "Memcached", "缓存服务"),
    ("FastAPI", "Flask", "内部工具"),
    ("SQLite", "PostgreSQL", "本地原型"),
]

CALCULATE_ITEMS = [
    "请计算 12 + 30 / 3",
    "一个项目每月成本 2300 元，连续 6 个月总成本是多少",
    "如果 4 台机器每台每天处理 120 个任务，7 天共处理多少任务",
    "预算 15000 元，实际花了 12380 元，还剩多少",
    "我们团队 9 个人，某 SaaS 每席位 129 元/月，如果按年预算估算请按 129 * 9 * 12 计算",
    "一次性迁移 48000 条数据，如果每批 6000 条，需要分几批处理",
]

CODING_ITEMS = [
    "帮我理解这段 Python 脚本为什么会报空指针错误",
    "给我一个修复 API 超时问题的 Python 排查思路",
    "分析这段脚本为什么没有正确写入 JSON 文件",
    "比较用脚本和手工方式处理日志的优缺点",
]

PLANNING_ITEMS = [
    "给我一个分步骤方案来整理小团队的技术选型流程",
    "如何规划一次低风险的数据库迁移",
    "帮我制定一个一周内完成产品原型的执行计划",
    "设计一个适合中小团队的代码评审流程",
]

SUMMARIZE_ITEMS = [
    "如何总结一篇技术文章的关键结论并保留证据点",
    "给我一个总结会议纪要的通用步骤",
    "如何把长文档压缩成面向决策的摘要",
    "总结一个产品需求文档时应该优先提炼什么",
    "我已经收集了官方文档、社区最佳实践和故障复盘，如何组织成给负责人的正式结论",
]

FACT_CHECK_ITEMS = [
    "如何核验一条技术建议是否可靠",
    "面对多个资料来源时，怎样确认事实没有冲突",
    "怎样验证一个数据库性能结论是否站得住",
    "如何判断一篇教程里的代码建议是否过时",
]

LOGIC_PRIMITIVE_ITEMS = [
    ("对于问题“请计算 18 * 7”，应该优先调用哪类工具来保证结果可靠？", "calculate", "tool_selection", "calculator", "HYP"),
    ("比较 PostgreSQL 和 MySQL 之前，应该先明确哪些比较维度？", "compare", "comparison", "search", "PLAN"),
    ("如果需求里同时要求低成本、快速上线和后期可扩展，应该怎么先拆约束？", "planning", "constraint_planning", "search", "PLAN"),
    ("要验证一个数据库性能结论是否站得住，应该先补哪类证据？", "fact_check", "evidence_judgment", "search", "HYP"),
    ("多个来源给出相互冲突的数据库迁移建议时，应该先怎么处理冲突？", "fact_check", "conflict_detection", "search", "HYP"),
    ("拿到若干检索结果后，如何整合成一份带风险说明的结论？", "summarize", "result_integration", "search", "DRAFT"),
    ("什么时候应该停止继续检索并开始形成结论？", "fact_check", "termination_judgment", "search", "DRAFT"),
    ("面对一项模糊任务时，如何先把执行步骤列出来？", "planning", "step_planning", "search", "PLAN"),
    ("看到一个新问题时，应该先判断它是计算、检索还是代码执行任务吗？", "planning", "classification", "search", "PLAN"),
]

LOGIC_HARD_CASE_ITEMS = [
    (
        "有篇博客说 Python 3.13 默认已经启用无 GIL 构建，这种说法更应该查官方资料还是直接本地跑个脚本？",
        "fact_check",
        "tool_selection",
        "search",
        "HYP",
    ),
    (
        "套餐说明写着每席位 79 元，12 人团队按年付总价是多少？请按 79 * 12 * 12 精确计算，不要泛泛比较。",
        "calculate",
        "tool_selection",
        "calculator",
        "HYP",
    ),
    (
        "我已经拿到官方文档、发布日期和迁移指南三项证据，还要继续检索才能判断这个功能是否可用吗？",
        "fact_check",
        "termination_judgment",
        "search",
        "DRAFT",
    ),
    (
        "两个来源对同一个数据库参数给出了相反建议，在冲突还没消解前应该直接下结论吗？",
        "fact_check",
        "conflict_detection",
        "search",
        "HYP",
    ),
    (
        "我已经收集了几份检索材料，请把它们整合成一份面向管理层的正式结论，语言要完整稳妥。",
        "summarize",
        "result_integration",
        "bigmodel_proxy",
        "DRAFT",
    ),
    (
        "我已经收集了官方文档、社区最佳实践和故障复盘，现在要写给负责人一份简洁但正式的结论，应该怎么组织？",
        "summarize",
        "result_integration",
        "bigmodel_proxy",
        "DRAFT",
    ),
    (
        "一个问题里同时出现代码片段和预算数字时，应该先判断它到底是要运行代码还是做精确计算吗？",
        "planning",
        "classification",
        "search",
        "PLAN",
    ),
    (
        "这个问题里既有一段 Python 代码，又问到一次性迁移 48000 条数据预计分几批处理，每批 6000 条。你会先走哪类工具？",
        "calculate",
        "tool_selection",
        "calculator",
        "HYP",
    ),
]

PROMPT_PREFIXES = [
    "",
    "请给出一个稳妥的处理思路：",
    "从中小团队落地角度看，",
    "如果要尽快推进，",
]

PROMPT_SUFFIXES = [
    "",
    "，请先列出关键步骤再给结论。",
    "，并指出仍需补充的证据。",
    "，输出时尽量保持结构化。",
]


def generate_task_pool(seed: int = 7, count_per_type: int = 24) -> list[PretrainSample]:
    rng = random.Random(seed)
    samples: list[PretrainSample] = []

    samples.extend(_generate_compare_samples(rng, count_per_type))
    samples.extend(_generate_simple_samples(rng, CALCULATE_ITEMS, "calculate", count_per_type))
    samples.extend(_generate_simple_samples(rng, CODING_ITEMS, "coding", count_per_type))
    samples.extend(_generate_simple_samples(rng, PLANNING_ITEMS, "planning", count_per_type))
    samples.extend(_generate_simple_samples(rng, SUMMARIZE_ITEMS, "summarize", count_per_type))
    samples.extend(_generate_simple_samples(rng, FACT_CHECK_ITEMS, "fact_check", count_per_type))
    samples.extend(_generate_logic_primitive_samples(rng, count_per_type))
    samples.extend(_generate_logic_hard_case_samples(rng, count_per_type))
    return [annotate_sample(sample) for sample in samples]


def import_raw_tasks(paths: list[str | Path]) -> list[PretrainSample]:
    samples: list[PretrainSample] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            samples.extend(_import_jsonl_tasks(path))
        elif suffix == ".json":
            samples.extend(_import_json_tasks(path))
        else:
            samples.extend(_import_text_tasks(path))
    return [annotate_sample(sample) for sample in samples if sample.user_input.strip()]


def build_samples_from_experience(
    experience_path: str | Path,
    *,
    min_value_score: float = 0.7,
    limit: int = 300,
) -> list[PretrainSample]:
    episodes = ExperienceStore(experience_path).load_all()
    samples: list[PretrainSample] = []

    for episode in episodes:
        sample = episode_to_pretrain_sample(episode, min_value_score=min_value_score)
        if sample is not None:
            samples.append(sample)

    deduped = dedupe_samples(samples)
    deduped.sort(key=lambda item: (item.quality_score, len(item.plan_action_items), len(item.user_input)), reverse=True)
    return deduped[:limit]


def annotate_sample(sample: PretrainSample) -> PretrainSample:
    task_type = sample.task_type
    user_input = sample.user_input
    logic_skill = sample.logic_skill or infer_logic_skill(user_input, task_type)
    metadata = dict(sample.metadata)
    forced_expected_tool = str(metadata.get("forced_expected_tool", ""))
    forced_target_slot = str(metadata.get("forced_target_slot", ""))

    expected_tool = "search"
    target_slot = "PLAN"
    action_items = ["明确目标与约束", "识别需要验证的维度", "形成结构化输出"]
    unknowns = ["缺少关键事实支撑"]
    evidence_targets = ["外部事实"]
    draft_summary = "基于计划形成可验证的初步结论。"
    quality_score = 0.65

    if task_type == "compare":
        expected_tool = "search"
        target_slot = "PLAN"
        action_items = ["确定比较对象与场景", "列出比较维度", "收集事实后给出取舍建议"]
        unknowns = ["比较维度可能不完整", "缺少场景化证据"]
        evidence_targets = ["成本", "性能", "生态", "适用场景"]
        draft_summary = "按比较维度整合证据，输出带取舍理由的建议。"
        quality_score = 0.9
    elif task_type == "calculate":
        expected_tool = "calculator"
        target_slot = "HYP"
        action_items = ["识别题目中的数值关系", "执行精确计算", "核对结果与单位"]
        unknowns = ["运算顺序或单位可能被误读"]
        evidence_targets = ["精确数值", "单位"]
        draft_summary = "得到数值结果后给出简要解释。"
        quality_score = 0.92
    elif task_type == "coding":
        expected_tool = "code_executor"
        target_slot = "PLAN"
        action_items = ["定位异常现象", "构造最小复现或检查点", "验证修复方向"]
        unknowns = ["报错上下文可能不完整", "需要执行验证"]
        evidence_targets = ["异常堆栈", "输入输出", "可执行验证"]
        draft_summary = "基于复现和证据给出修复建议。"
        quality_score = 0.84
    elif task_type == "planning":
        expected_tool = "search"
        target_slot = "PLAN"
        action_items = ["拆解阶段目标", "识别依赖与风险", "输出执行顺序与里程碑"]
        unknowns = ["约束条件不完整", "风险边界未核实"]
        evidence_targets = ["执行约束", "关键依赖", "风险点"]
        draft_summary = "形成可执行、可追踪的行动计划。"
        quality_score = 0.86
    elif task_type == "summarize":
        expected_tool = "search"
        target_slot = "DRAFT"
        action_items = ["提取核心信息", "压缩为少量结论", "保留证据与风险"]
        unknowns = ["原文重点可能分散", "证据引用可能不足"]
        evidence_targets = ["关键结论", "证据点", "未决问题"]
        draft_summary = "输出面向决策的高密度摘要。"
        quality_score = 0.78
    elif task_type == "fact_check":
        expected_tool = "search"
        target_slot = "HYP"
        action_items = ["拆出待核验陈述", "交叉查找来源", "确认是否冲突并给出结论"]
        unknowns = ["来源可信度不一致", "结论可能随时间变化"]
        evidence_targets = ["来源可信度", "时间有效性", "交叉印证"]
        draft_summary = "根据交叉证据判断结论是否成立。"
        quality_score = 0.82

    if logic_skill == "tool_selection":
        action_items = ["识别问题本质", "判断最可靠的外部能力", "避免凭记忆直接回答"]
        unknowns = ["若工具选错，后续结论会失真"]
        evidence_targets = ["工具能力边界", "任务所需证据类型"]
        draft_summary = "先选对工具，再进入后续推理。"
        quality_score = max(quality_score, 0.9)
    elif logic_skill == "evidence_judgment":
        action_items = ["拆出待验证结论", "判断缺的证据类型", "再决定是否继续求证"]
        unknowns = ["当前证据是否足够仍不明确"]
        evidence_targets = ["证据类型", "证据充分性", "来源质量"]
    elif logic_skill == "conflict_detection":
        action_items = ["识别冲突陈述", "标记冲突来源", "优先澄清分歧再给结论"]
        unknowns = ["冲突可能来自时效或口径差异"]
        evidence_targets = ["冲突点", "来源时效", "定义口径"]
        draft_summary = "冲突未消解前，不应把猜测包装成确定结论。"
    elif logic_skill == "result_integration":
        target_slot = "DRAFT"
        action_items = ["提炼共同结论", "保留关键证据", "显式写出风险和未决点"]
        evidence_targets = ["共同证据", "分歧证据", "剩余风险"]
        draft_summary = "把外部结果整合成清晰结论，必要时交给更强生成能力润色输出。"
    elif logic_skill == "termination_judgment":
        target_slot = "DRAFT"
        action_items = ["检查是否已有足够证据", "判断边际检索收益", "满足阈值后收束结论"]
        unknowns = ["是否还存在关键未决点"]
        evidence_targets = ["证据充分性", "剩余风险"]
        draft_summary = "当证据足够且风险可见时，应停止扩展并形成结论。"
    elif logic_skill == "classification":
        action_items = ["识别问题属于哪类推理任务", "映射到合适工具与槽位", "再进入执行阶段"]
        evidence_targets = ["任务类型", "工具映射", "输出结构"]

    expected_tool = forced_expected_tool or expected_tool
    target_slot = forced_target_slot or target_slot

    return PretrainSample(
        user_input=user_input,
        task_type=task_type,
        source_type=sample.source_type,
        expected_tool=expected_tool,
        target_slot=target_slot,
        logic_skill=logic_skill,
        plan_summary=f"围绕任务 `{task_type}` 建立求解路径。",
        plan_action_items=action_items,
        plan_unknowns=unknowns,
        evidence_targets=evidence_targets,
        draft_summary=draft_summary,
        quality_score=quality_score,
        metadata=metadata,
    )


def episode_to_pretrain_sample(
    episode: EpisodeRecord,
    *,
    min_value_score: float = 0.7,
) -> PretrainSample | None:
    if not episode.success or episode.value_score < min_value_score:
        return None

    prompt = episode.user_input.strip()
    if not prompt:
        return None

    features = dict(episode.episode_features or {})
    expected_tool = str(features.get("used_tool", "")).strip()
    if not expected_tool:
        return None

    task_type = infer_task_type(prompt)
    logic_skill = str(features.get("logic_skill", "")).strip() or infer_logic_skill(prompt, task_type)
    action_items = [str(item).strip() for item in features.get("plan_action_items", []) if str(item).strip()]
    unknowns = [str(item).strip() for item in features.get("plan_unknowns", []) if str(item).strip()]
    evidence_targets = [str(item).strip() for item in features.get("evidence_targets", []) if str(item).strip()]
    plan_summary = str(features.get("plan_summary", "")).strip()
    draft_summary = str(features.get("draft_summary", "")).strip()

    slot_order = [step.target_slot for step in episode.steps if step.target_slot]
    target_slot = next((slot for slot in slot_order if slot in {"PLAN", "HYP", "DRAFT"}), "")

    sample = PretrainSample(
        user_input=prompt,
        task_type=task_type,
        source_type="experience_auto",
        expected_tool=expected_tool,
        target_slot=target_slot,
        logic_skill=logic_skill,
        plan_summary=plan_summary,
        plan_action_items=action_items,
        plan_unknowns=unknowns,
        evidence_targets=evidence_targets,
        draft_summary=draft_summary,
        quality_score=max(0.72, min(float(episode.value_score), 0.98)),
        metadata={
            "control_version": str(features.get("control_version", "")),
            "action_sequence": list(features.get("action_sequence", [])),
            "result_brief": str(features.get("result_brief", "")),
        },
    )
    return annotate_sample(sample)


def save_pretrain_samples(path: str | Path, samples: list[PretrainSample]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")


def merge_and_filter_samples(
    samples: list[PretrainSample],
    *,
    min_quality_score: float = 0.72,
    max_samples: int = 5000,
) -> list[PretrainSample]:
    deduped = dedupe_samples(samples)
    filtered = [sample for sample in deduped if score_sample_quality(sample) >= min_quality_score]
    filtered.sort(
        key=lambda item: (item.quality_score, len(item.evidence_targets), len(item.plan_action_items), len(item.user_input)),
        reverse=True,
    )
    return filtered[:max_samples]


def dedupe_samples(samples: list[PretrainSample]) -> list[PretrainSample]:
    best_by_key: dict[str, PretrainSample] = {}
    for sample in samples:
        normalized = normalize_user_input(sample.user_input)
        existing = best_by_key.get(normalized)
        scored = score_sample_quality(sample)
        sample.quality_score = scored
        if existing is None or scored > existing.quality_score:
            best_by_key[normalized] = sample
    return list(best_by_key.values())


def score_sample_quality(sample: PretrainSample) -> float:
    score = float(sample.quality_score or 0.0)
    text = sample.user_input.strip()
    if len(text) >= 12:
        score += 0.05
    if len(sample.plan_action_items) >= 3:
        score += 0.08
    if sample.evidence_targets:
        score += 0.06
    if sample.plan_unknowns:
        score += 0.04
    if sample.expected_tool in {"search", "calculator", "code_executor", "bigmodel_proxy"}:
        score += 0.05
    if sample.target_slot in {"PLAN", "HYP", "DRAFT"}:
        score += 0.04
    if sample.logic_skill in LOGIC_SKILLS:
        score += 0.04
    if _has_diverse_signal(text):
        score += 0.03
    return round(max(0.0, min(score, 0.99)), 4)


def export_review_pack(path: str | Path, samples: list[PretrainSample], limit: int = 50) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    selected = sorted(samples, key=lambda item: (item.quality_score, len(item.user_input)), reverse=True)[:limit]
    with target.open("w", encoding="utf-8") as handle:
        for sample in selected:
            payload = {
                "user_input": sample.user_input,
                "task_type": sample.task_type,
                "logic_skill": sample.logic_skill,
                "source_type": sample.source_type,
                "expected_tool": sample.expected_tool,
                "target_slot": sample.target_slot,
                "quality_score": sample.quality_score,
                "plan_action_items": sample.plan_action_items,
                "plan_unknowns": sample.plan_unknowns,
                "evidence_targets": sample.evidence_targets,
                "plan_summary": sample.plan_summary,
                "draft_summary": sample.draft_summary,
                "review_status": "pending",
                "review_note": "",
                "override_task_type": "",
                "override_expected_tool": "",
                "override_target_slot": "",
                "override_plan_summary": "",
                "override_action_items": [],
                "override_unknowns": [],
                "override_evidence_targets": [],
                "override_draft_summary": "",
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def apply_review_feedback(
    dataset_path: str | Path,
    review_pack_path: str | Path,
    *,
    output_path: str | Path | None = None,
    min_quality_score: float = 0.72,
    max_samples: int = 5000,
) -> list[PretrainSample]:
    base_samples = load_pretrain_samples(dataset_path)
    review_payloads = load_review_feedback(review_pack_path)
    if not review_payloads:
        return base_samples

    by_key = {normalize_user_input(sample.user_input): sample for sample in base_samples}
    for payload in review_payloads:
        status = str(payload.get("review_status", "pending")).strip().lower()
        if status in {"", "pending"}:
            continue

        sample = review_payload_to_sample(payload)
        key = normalize_user_input(sample.user_input)
        if status == "reject":
            by_key.pop(key, None)
            continue

        if status in {"accept", "edit"}:
            existing = by_key.get(key)
            sample.quality_score = max(sample.quality_score, 0.95 if status == "edit" else 0.9)
            if existing is None or sample.quality_score >= existing.quality_score:
                by_key[key] = sample

    merged = merge_and_filter_samples(list(by_key.values()), min_quality_score=min_quality_score, max_samples=max_samples)
    target = Path(output_path) if output_path is not None else Path(dataset_path)
    save_pretrain_samples(target, merged)
    return merged


def load_pretrain_samples(path: str | Path) -> list[PretrainSample]:
    target = Path(path)
    if not target.exists():
        return []
    samples: list[PretrainSample] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            samples.append(PretrainSample(**json.loads(line)))
    return samples


def load_review_feedback(path: str | Path) -> list[dict[str, object]]:
    target = Path(path)
    if not target.exists():
        return []
    payloads: list[dict[str, object]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                payloads.append(payload)
    return payloads


def review_payload_to_sample(payload: dict[str, object]) -> PretrainSample:
    action_items = payload.get("override_action_items") or payload.get("plan_action_items") or []
    unknowns = payload.get("override_unknowns") or payload.get("plan_unknowns") or []
    evidence_targets = payload.get("override_evidence_targets") or payload.get("evidence_targets") or []
    task_type = str(payload.get("override_task_type") or payload.get("task_type") or infer_task_type(str(payload.get("user_input", ""))))
    logic_skill = str(payload.get("logic_skill") or infer_logic_skill(str(payload.get("user_input", "")), task_type))

    sample = PretrainSample(
        user_input=str(payload.get("user_input", "")).strip(),
        task_type=task_type,
        source_type="human_review_patch",
        expected_tool=str(payload.get("override_expected_tool") or payload.get("expected_tool") or "search"),
        target_slot=str(payload.get("override_target_slot") or payload.get("target_slot") or "PLAN"),
        logic_skill=logic_skill,
        plan_summary=str(payload.get("override_plan_summary") or payload.get("plan_summary") or ""),
        plan_action_items=[str(item) for item in action_items if str(item).strip()],
        plan_unknowns=[str(item) for item in unknowns if str(item).strip()],
        evidence_targets=[str(item) for item in evidence_targets if str(item).strip()],
        draft_summary=str(payload.get("override_draft_summary") or payload.get("draft_summary") or ""),
        quality_score=float(payload.get("quality_score", 0.0) or 0.0),
        metadata={"review_note": str(payload.get("review_note", ""))},
    )
    if not sample.plan_summary:
        sample.plan_summary = f"围绕任务 `{sample.task_type}` 建立求解路径。"
    if not sample.draft_summary:
        sample.draft_summary = "基于人工修订后的计划形成结论。"
    return sample


def sample_to_episode(sample: PretrainSample) -> EpisodeRecord:
    plan_payload = {
        "kind": "plan",
        "summary": sample.plan_summary,
        "action_items": sample.plan_action_items,
        "unknowns": sample.plan_unknowns,
        "evidence_targets": sample.evidence_targets,
        "keywords": [],
        "confidence_band": "medium",
    }
    draft_payload = {
        "kind": "draft",
        "summary": sample.draft_summary,
        "support_items": sample.evidence_targets[:2],
        "open_risks": sample.plan_unknowns[:2],
        "confidence_band": "medium" if sample.quality_score < 0.88 else "high",
    }
    steps = build_synthetic_steps(sample)

    summary = " | ".join(
        [
            json.dumps(plan_payload, ensure_ascii=False),
            f"tool={sample.expected_tool}",
            json.dumps(draft_payload, ensure_ascii=False),
        ]
    )
    return EpisodeRecord(
        user_input=sample.user_input,
        answer=sample.draft_summary,
        summary=summary,
        success=True,
        value_score=max(0.45, min(sample.quality_score, 0.98)),
        episode_features={
            "source_type": sample.source_type,
            "task_type": sample.task_type,
            "logic_skill": sample.logic_skill,
            "expected_tool": sample.expected_tool,
            "plan_summary": sample.plan_summary,
            "plan_action_items": sample.plan_action_items,
            "plan_unknowns": sample.plan_unknowns,
            "evidence_targets": sample.evidence_targets,
            "draft_summary": sample.draft_summary,
            "quality_score": sample.quality_score,
        },
        outcome_signature={
            "success": True,
            "value_score": max(0.45, min(sample.quality_score, 0.98)),
            "confidence_band": "medium" if sample.quality_score < 0.88 else "high",
            "used_external_result": sample.expected_tool != "",
        },
        steps=steps,
    )


def infer_task_type(text: str) -> str:
    lower = text.lower()
    if any(token in text for token in ("比较", "对比", "区别", "优缺点")) or " vs " in lower:
        return "compare"
    if any(token in text for token in ("计算", "多少", "预算", "总成本", "还剩")):
        return "calculate"
    if any(token in lower for token in ("python", "script", "api", "json", "bug", "debug")) or "代码" in text or "脚本" in text:
        return "coding"
    if any(token in text for token in ("计划", "规划", "方案", "流程", "步骤")):
        return "planning"
    if any(token in text for token in ("总结", "摘要", "纪要", "提炼")):
        return "summarize"
    if any(token in text for token in ("核验", "验证", "可靠", "冲突", "站得住")):
        return "fact_check"
    return "planning"


def infer_logic_skill(text: str, task_type: str = "") -> str:
    lower = text.lower()
    if any(token in text for token in ("比较", "对比", "维度")):
        return "comparison"
    if any(token in text for token in ("约束", "依赖", "里程碑")):
        return "constraint_planning"
    if any(token in text for token in ("证据", "来源", "核验", "站得住")):
        return "evidence_judgment"
    if any(token in text for token in ("冲突", "分歧")):
        return "conflict_detection"
    if any(token in text for token in ("哪类工具", "优先调用", "工具")):
        return "tool_selection"
    if any(token in text for token in ("整合", "汇总", "带风险说明")):
        return "result_integration"
    if any(token in text for token in ("什么时候应该停止", "停止继续", "收束")):
        return "termination_judgment"
    if any(token in text for token in ("步骤", "分步骤", "执行计划", "列出来")):
        return "step_planning"
    if any(token in text for token in ("判断它是", "先判断")):
        return "classification"
    if task_type == "compare":
        return "comparison"
    if task_type == "planning":
        return "step_planning"
    if task_type == "fact_check":
        return "evidence_judgment"
    if task_type == "calculate":
        return "tool_selection"
    return "tool_selection"


def normalize_user_input(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^\w\u4e00-\u9fff ]+", "", lowered)
    return lowered


def build_synthetic_steps(sample: PretrainSample) -> list[StepRecord]:
    steps: list[StepRecord] = [
        StepRecord(
            step_idx=1,
            action=Action.WRITE_MEM.value,
            reason="Persist the task goal into working memory.",
            score=sample.quality_score,
            feature_snapshot={"bias": 1.0, "need_structure": 1.0, "answer_ready": 0.05},
            user_input=sample.user_input,
            target_slot="GOAL",
            reward=0.1,
            reward_reason="synthetic_goal_bootstrap",
            high_value=False,
        ),
        StepRecord(
            step_idx=2,
            action=Action.WRITE_MEM.value,
            reason=f"Create the primary reasoning scaffold in {sample.target_slot}.",
            score=sample.quality_score,
            feature_snapshot={"bias": 1.0, "need_structure": 1.0, "answer_ready": 0.1},
            user_input=sample.user_input,
            target_slot=sample.target_slot,
            reward=0.45,
            reward_reason="synthetic_plan_bootstrap",
            high_value=True,
        ),
        StepRecord(
            step_idx=3,
            action=Action.CALL_TOOL.value,
            reason=f"Prefer {sample.expected_tool} for this task type.",
            score=sample.quality_score,
            feature_snapshot={"bias": 1.0},
            user_input=sample.user_input,
            selected_tool=sample.expected_tool,
            reward=1.35,
            reward_reason="synthetic_tool_bootstrap",
            high_value=True,
        ),
        StepRecord(
            step_idx=4,
            action=Action.WRITE_MEM.value,
            reason="Materialize a draft from the gathered evidence.",
            score=sample.quality_score,
            feature_snapshot={"bias": 1.0, "answer_ready": 0.7, "has_result": 1.0},
            user_input=sample.user_input,
            target_slot="DRAFT",
            reward=0.2,
            reward_reason="synthetic_draft_bootstrap",
            high_value=False,
        ),
    ]

    if sample.task_type == "calculate":
        steps.append(
            StepRecord(
                step_idx=5,
                action=Action.VERIFY.value,
                reason="Double-check the numeric result before answering.",
                score=sample.quality_score,
                feature_snapshot={"bias": 1.0, "answer_ready": 0.8},
                user_input=sample.user_input,
                reward=0.2,
                reward_reason="synthetic_numeric_verify",
                high_value=False,
            )
        )
        answer_step_idx = 6
    else:
        answer_step_idx = 5

    steps.append(
        StepRecord(
            step_idx=answer_step_idx,
            action=Action.ANSWER.value,
            reason="Finish from a structured draft and tool-backed evidence.",
            score=sample.quality_score,
            feature_snapshot={"bias": 1.0, "answer_ready": 0.95, "has_result": 1.0, "has_draft": 1.0},
            user_input=sample.user_input,
            reward=1.0,
            reward_reason="synthetic_answer_bootstrap",
            high_value=True,
        )
    )
    return steps


def _generate_compare_samples(rng: random.Random, count: int) -> list[PretrainSample]:
    prompts: list[PretrainSample] = []
    focuses = ["适用性", "成本", "维护复杂度", "生态成熟度", "性能表现"]
    for _ in range(count):
        left, right, scene = rng.choice(COMPARE_ITEMS)
        focus = rng.choice(focuses)
        prompt = _vary_prompt(rng, f"比较 {left} 和 {right} 在 {scene} 里的 {focus}")
        prompts.append(
            PretrainSample(
                user_input=prompt,
                task_type="compare",
                source_type="template_compare",
                expected_tool="",
                target_slot="",
                logic_skill="comparison",
            )
        )
    return prompts


def _generate_simple_samples(rng: random.Random, base_items: list[str], task_type: str, count: int) -> list[PretrainSample]:
    prompts: list[PretrainSample] = []
    for index in range(count):
        prompt = _vary_prompt(rng, base_items[index % len(base_items)])
        prompts.append(
            PretrainSample(
                user_input=prompt,
                task_type=task_type,
                source_type=f"template_{task_type}",
                expected_tool="",
                target_slot="",
                logic_skill=infer_logic_skill(prompt, task_type),
            )
        )
    return prompts


def _generate_logic_primitive_samples(rng: random.Random, count: int) -> list[PretrainSample]:
    prompts: list[PretrainSample] = []
    for index in range(count):
        prompt, task_type, logic_skill, expected_tool, target_slot = LOGIC_PRIMITIVE_ITEMS[index % len(LOGIC_PRIMITIVE_ITEMS)]
        prompts.append(
            PretrainSample(
                user_input=_vary_prompt(rng, prompt),
                task_type=task_type,
                source_type=f"logic_{logic_skill}",
                expected_tool="",
                target_slot="",
                logic_skill=logic_skill,
                metadata={
                    "forced_expected_tool": expected_tool,
                    "forced_target_slot": target_slot,
                },
            )
        )
    return prompts


def _generate_logic_hard_case_samples(rng: random.Random, count: int) -> list[PretrainSample]:
    prompts: list[PretrainSample] = []
    for index in range(count):
        prompt, task_type, logic_skill, expected_tool, target_slot = LOGIC_HARD_CASE_ITEMS[index % len(LOGIC_HARD_CASE_ITEMS)]
        prompts.append(
            PretrainSample(
                user_input=_vary_prompt(rng, prompt),
                task_type=task_type,
                source_type=f"logic_hard_{logic_skill}",
                expected_tool="",
                target_slot="",
                logic_skill=logic_skill,
                metadata={
                    "forced_expected_tool": expected_tool,
                    "forced_target_slot": target_slot,
                    "hard_case": True,
                },
            )
        )
    return prompts


def _import_jsonl_tasks(path: Path) -> list[PretrainSample]:
    samples: list[PretrainSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            prompt = str(payload.get("prompt") or payload.get("question") or payload.get("instruction") or payload.get("input") or "").strip()
            if not prompt:
                continue
            task_type = infer_task_type(prompt)
            samples.append(
                PretrainSample(
                    user_input=prompt,
                    task_type=task_type,
                    source_type=f"import_jsonl:{path.name}",
                    expected_tool="",
                    target_slot="",
                    logic_skill=infer_logic_skill(prompt, task_type),
                    metadata={"path": str(path)},
                )
            )
    return samples


def _import_json_tasks(path: Path) -> list[PretrainSample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else payload.get("items", []) if isinstance(payload, dict) else []
    samples: list[PretrainSample] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt") or item.get("question") or item.get("instruction") or item.get("input") or "").strip()
        if not prompt:
            continue
        task_type = infer_task_type(prompt)
        samples.append(
            PretrainSample(
                user_input=prompt,
                task_type=task_type,
                source_type=f"import_json:{path.name}",
                expected_tool="",
                target_slot="",
                logic_skill=infer_logic_skill(prompt, task_type),
                metadata={"path": str(path)},
            )
        )
    return samples


def _import_text_tasks(path: Path) -> list[PretrainSample]:
    samples: list[PretrainSample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        prompt = line.strip(" -*\t")
        if len(prompt) < 8:
            continue
        task_type = infer_task_type(prompt)
        samples.append(
            PretrainSample(
                user_input=prompt,
                task_type=task_type,
                source_type=f"import_text:{path.name}",
                expected_tool="",
                target_slot="",
                logic_skill=infer_logic_skill(prompt, task_type),
                metadata={"path": str(path)},
            )
        )
    return samples


def _vary_prompt(rng: random.Random, prompt: str) -> str:
    prefix = rng.choice(PROMPT_PREFIXES)
    suffix = rng.choice(PROMPT_SUFFIXES)
    return f"{prefix}{prompt}{suffix}".strip()


def _has_diverse_signal(text: str) -> bool:
    has_ascii = bool(re.search(r"[a-zA-Z]", text))
    has_han = bool(re.search(r"[\u4e00-\u9fff]", text))
    has_digits = bool(re.search(r"\d", text))
    return sum(1 for flag in (has_ascii, has_han, has_digits) if flag) >= 2
