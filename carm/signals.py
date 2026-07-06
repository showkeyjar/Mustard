"""Shared signal detection utilities used across CARM modules.

Centralizes conflict detection, tokenization, and intent signal extraction
that were previously duplicated across core.py, policy.py, concepts.py,
and evolution.py.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import NamedTuple

from carm.intent import IntentCategory


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

CONFLICT_MARKERS = (
    "冲突",
    "相反建议",
    "相反",
    "矛盾",
    "不一致",
    "先怎么处理冲突",
    "还没消解前",
    "直接下结论吗",
)


def is_conflict_task(text: str) -> bool:
    """Return True if the text contains conflict-related markers."""
    return any(marker in text for marker in CONFLICT_MARKERS)


# ---------------------------------------------------------------------------
# Intent signals
# ---------------------------------------------------------------------------

COMPARE_TOKENS = (
    "比较",
    "区别",
    "优缺点",
    "vs",
    "对比",
    "分别适用",
    "各自适用",
    "分别",
)
CALC_TOKENS = (
    "多少",
    "计算",
    "cost",
    "price",
    "sum",
    "数字",
    "预算",
    "总价",
    "每席位",
    "按年",
    "每月",
    "分几批",
    "平方根",
    "开方",
    "开根号",
    "平方",
    "次方",
    "乘以",
    "除以",
    # Arithmetic operation words
    "加上",
    "加",
    "减去",
    "减",
    "乘",
    "除",
    "等于",
    # Chinese large number units
    "万亿",
    "亿",
    "万",
    "千万",
    "百万",
    "十万",
    # Geometry calculation keywords
    "面积",
    "周长",
    "体积",
    "半径",
    "直径",
    # Classic Chinese math puzzle keywords
    "鸡兔同笼",
    "鸡",
    "兔",
    "头",
    "腿",
    "脚",
    # Unit conversion keywords
    "公里",
    "千米",
    "小时",
    "等于多少",
)
CODE_TOKENS_EN = (
    "python",
    "code",
    "script",
    "crawler",
    "spider",
    "scrape",
    "scraping",
    "hack",
    "debug",
)
CODE_TOKENS_ZH = ("代码", "脚本", "报错", "爬虫", "抓取", "抓数据", "数据采集", "采集")
CODE_ACTION_TOKENS = (
    "写",
    "实现",
    "编写",
    "开发",
    "运行",
    "执行",
    "跑一下",
    "跑一遍",
    "再跑",
    "函数",
    "方法",
    "算法",
    "排序",
    "查找",
    "遍历",
    "反转",
    "去重",
    "合并",
    "递归",
    "斐波那契",
    "二分",
    "阶乘",
    # Real-world code actions
    "爬",
    "抓",
    "画图",
    "绘图",
    "可视化",
    "画个图",
    "画出来",
    "跑个",
    "跑一下",
)

# Tokens that indicate a WRITING/SYNTHESIS intent, not a code intent.
# When these are present, "写" should be interpreted as writing/essay,
# not code execution.
WRITING_TOKENS = (
    "作文",
    "议论文",
    "文章",
    "读后感",
    "小说",
    "故事",
    "散文",
    "诗歌",
    "词",
    "赋",
    "读后感",
)

# Tokens that indicate a WRITING/SYNTHESIS intent (essay generation, summarization, etc.)
WRITING_ACTION_TOKENS = (
    "生成",
    "创作",
    "撰写",
    "写一份",
    "写一篇",
    "写一篇关于",
    "写一下",
    "总结",
    "归纳",
    "概括",
    "写首诗",
    "写封信",
    "写一封",
    "写个总结",
    "写个摘要",
    "写个读书",
    "写个笔记",
    "写篇读书",
    "写篇笔记",
    "起草",
    "拟一份",
    "写一份",
    "生成一份",
)

# Search-intent tokens that indicate the user is looking for information
SEARCH_TOKENS = (
    "是什么",
    "什么是",
    "有什么",
    "怎么",
    "怎么样",
    "如何",
    "在哪里",
    "去哪儿",
    "什么地方",
    "推荐",
    "介绍",
    "资料",
    "信息",
    "新闻",
    "最新",
    "排名",
    "评分",
    "评价",
    "口碑",
    "排行",
    "哪一",
    "哪个",
    "哪家",
    "哪几",
    "哪些",
    "哪种",
    "哪款",
    "哪部",
    "哪个好",
    "谁更",
    "谁最",
    # Real-world: tech metrics and status queries
    "指标",
    "监控",
    "性能",
    "参数",
    "状态",
    "利用率",
    "容量",
    "配置",
    "版本",
    "日志",
    "报错",
    "错误",
    "异常",
    "告警",
    "场景",
    "适用",
)

# Tokens that indicate a SEARCH action (not just a SEARCH question)
# NOTE: Single-char "搜"/"查" are too broad ("热搜"=trending, "查bug"=debug).
# Use compound forms that clearly indicate a search ACTION.
SEARCH_ACTION_TOKENS = (
    "搜索",
    "查找",
    "查询",
    "检索",
    "搜一下",
    "查一下",
    "找一找",
    "找一下",
    "搜搜",
    "搜搜看",
    "查查看",
)

# Search-related URLs/domains
SEARCH_URLS = (
    "http",
    "www",
    ".com",
    ".cn",
    ".org",
    ".net",
    ".io",
)

# English tech-metric terms that signal a SEARCH intent (status/metrics lookup)
TECH_METRIC_TOKENS = (
    "latency",
    "qps",
    "tps",
    "p99",
    "p95",
    "p90",
    "gmv",
    "dau",
    "mau",
    "cpu",
    "gpu",
    "memory",
    "ram",
    "disk",
    "throughput",
    "uptime",
    "availability",
    "bandwidth",
    "iops",
    "rps",
    "error_rate",
    "timeout",
    "slow_query",
    "benchmark",
    "fps",
    "rpm",
    "concurrency",
)

FORMAL_TOKENS = ("正式", "优雅", "自然", "对话", "口语", "书面")

# Tokens that indicate the user wants to EXPLAIN or UNDERSTAND something
EXPLAIN_TOKENS = (
    "为什么",
    "解释",
    "说明",
    "原因",
    "原理",
    "机制",
    "逻辑",
    "如何理解",
    "怎么理解",
    "含义",
    "意义",
    "解读",
    "分析",
)

# Tokens that indicate a CONSULTATIVE answer
# (e.g., "帮我分析一下这个方案", "看看这个代码有什么问题")
# NOTE: "帮我" alone is very broad — only triggers when combined with
# advisory verbs.  "帮我订个外卖" has no advisory verb, so consult
# should NOT fire.
CONSULT_TOKENS = (
    "帮我分析",
    "帮我评估",
    "帮我优化",
    "帮我检查",
    "帮我建议",
    "帮我选择",
    "帮我比较",
    "帮我对比",
    "帮我查",
    "帮我找",
    "帮我搜",
    "看看",
    "分析",
    "评估",
    "建议",
    "意见",
    "点评",
    "检查",
    "诊断",
    "纠错",
    "优化",
    "对比",
    # Advisory/selection signals — "应该用哪种" is seeking advice, not execution
    "应该",
    "应该选",
    "应该用",
    "适合",
    "推荐用",
)

# Tokens that signal DEEP ANALYSIS intent — the user wants LLM-powered
# synthesis/reasoning, not just a search lookup.  When these co-occur with
# consult tokens, the intent is bigmodel_proxy, not search.
DEEP_ANALYSIS_TOKENS = (
    "可行性",
    "方案",
    "策略",
    "规划",
    "报告",
    "提案",
    "计划",
    "决策",
    "风险",
    "影响",
    "趋势",
    "前景",
    "展望",
)

# Tokens that indicate a DEBUG CONSULTATIVE intent -- the user wants help
# understanding or fixing code, NOT executing it. When these appear alongside
# code tokens, the intent is advisory (search for solutions), not execution.
DEBUG_CONSULT_TOKENS = (
    "怎么解决",
    "怎么办",
    "如何解决",
    "怎么修",
    "如何修复",
    "什么问题",
    "什么原因",
    "为什么报错",
    "出错了怎么办",
    "出了什么问题",
    "怎么排查",
    "如何排查",
    "怎么处理",
    "如何处理",
    "如何调试",
)

# Pattern for deep reasoning/comparative analysis queries.
# "为什么...而..." / "为什么A而B不需要" — these are open-ended reasoning
# questions that need LLM synthesis, not just search.
_DEEP_REASON_PATTERN = re.compile(r"为什么.{2,20}而.{2,20}")

# Tokens that indicate a TRANSLATION intent
# NOTE: "把" alone is NOT a translate token — "把CSV画个图" is code, not translation.
# "把" is only a translate signal when followed by "成/译" pattern.
TRANSLATE_TOKENS = (
    "翻译",
    "译成",
    "译",
    "翻成",
    "翻译成",
    "译成",
    "翻译一下",
    "翻译这段话",
    "翻译这段",
    "翻一下",
    "翻译这段文字",
    "翻译这句话",
    "翻译成中文",
    "翻译成英文",
)

# Pattern: "把...成/译" → translation intent (e.g. "把hello翻成中文")
_TRANSLATE_PATTERN = re.compile(r"把.{1,30}(成|译|翻译|翻成)")


def has_translate_signal(text: str) -> bool:
    """Return True if the text contains translation intent signals."""
    if any(token in text for token in TRANSLATE_TOKENS):
        return True
    # "把...成/译" pattern without explicit "翻译" keyword
    if _TRANSLATE_PATTERN.search(text):
        return True
    return False


# Tokens that indicate a TEXT POLISHING intent
POLISH_TOKENS = (
    "润色",
    "修改",
    "改写",
    "改一下",
    "改写一下",
    "改写成",
    "优化一下",
    "优化这段",
    "优化这段文字",
    "优化这句话",
    "改个说法",
    "换个说法",
    "换个表达",
    "换个表达方式",
    "换个表达方式",
    "改写成",
    "改写为",
    "改写一下",
)

# Tokens that indicate anaphoric (context-dependent) references
# These words refer to entities from previous conversation turns
ANAPHORA_TOKENS = (
    "它",
    "这",
    "这个",
    "那个",
    "那篇",
    "刚才",
    "上次",
    "之前",
    "之前查的",
    "刚才查的",
    "上次查的",
    "那个东西",
    "那个问题",
    "那个结果",
)

# Tokens that indicate a QUESTION or inquiry
QUESTION_TOKENS = (
    "?",
    "？",
    "吗",
    "呢",
    "吧",
    "么",
    "如何",
    "怎么",
    "为什么",
    "是什么",
    "多少",
    "哪里",
    "什么时候",
    "谁",
    "什么",
    "哪里",
    "哪个",
    "哪些",
    "哪种",
    "怎样",
    "怎么样",
)

# Tokens for comparison evidence gathering
COMPARISON_EVIDENCE_TOKENS = (
    "数据",
    "统计",
    "来源",
    "出处",
    "证据",
    "事实",
    "依据",
    "根据",
    "排名",
    "排行",
    "评分",
    "评价",
    "口碑",
)

# ---------------------------------------------------------------------------
# Travel / lifestyle tokens
# ---------------------------------------------------------------------------

TRAVEL_TOKENS = (
    "机票",
    "航班",
    "飞机",
    "起飞",
    "降落",
    "登机",
    "值机",
    "航空公司",
    "机票预订",
    "订票",
    "购票",
    "买票",
    "高铁",
    "动车",
    "火车票",
    "列车",
    "车次",
    "软卧",
    "硬卧",
    "二等座",
    "一等座",
    "商务座",
    "酒店",
    "住宿",
    "预订酒店",
    "宾馆",
    "民宿",
    "客栈",
    "天气",
    "天气预报",
    "气温",
    "降雨",
    "下雪",
    "台风",
    "气候",
    "地图",
    "导航",
    "路线",
    "怎么走",
    "怎么去",
    "出行",
    "旅游",
    "旅行",
    "度假",
    "景点",
    "门票",
    "景区",
    "景点推荐",
    "必去景点",
    "附近有什么",
    "周边游",
    "一日游",
    "两日游",
    "三日游",
    "攻略",
    "旅游攻略",
    "旅行攻略",
)

# ---------------------------------------------------------------------------
# Date exclusion list (not an arithmetic question)
# ---------------------------------------------------------------------------

DATE_EXCLUSIONS = (
    "日期",
    "今天",
    "明天",
    "后天",
    "昨天",
    "前天",
    "几号",
    "星期几",
    "周几",
    "几月",
    "什么时候",
    "几点",
    "哪一年",
    "哪个月",
    "日期查询",
)

# ---------------------------------------------------------------------------
# Anaphora detection
# ---------------------------------------------------------------------------


def has_anaphora_signal(text: str) -> bool:
    """Return True if the text contains anaphoric references (指代).

    These references (它/这/刚才/上次) indicate the user is referring to
    entities or results from previous conversation turns.
    """
    return any(token in text for token in ANAPHORA_TOKENS)


# ---------------------------------------------------------------------------
# Debug consult / deep reason
# ---------------------------------------------------------------------------


def has_debug_consult_signal(text: str) -> bool:
    """Return True if the text indicates a debug consultative intent.

    The user wants HELP with code (understanding errors, finding solutions),
    not to EXECUTE code.  For example:
    - "代码报错了怎么解决" → True (seeking help)
    - "帮我写一个排序" → False (execution intent)
    """
    return any(token in text for token in DEBUG_CONSULT_TOKENS)


def has_deep_reason_signal(text: str) -> bool:
    """Return True if the text requires deep reasoning/comparative analysis.

    Matches patterns like "为什么...而..." that indicate the user wants
    a thoughtful comparison or causal explanation — best handled by bigmodel_proxy.
    """
    return bool(_DEEP_REASON_PATTERN.search(text))


# ---------------------------------------------------------------------------
# Travel
# ---------------------------------------------------------------------------


def has_travel_signal(text: str) -> bool:
    """Return True if the text contains travel/lifestyle intent signals."""
    return any(token in text for token in TRAVEL_TOKENS)


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Tokenize text into words (Chinese chars or English words)."""
    tokens = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        if "\u4e00" <= text[i] <= "\u9fff":
            tokens.append(text[i])
            i += 1
        else:
            j = i
            while (
                j < len(text)
                and not text[j].isspace()
                and not ("\u4e00" <= text[j] <= "\u9fff")
            ):
                j += 1
            tokens.append(text[i:j])
            i = j
    return tokens


def word_count(text: str) -> int:
    """Return the number of words/tokens in the text."""
    return len(tokenize(text))


def token_counts(text: str) -> Counter:
    """Return a Counter mapping each token to its frequency."""
    return Counter(tokenize(text))


# ---------------------------------------------------------------------------
# Signal detection helpers
# ---------------------------------------------------------------------------


def has_compare_signal(text: str) -> bool:
    """Return True if the text contains comparison keywords."""
    return any(token in text for token in COMPARE_TOKENS)


def has_calc_signal(text: str) -> bool:
    """Return True if the text contains calculation intent signals.

    This includes:
    - Direct arithmetic words (多少, 计算, 等于多少)
    - Large number units (万亿, 亿, 万)
    - Geometry terms (面积, 周长, 体积)
    - Unit conversion terms (公里, 千米, 小时)

    BUT: if the query contains NO digits at all, it's likely a knowledge
    question ("一年有多少天") rather than a calculation. Calculation
    requires at least one numeric operand.
    """
    # Exclude pure date queries
    if any(exclusion in text for exclusion in DATE_EXCLUSIONS):
        return False
    # No digits = no calculation possible ("一年有多少天" is knowledge, not calc)
    # BUT: classic Chinese math puzzles (鸡兔同笼) have numbers implied by the puzzle
    # structure — the "10个头" and "26条腿" are implicit operands.  Treat these
    # puzzles as calculation even without explicit digits in the query text itself.
    if not re.search(r"\d", text):
        if "鸡兔同笼" in text or ("头" in text and "腿" in text and "鸡" in text):
            pass  # Classic puzzle — treat as calculation
        else:
            return False
    return any(token in text for token in CALC_TOKENS)


def has_code_signal(text: str) -> bool:
    """Return True if the text contains code execution intent signals.

    Covers English code keywords (python, code, script) and Chinese code
    terms (代码, 脚本, 报错) as well as code action verbs (写, 实现,
    运行, 执行, etc.).

    However, debug consultative intent ("怎么解决/什么问题") overrides
    code signals — "代码报错了怎么解决" is seeking help, not execution.
    """
    has_en = any(token in text.lower() for token in CODE_TOKENS_EN)
    has_zh = any(token in text for token in CODE_TOKENS_ZH)
    has_action = any(token in text for token in CODE_ACTION_TOKENS)
    has_consult = any(token in text for token in CONSULT_TOKENS)

    # Writing tokens override code action ("写一篇作文" is writing, not code)
    if any(token in text for token in WRITING_TOKENS) and "写" in text:
        has_action = False

    # Writing action tokens ("生成一份报告") override code
    if any(token in text for token in WRITING_ACTION_TOKENS):
        has_action = False

    # Debug consultative intent ("怎么解决"/"什么问题") overrides code signals
    # "代码报错了怎么解决" is seeking help, not execution — route to search.
    has_debug_consult = has_debug_consult_signal(text)
    if has_debug_consult and not any(
        v in text for v in ("运行", "写", "实现", "编写", "执行", "跑一下")
    ):
        return False

    # Consultative/advisory intent ("优化/分析/选择") overrides algorithm signals
    # "如何选择排序算法" is advisory, not execution — but "写一个排序" is still code.
    # Consult only overrides when there is NO strong code action verb ("运行/写/实现/执行").
    _strong_code_verbs = ("运行", "写", "实现", "编写", "执行", "跑一下")
    has_strong_code_verb = any(v in text for v in _strong_code_verbs)
    if has_consult and not has_strong_code_verb:
        return False

    # Return True if we have at least one code keyword AND an action verb,
    # OR if we have a strong English code keyword.
    # BUT: "Python的GIL是什么" — "Python" is a code keyword, but "是什么" is
    # a knowledge/explain signal with no code action. Treat as knowledge, not code.
    if has_en:
        if "是什么" in text or "为什么" in text:
            if not has_action and not has_strong_code_verb:
                return False
        return True
    # "Python的GIL是什么" — 虽然"Python"是code token，但"是什么"是explain/search信号
    # 没有代码动作动词时，视为知识查询而非代码执行
    if has_zh and not has_action:
        return False
    if has_zh and has_action:
        return True
    if has_action and any(
        kw in text for kw in ("python", "py", "java", "js", "c++", "cpp", "go", "rust")
    ):
        return True
    return False


def has_formal_signal(text: str) -> bool:
    """Return True if the text contains formal style intent signals."""
    return any(token in text for token in FORMAL_TOKENS)


def has_explain_signal(text: str) -> bool:
    """Return True if the text contains explain/understand intent signals."""
    return any(token in text for token in EXPLAIN_TOKENS)


def has_search_action_signal(text: str) -> bool:
    """Return True if the text contains explicit search action keywords."""
    return any(token in text for token in SEARCH_ACTION_TOKENS)


def has_search_signal(text: str) -> bool:
    """Return True if the text contains search intent signals."""
    # Travel signal also counts as search
    # English tech metrics also count as search (status/metrics lookup)
    # Date-related knowledge questions ("一年有多少天") are search, not calc
    has_basic = (
        has_travel_signal(text)
        or has_search_action_signal(text)
        or any(token in text for token in SEARCH_TOKENS)
        or any(token in text.lower() for token in TECH_METRIC_TOKENS)
    )
    if has_basic:
        return True
    # "多少" + time/date word = knowledge question, not calculation
    _time_quantity_words = ("天", "年", "周", "月", "日", "季度", "世纪")
    if "多少" in text and any(w in text for w in _time_quantity_words):
        return True
    # Date exclusion words + question pattern = knowledge search
    if any(ex in text for ex in DATE_EXCLUSIONS) and (
        "多少" in text or "什么" in text or "几" in text
    ):
        return True
    return False


def has_comparison_evidence_signal(text: str) -> bool:
    """Return True if the text contains comparison evidence gathering signals."""
    return any(token in text for token in COMPARISON_EVIDENCE_TOKENS)


def has_writing_signal(text: str) -> bool:
    """Return True if the text contains writing/synthesis intent signals.

    Includes WRITING_TOKENS (essay types) and WRITING_ACTION_TOKENS (generate,
    create, write a report, etc.).
    """
    return any(token in text for token in WRITING_TOKENS + WRITING_ACTION_TOKENS)


def has_translate_signal(text: str) -> bool:
    """Return True if the text contains translation intent signals."""
    return any(token in text for token in TRANSLATE_TOKENS)


def has_polish_signal(text: str) -> bool:
    """Return True if the text contains text polishing intent signals."""
    return any(token in text for token in POLISH_TOKENS)


def has_consult_signal(text: str) -> bool:
    """Return True if the text contains consultative intent signals."""
    return any(token in text for token in CONSULT_TOKENS)


def has_deep_analysis_signal(text: str) -> bool:
    """Return True if the text contains deep analysis intent signals.

    When consult tokens co-occur with deep analysis tokens (可行性/方案/策略...),
    the intent is LLM-powered synthesis, not a simple search lookup.
    """
    return any(token in text for token in DEEP_ANALYSIS_TOKENS)


# ---------------------------------------------------------------------------
# Low-intent / no-intent detection
# ---------------------------------------------------------------------------

# Fillers and noise words that alone carry zero actionable intent
_LOW_INTENT_FILLERS = frozenset(
    {
        "嗯",
        "啊",
        "哦",
        "额",
        "呃",
        "唉",
        "诶",
        "哈",
        "嘿",
        "呀",
        "好",
        "好的",
        "行",
        "对",
        "是",
        "嗯嗯",
        "噢",
        "喔",
        "OK",
        "ok",
        "yes",
        "no",
        "nope",
        "nah",
    }
)

# Patterns that look intent-like but are actually too vague to route
_LOW_INTENT_PATTERNS = (
    re.compile(r"^.{0,4}$"),  # ≤4 chars total
    re.compile(r"^(帮我)?看看$"),  # just "看看" or "帮我看看"
    re.compile(r"^(帮我)?弄(一下|一弄)?$"),  # just "帮我弄一下"
    re.compile(r"^(不是|不要|别)(这个|那个|这样|那样)$"),  # "不是那个"
    re.compile(r"^(太|好|真|好|超).{0,4}(了|啊|吧)$"),  # pure emotion "太慢了"
    re.compile(r"^(能不能|可以|可否|能否).{0,6}$"),  # bare "能不能快点"
)


def has_low_intent_signal(text: str) -> bool:
    """Return True if the text has no actionable intent.

    These queries should NOT be routed to any tool — they are filler,
    pure emotion, or too vague to act on. The router should return a
    "no tool" result and prompt the user to clarify.

    Detection strategy:
      1. Very short text (≤4 chars) with no strong signal keywords
      2. Pure filler/emotion words
      3. Specific low-intent patterns (帮我看看, 太慢了, 不是那个)
      4. Text that triggers no signal at all (no calc/code/search/consult)
    """
    stripped = text.strip()

    # Rescue: if ANY strong action verb is present, it's NOT low-intent
    # "算一下" / "搞不定" / "画个图" / "订个" / "查一下" etc.
    _strong_action_verbs = (
        "算",
        "计算",
        "查",
        "搜索",
        "搜",
        "写",
        "编",
        "画",
        "跑",
        "执行",
        "运行",
        "实现",
        "开发",
        "翻译",
        "润色",
        "分析",
        "画图",
        "绘图",
        "爬",
        "抓",
        "订",
        "买",
        "找",
        "问",
        "搞",
        "弄",
        "做",
        "调",
        "改",
        "删",
        "加",
        "减",
        "设置",
        "配置",
        "安装",
        "部署",
        "启动",
        "停止",
        "重启",
    )
    if any(v in stripped for v in _strong_action_verbs):
        return False

    # Rescue: if any tech metric keyword is present, it's NOT low-intent
    if any(t in stripped.lower() for t in TECH_METRIC_TOKENS):
        return False

    # Rescue: if any CODE_TOKENS are present, it's NOT low-intent
    if any(t in stripped.lower() for t in CODE_TOKENS_EN):
        return False

    # Rule 1: Very short + no strong signal
    if len(stripped) <= 4:
        if stripped in _LOW_INTENT_FILLERS:
            return True
        # Short but has a real signal? Keep it.
        if not any(
            f(stripped)
            for f in (
                has_calc_signal,
                has_code_signal,
                has_search_action_signal,
                has_travel_signal,
                has_translate_signal,
            )
        ):
            # Check tech metrics too
            if not any(t in stripped.lower() for t in TECH_METRIC_TOKENS):
                return True

    # Rule 2: Pure filler/emotion
    if stripped in _LOW_INTENT_FILLERS:
        return True

    # Rule 3: Known low-intent patterns
    for pat in _LOW_INTENT_PATTERNS:
        if pat.search(stripped):
            return True

    # Rule 4: No signal at all — the query triggers zero intent detectors
    has_any_signal = any(
        f(stripped)
        for f in (
            has_calc_signal,
            has_code_signal,
            has_search_signal,
            has_consult_signal,
            has_writing_signal,
            has_translate_signal,
            has_polish_signal,
            has_compare_signal,
            has_explain_signal,
            has_travel_signal,
            has_deep_analysis_signal,
            has_deep_reason_signal,
        )
    )
    if not has_any_signal:
        # Extra check: English tech metrics and code tokens
        if not any(t in stripped.lower() for t in TECH_METRIC_TOKENS):
            if not any(t in stripped.lower() for t in CODE_TOKENS_EN):
                return True

    return False


# Tokens that signal MULTI-STEP reasoning — a single intent that requires
# multiple tools executed in sequence (e.g., search → compare → synthesize).
MULTI_STEP_TOKENS = (
    "对比分析",
    "比较分析",
    "对比并",
    "比较并",
    "分析并给出",
    "对比给出",
    "分析总结",
    "分析归纳",
    "调研并",
    "研究并",
)


def has_multi_step_signal(text: str) -> bool:
    """Return True if text contains multi-step reasoning signals.

    Multi-step queries have a single intent but require sequential tool
    execution (e.g., gather evidence → compare → synthesize).
    These differ from multi-intent queries which have multiple independent
    intents joined by connectors ("顺便", "然后").
    """
    return any(token in text for token in MULTI_STEP_TOKENS)


# ---------------------------------------------------------------------------
# Multi-intent detection
# ---------------------------------------------------------------------------

MULTI_INTENT_CONNECTORS = (
    "顺便",
    "然后",
    "同时",
    "另外",
    "并且",
    "以及",
    "再",
    "接着",
    "之后",
    "先",
    "先搜索",
    "先查",
)

MULTI_INTENT_DELIMITERS = (",", "，", ";", "；", "|")


class SplitIntent(NamedTuple):
    """One split sub-intent from a multi-intent query."""

    text: str
    primary_signal: (
        IntentCategory | str
    )  # IntentCategory for known signals, str for compat
    priority: int  # lower = should run first (e.g. data before analysis)


def has_multi_intent_signal(text: str) -> bool:
    """Return True if text contains explicit or implicit multi-intent signals.

    Detection patterns:
      1. Explicit connector-driven: "顺便", "然后", etc. between two intents
      2. Comma-driven: two comma-separated clauses with distinct signals
      3. Implicit: query simultaneously carries search + writing signals
         (e.g. "规划行程" needs both search for info and writing for plan)
    """
    # Pattern 1: connector-driven
    for conn in MULTI_INTENT_CONNECTORS:
        if conn in text:
            parts = text.split(conn, 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                left, right = parts[0].strip(), parts[1].strip()
                left_sig = _tool_signal(left)
                right_sig = _tool_signal(right)
                # Only count as multi-intent when BOTH sides have a signal
                # and they are different tools.
                if left_sig and right_sig and left_sig != right_sig:
                    return True

    # Pattern 3: implicit multi-intent — search + writing in same query
    # e.g. "规划3天北京行程" has both travel (search) + planning (writing)
    if _has_implicit_multi_intent(text):
        return True

    return False


# Verbs that imply a synthetic/writing task (need bigmodel_proxy for output)
_PLANNING_VERBS = (
    "规划",
    "制定",
    "设计",
    "安排",
    "策划",
    "方案",
    "计划",
    "生成",
    "编写",
    "起草",
    "撰写",
    "编一个",
    "写一个",
    "做个",
    "做一个",
)


def _has_implicit_multi_intent(text: str) -> bool:
    """Detect implicit multi-intent: query needs both search AND synthesis.

    Pattern: a planning/writing verb + a search-requiring topic (travel,
    weather, venue, etc.) without an explicit connector.

    Example: "规划3天的北京旅游行程" → search(景点/酒店) + writing(行程安排)
    """
    has_plan_verb = any(verb in text for verb in _PLANNING_VERBS)
    if not has_plan_verb:
        return False
    # Check if the query also has a topic that needs search
    has_search_topic = has_travel_signal(text) or has_search_action_signal(text)
    # Must be both: planning verb (writing) + search-worthy topic
    return has_search_topic


def _tool_signal(text: str) -> IntentCategory | None:
    """Return the strongest intent category for a text segment, or None."""
    if has_low_intent_signal(text):
        return None
    if has_calc_signal(text):
        return IntentCategory.CALC
    if has_code_signal(text):
        return IntentCategory.CODE
    if has_search_action_signal(text) or has_travel_signal(text):
        return IntentCategory.SEARCH
    # English tech metrics → search (status/metrics lookup)
    if any(t in text.lower() for t in TECH_METRIC_TOKENS):
        return IntentCategory.SEARCH
    if has_writing_signal(text) or has_translate_signal(text):
        return IntentCategory.CONSULT
    return None


def split_multi_intent(text: str) -> list[SplitIntent]:
    """Split a multi-intent query into sequential sub-intents.

    Handles two patterns:
      1. 先X然后Y / 顺便Y  — connector-driven split
      2. X,Y  — comma-driven split when both segments have strong signals
    """
    # Pattern 1: connector-driven
    for conn in MULTI_INTENT_CONNECTORS:
        if conn in text:
            parts = text.split(conn, 1)
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()
                left_sig = _tool_signal(left)
                right_sig = _tool_signal(right)
                if left_sig and right_sig and left_sig != right_sig:
                    intents = [
                        SplitIntent(
                            text=left,
                            primary_signal=left_sig,
                            priority=1 if left_sig == IntentCategory.SEARCH else 2,
                        ),
                        SplitIntent(
                            text=right,
                            primary_signal=right_sig,
                            priority=1 if right_sig == IntentCategory.SEARCH else 2,
                        ),
                    ]
                    return sorted(intents, key=lambda x: x.priority)

    # Pattern 2: comma-driven, only when both sides have strong signals
    for delim in MULTI_INTENT_DELIMITERS:
        if delim in text:
            parts = [p.strip() for p in text.split(delim) if p.strip()]
            if len(parts) >= 2:
                signals_list = []
                for part in parts:
                    if has_calc_signal(part):
                        signals_list.append(IntentCategory.CALC)
                    elif has_code_signal(part):
                        signals_list.append(IntentCategory.CODE)
                    elif has_search_action_signal(part):
                        signals_list.append(IntentCategory.SEARCH)
                    else:
                        signals_list.append(None)
                # Only treat as multi-intent if at least 2 parts have
                # distinct strong signals.
                non_null = [s for s in signals_list if s is not None]
                if len(set(non_null)) >= 2:
                    return [
                        SplitIntent(
                            text=part,
                            primary_signal=sig or IntentCategory.CONSULT,
                            priority=1 if sig == IntentCategory.SEARCH else 2,
                        )
                        for part, sig in zip(parts, signals_list)
                    ]

    # Pattern 3: implicit multi-intent (planning verb + search topic)
    # Split into: search(gather info) → bigmodel_proxy(synthesize plan)
    if _has_implicit_multi_intent(text):
        return [
            SplitIntent(
                text=text,  # use full query for search
                primary_signal=IntentCategory.SEARCH,
                priority=1,
            ),
            SplitIntent(
                text=text,  # use full query for synthesis
                primary_signal=IntentCategory.CONSULT,
                priority=2,
            ),
        ]

    return []
