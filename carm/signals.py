"""Shared signal detection utilities used across CARM modules.

Centralizes conflict detection, tokenization, and intent signal extraction
that were previously duplicated across core.py, policy.py, concepts.py,
and evolution.py.
"""

from __future__ import annotations

import re
from collections import Counter


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

COMPARE_TOKENS = ("比较", "区别", "优缺点", "vs", "对比")
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
    # Unit conversion keywords
    "公里",
    "千米",
    "小时",
    "等于多少",
)
CODE_TOKENS_EN = ("python", "code", "script")
CODE_TOKENS_ZH = ("代码", "脚本", "报错")
CODE_ACTION_TOKENS = (
    "写",
    "实现",
    "编写",
    "开发",
    "运行",
    "执行",
    "跑一下",
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
)

# Tokens that indicate a WRITING/SYNTHESIS intent, not a code intent.
# When these are present, "写" should be interpreted as writing/essay,
# not code execution.
WRITING_TOKENS = (
    "作文",
    "议论文",
    "文章",
    "散文",
    "总结",
    "归纳",
    "提炼",
    "报告",
    "汇报",
    "方案",
    "建议书",
    "心得",
    "读后感",
    "观后感",
    "一首",
    "一篇",
    "一封",
    "封信",
)

# Tokens that indicate an EXPLANATION intent, not an execution intent.
# When these are present alongside algorithm names, the user likely wants
# to understand the concept, not run code.
EXPLAIN_TOKENS = (
    "解释",
    "什么是",
    "介绍一下",
    "说说",
    "讲解",
    "理解",
    "概念",
    "原理",
    "含义",
    "为什么",
    "怎么回事",
    "指的是",
    "指的是什么",
    "如何理解",
)

# Well-known algorithm names that are strong signals for code execution.
# These override the need for a language keyword + action verb pairing.
ALGORITHM_TOKENS = (
    "排序",
    "快速排序",
    "冒泡排序",
    "归并排序",
    "二分查找",
    "斐波那契",
    "递归",
    "遍历",
    "反转",
    "去重",
    "二分",
    "阶乘",
)
# Tokens that indicate a CONSULTATIVE/ADVISORY intent, not an execution intent.
# When these appear alongside code/algorithm tokens, the user wants
# advice/analysis/knowledge, not code execution.
# Tokens that indicate a TRAVEL/LIFESTYLE service intent.
# These should route to search (factual info) rather than calculator/code.
TRAVEL_TOKENS = (
    "机票",
    "航班",
    "飞机票",
    "火车票",
    "高铁票",
    "酒店",
    "住宿",
    "民宿",
    "天气",
    "气温",
    "降雨",
    "地图",
    "导航",
    "路线",
    "地铁",
    "公交",
    "打车",
    "出租车",
    "滴滴",
    "出行",
    "旅游",
    "景点",
    "门票",
    "攻略",
    "租车",
    "自驾",
)

CONSULT_TOKENS = (
    "优化",
    "分析",
    "选择",
    "选哪个",
    "哪种",
    "如何选择",
    "该怎么",
    "瓶颈",
    "改进",
    "提升",
    "性能",
    "评估",
    "对比",
)

# Tokens that indicate a TRANSLATION intent — always bigmodel_proxy.
TRANSLATE_TOKENS = (
    "翻译",
    "译成",
    "译为",
    "translate",
)

# Tokens that indicate a TEXT POLISHING intent — always bigmodel_proxy.
POLISH_TOKENS = (
    "润色",
    "修改",
    "改写",
    "修饰",
    "调整措辞",
    "文风",
    "语气",
)

FORMAL_TOKENS = (
    "负责人",
    "管理层",
    "正式",
    "简洁",
    "组织",
    "结论",
    "几份资料",
    "材料",
    "决策建议",
    "日志",
    "告警",
    "复盘",
)


def has_code_signal(text: str) -> bool:
    """Return True if the text indicates a code execution intent.

    Requires either a code-language keyword paired with an action verb,
    an explicit code/debugging token, or a well-known algorithm name
    (which is itself a strong signal for code execution).
    Bare language names like "Python" without a coding verb do not
    trigger this signal.

    Explain tokens and writing tokens override algorithm-name signals —
    "解释递归" should route to search, "写一篇议论文" to bigmodel_proxy.
    Explicit search actions ("搜索一下Python教程") also override code.
    """
    lower = text.lower()
    has_lang = any(token in lower for token in CODE_TOKENS_EN)
    has_action = any(token in text for token in CODE_ACTION_TOKENS)
    has_zh_code = any(token in text for token in CODE_TOKENS_ZH)
    has_algorithm = any(token in text for token in ALGORITHM_TOKENS)
    has_explain = any(token in text for token in EXPLAIN_TOKENS)
    has_writing = any(token in text for token in WRITING_TOKENS)
    has_search_action = has_search_action_signal(text)
    has_translate = has_translate_signal(text)
    has_polish = has_polish_signal(text)
    has_consult = has_consult_signal(text)
    # Writing intent ("写一篇议论文") overrides code intent from "写"
    if has_writing:
        return False
    # Translation/polish intent always overrides code intent
    if has_translate or has_polish:
        return False
    # Explicit search actions ("搜索一下Python") override code intent
    if has_search_action:
        return False
    # Consultative/advisory intent ("优化/分析/选择") overrides algorithm signals
    # "如何选择排序算法" is advisory, not execution — but "写一个排序" is still code.
    # Consult only overrides when there is NO strong code action verb ("运行/写/实现/执行").
    _strong_code_verbs = ("运行", "写", "实现", "编写", "执行", "跑一下")
    has_strong_code_verb = any(v in text for v in _strong_code_verbs)
    if has_consult and not has_strong_code_verb:
        return False
    # Language name alone is not enough — must pair with action or have
    # an explicit code/debugging/algorithm token.
    # BUT: explain tokens override algorithm signals — "解释递归"
    # should go to search, not code_executor.
    if has_explain:
        return has_zh_code or (has_lang and has_action)
    return has_zh_code or (has_lang and has_action) or has_algorithm


def has_explain_signal(text: str) -> bool:
    """Return True if the text indicates an explanation/knowledge intent.

    When this is True, code execution routing should be suppressed —
    the user wants to understand a concept, not run code.
    """
    return any(token in text for token in EXPLAIN_TOKENS)


COMPARISON_EVIDENCE_TOKENS = ("比较", "对比", "区别", "优缺点", "性能表现", "适用性")
SEARCH_TOKENS = (
    "最佳实践",
    "推荐",
    "教程",
    "指南",
    "文档",
    "如何",
    "怎么",
    "怎样",
    "什么",
    "哪个",
    "哪一种",
    "有没有",
    "好不好",
    "怎么样",
    "什么意思",
    "了解",
    "学习",
    "入门",
    "进阶",
    "经验",
    "案例",
    "示例",
    "最新",
    "趋势",
    "新闻",
    "动态",
)


def has_compare_signal(text: str) -> bool:
    return any(token in text for token in COMPARE_TOKENS)


def has_calc_signal(text: str) -> bool:
    """Return True if the text indicates a calculation intent.

    Excludes date/time queries (e.g. "今天的日期是多少") which have "多少"
    but are not calculation requests.
    """
    _date_keywords = (
        "日期",
        "时间",
        "几点",
        "什么时候",
        "哪天",
        "星期几",
        "几号",
        "多少天",
    )
    if any(kw in text for kw in _date_keywords):
        return False
    return any(token in text for token in CALC_TOKENS) or _has_arithmetic_op(text)


def has_formal_signal(text: str) -> bool:
    return any(token in text for token in FORMAL_TOKENS)


def has_comparison_evidence_signal(text: str) -> bool:
    return any(token in text for token in COMPARISON_EVIDENCE_TOKENS)


def has_travel_signal(text: str) -> bool:
    """Return True if the text indicates a travel/lifestyle service intent."""
    return any(token in text for token in TRAVEL_TOKENS)


def has_search_signal(text: str) -> bool:
    """Return True if the text indicates a search/information-seeking intent.

    Includes travel/lifestyle queries which are fact-based information needs.
    """
    return any(token in text for token in SEARCH_TOKENS) or has_travel_signal(text)


def has_writing_signal(text: str) -> bool:
    """Return True if the text indicates a writing/synthesis intent.

    When this is True alongside "写", the intent is writing/essay
    (bigmodel_proxy), not code execution.
    """
    return any(token in text for token in WRITING_TOKENS)


def has_translate_signal(text: str) -> bool:
    """Return True if the text indicates a translation intent."""
    return any(token in text for token in TRANSLATE_TOKENS)


def has_polish_signal(text: str) -> bool:
    """Return True if the text indicates a text polishing intent."""
    return any(token in text for token in POLISH_TOKENS)


def has_consult_signal(text: str) -> bool:
    """Return True if the text indicates a consultative/advisory intent.

    When code/algorithm tokens are also present, the user likely wants
    advice or analysis, not code execution.  For example:
    - "如何选择合适的排序算法" → search (advisory)
    - "优化排序算法性能" → search (advisory)
    - "写一个快速排序" → code_executor (execution)
    """
    return any(token in text for token in CONSULT_TOKENS)


def has_search_action_signal(text: str) -> bool:
    """Return True if the text explicitly requests a search action.

    "搜索一下" / "搜一下" / "查一下" / "帮我查" are explicit search
    actions that should override code intent.
    """
    _search_action_verbs = (
        "搜索一下",
        "搜一下",
        "查一下",
        "帮我搜",
        "帮我查",
        "搜索",
        "查找资料",
    )
    return any(v in text for v in _search_action_verbs)


def _has_arithmetic_op(text: str) -> bool:
    """Check for explicit arithmetic operators between numbers."""
    return bool(re.search(r"\d+\s*[\*\/+\-]\s*\d+", text))


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Tokenize text into ASCII and Chinese n-gram tokens.

    This is the canonical tokenizer used by core, policy, and concepts modules.
    - ASCII: runs of >=2 alphanumeric/underscore chars (lowercased)
    - Chinese: runs of >=2 CJK chars, plus 2-gram and 3-gram substrings
    - Deduplicated, preserving first occurrence order
    """
    ascii_tokens: list[str] = []
    current: list[str] = []
    for char in text.lower():
        if char.isascii() and (char.isalnum() or char == "_"):
            current.append(char)
        else:
            if len(current) >= 2:
                ascii_tokens.append("".join(current))
            current = []
    if len(current) >= 2:
        ascii_tokens.append("".join(current))

    chinese_runs: list[str] = []
    current = []
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            current.append(char)
        else:
            if len(current) >= 2:
                chinese_runs.append("".join(current))
            current = []
    if len(current) >= 2:
        chinese_runs.append("".join(current))

    chinese_tokens: list[str] = []
    for run in chinese_runs:
        chinese_tokens.append(run)
        for size in (2, 3):
            for idx in range(0, max(0, len(run) - size + 1)):
                chinese_tokens.append(run[idx : idx + size])

    return list(dict.fromkeys(ascii_tokens + chinese_tokens))


def token_counts(text: str) -> Counter:
    """Return Counter of token frequencies."""
    return Counter(tokenize(text))
