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
    "搜索",
    "遍历",
    "反转",
    "去重",
    "合并",
    "递归",
    "斐波那契",
    "二分",
    "阶乘",
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

    Explain tokens (EXPLAIN_TOKENS) override algorithm-name signals —
    "解释递归" should route to search, not code_executor.
    """
    lower = text.lower()
    has_lang = any(token in lower for token in CODE_TOKENS_EN)
    has_action = any(token in text for token in CODE_ACTION_TOKENS)
    has_zh_code = any(token in text for token in CODE_TOKENS_ZH)
    has_algorithm = any(token in text for token in ALGORITHM_TOKENS)
    has_explain = any(token in text for token in EXPLAIN_TOKENS)
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
    return any(token in text for token in CALC_TOKENS) or _has_arithmetic_op(text)


def has_formal_signal(text: str) -> bool:
    return any(token in text for token in FORMAL_TOKENS)


def has_comparison_evidence_signal(text: str) -> bool:
    return any(token in text for token in COMPARISON_EVIDENCE_TOKENS)


def has_search_signal(text: str) -> bool:
    """Return True if the text indicates a search/information-seeking intent."""
    return any(token in text for token in SEARCH_TOKENS)


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
