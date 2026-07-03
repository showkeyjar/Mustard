"""Lightweight semantic encoder for CARM intent understanding.

Provides semantic similarity and intent classification that augments the
existing keyword-based signal detection in carm.signals. Uses a two-tier
approach:

Tier 1 (zero-dependency): Pattern-based intent expansion using synonym
dictionaries and simple morphological rules. Always available.

Tier 2 (optional): sentence-transformers embedding for true semantic
matching. Activated when the library is installed and a model is cached.
Gracefully degrades to Tier 1 when unavailable.

Design principles:
- Zero breaking changes: existing keyword signals still work exactly as before
- Additive: semantic signals are blended with keyword signals, not replacing them
- Lazy loading: heavy dependencies are only imported when first used
- Cache-friendly: embeddings are cached for repeated queries
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Intent label space — aligns with tool selection in policy.py
# ---------------------------------------------------------------------------

INTENT_LABELS = ("search", "calculator", "code_executor", "bigmodel_proxy", "ambiguous")

# Synonym expansion map: maps intent labels to sets of trigger phrases
# These go BEYOND the keyword triggers in signals.py — they cover synonyms,
# paraphrases, and implicit intent expressions.

INTENT_SYNONYMS: dict[str, list[str]] = {
    "search": [
        # Direct synonyms for comparison
        "比较",
        "对比",
        "区别",
        "优缺点",
        "差异",
        "异同",
        "哪个好",
        "选哪个",
        "取舍",
        "权衡",
        "优劣",
        "不同",
        "不同之处",
        "差别",
        "区分",
        "选择",
        # Fact-checking synonyms
        "查",
        "搜索",
        "找",
        "检索",
        "了解",
        "调查",
        "核实",
        "确认",
        "验证",
        "查证",
        "考证",
        "查一下",
        "搜一下",
        "找找",
        # Evidence gathering
        "根据",
        "依据",
        "证据",
        "事实",
        "来源",
        "出处",
        "数据",
        "统计",
        "可信",
        "可靠",
        "过时",
        "最新",
        # Research/knowledge
        "是什么",
        "怎么理解",
        "原理",
        "概念",
        "定义",
        "解释",
        "介绍",
        "教程",
        "指南",
        "文档",
        "参考",
        "文献",
        # Practical guidance
        "最佳实践",
        "推荐",
        "如何",
        "怎么",
        "怎样",
        "什么",
        "哪个",
        "怎么样",
        "好不好",
        "有没有",
        "什么意思",
        "学习",
        "入门",
        "进阶",
        "经验",
        "案例",
        "示例",
        "趋势",
        "新闻",
        "动态",
        # Conflict
        "冲突",
        "矛盾",
        "不一致",
        "相反",
        "分歧",
        # Explanation/knowledge (overrides code intent — "解释递归" → search)
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
        "如何理解",
    ],
    "calculator": [
        # Direct calculation synonyms
        "计算",
        "算",
        "求值",
        "估算",
        "预算",
        "总价",
        "费用",
        "成本",
        "花多少",
        "需要多少",
        "多少钱",
        "总额",
        "合计",
        # Arithmetic indicators
        "乘以",
        "除以",
        "加上",
        "减去",
        "倍",
        "折",
        "百分比",
        "比例",
        "率",
        "增长率",
        "同比",
        "环比",
        # Financial/budget
        "每席位",
        "按年",
        "每月",
        "分几批",
        "扩容",
        "降价",
        "涨价",
        "价格",
        "单价",
        "收费",
        "计费",
        "报价",
        # Forecast
        "预测",
        "预计",
        "预估",
        "推算",
        "测算",
        # Math context
        "多少",
        "数字",
        "数量",
        "量化",
        # Mathematical operations
        "平方根",
        "开方",
        "开根号",
        "平方",
        "次方",
        "乘以",
        "除以",
        "加上",
        "减去",
        "加",
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
    ],
    "code_executor": [
        # Direct coding synonyms — only with action verbs
        "代码",
        "编程",
        "程序",
        "脚本",
        "实现",
        "编写",
        "开发",
        "函数",
        "算法",
        # Common algorithm names (strong signal for code execution)
        "排序",
        "快速排序",
        "冒泡排序",
        "归并排序",
        "二分查找",
        "查找",
        "搜索算法",
        "遍历",
        "反转",
        "去重",
        "合并",
        "递归",
        "斐波那契",
        "二分",
        "阶乘",
        "factorial",
        # Debugging
        "报错",
        "错误",
        "异常",
        "bug",
        "调试",
        "debug",
        "排查",
        "修复",
        "解决",
        # Execution
        "运行",
        "执行",
        "跑一下",
        "试试",
        # Logic
        "逻辑",
        "方法",
        "类",
        "模块",
        # Language+action combos (not bare language names)
        "写代码",
        "写程序",
        "写脚本",
        "写函数",
        "python函数",
        "python代码",
        "python实现",
        "python脚本",
    ],
    "bigmodel_proxy": [
        # Formal synthesis
        "管理层",
        "负责人",
        "正式",
        "简洁",
        "组织",
        "结论",
        "决策建议",
        "报告",
        "总结",
        "归纳",
        "提炼",
        # Multi-source
        "几份资料",
        "几份材料",
        "综合",
        "整合",
        "汇总",
        "合并",
        "梳理",
        "整理",
        # Creative/complex
        "起草",
        "撰写",
        "润色",
        "改写",
        "扩写",
        "摘要",
        "概要",
        "日志",
        "告警",
        "复盘",
        "反思",
        "分析",
    ],
}

# Build reverse lookup: phrase -> intent (for fast lookup)
_PHRASE_TO_INTENT: dict[str, str] = {}
for intent, phrases in INTENT_SYNONYMS.items():
    for phrase in phrases:
        _PHRASE_TO_INTENT[phrase] = intent


class SemanticEncoder:
    """Lightweight semantic encoder that augments keyword-based signals.

    Provides:
    1. intent_scores(): similarity scores per intent label
    2. intent_top(): top-k intent predictions
    3. semantic_similarity(): pairwise similarity between two texts

    Tier 1 is always available (pattern-based).
    Tier 2 activates when sentence-transformers is installed.
    """

    def __init__(self, cache_path: str | Path | None = None) -> None:
        self._model = None
        self._model_available = False
        self._cache: OrderedDict[str, dict[str, float]] = OrderedDict()
        self._cache_max = 512
        self._cache_path = Path(cache_path) if cache_path else None
        self._load_cache()
        self._init_model()

    def _init_model(self) -> None:
        """Try to load sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

            self._model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            self._model_available = True
        except (ImportError, Exception):
            self._model = None
            self._model_available = False

    def intent_scores(self, text: str) -> dict[str, float]:
        """Return intent scores for the given text.

        Returns a dict mapping each intent label to a score in [0, 1].
        Combines pattern-based and (if available) embedding-based signals.
        """
        # Check cache first
        cache_key = hashlib.md5(text.encode("utf-8")).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Tier 1: Pattern-based scoring
        scores = self._pattern_scores(text)

        # Tier 2: Embedding-based scoring (if available)
        if self._model_available:
            emb_scores = self._embedding_scores(text)
            if emb_scores:
                # Blend: 0.4 pattern + 0.6 embedding
                for intent in INTENT_LABELS:
                    pattern_val = scores.get(intent, 0.0)
                    embed_val = emb_scores.get(intent, 0.0)
                    scores[intent] = 0.4 * pattern_val + 0.6 * embed_val

        # Normalize to sum to ~1.0 for non-ambiguous
        total = sum(v for k, v in scores.items() if k != "ambiguous")
        if total > 0:
            for k in scores:
                if k != "ambiguous":
                    scores[k] /= total

        # If top two intents are very close AND both have meaningful scores,
        # mark as ambiguous. Low absolute scores should not produce ambiguity.
        sorted_scores = sorted(
            ((k, v) for k, v in scores.items() if k != "ambiguous"),
            key=lambda x: x[1],
            reverse=True,
        )
        if len(sorted_scores) >= 2 and sorted_scores[0][1] > 0.3:
            ratio = sorted_scores[1][1] / sorted_scores[0][1]
            if ratio > 0.8:
                scores["ambiguous"] = 1.0 - abs(
                    sorted_scores[0][1] - sorted_scores[1][1]
                )
            else:
                scores["ambiguous"] = 0.0
        else:
            scores["ambiguous"] = 0.0

        # Cache result
        self._cache[cache_key] = scores
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)

        return scores

    def intent_top(self, text: str, k: int = 2) -> list[tuple[str, float]]:
        """Return top-k intent predictions as (label, score) pairs."""
        scores = self.intent_scores(text)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def semantic_similarity(self, text_a: str, text_b: str) -> float:
        """Compute semantic similarity between two texts.

        Returns value in [0, 1]. Uses embeddings if available, else
        falls back to n-gram overlap.
        """
        if self._model_available and self._model is not None:
            try:
                import numpy as np  # type: ignore[import-untyped]

                emb_a = self._model.encode(text_a, normalize_embeddings=True)
                emb_b = self._model.encode(text_b, normalize_embeddings=True)
                sim = float(np.dot(emb_a, emb_b))
                return max(0.0, min(1.0, (sim + 1.0) / 2.0))
            except Exception:
                pass

        # Fallback: n-gram Jaccard similarity
        return self._ngram_similarity(text_a, text_b)

    def _pattern_scores(self, text: str) -> dict[str, float]:
        """Tier 1: Score intents based on synonym pattern matching."""
        scores: dict[str, float] = {intent: 0.0 for intent in INTENT_LABELS}
        text_lower = text.lower()

        # Date/time queries should not trigger calculator intent
        _date_keywords = (
            "日期",
            "时间",
            "几点",
            "什么时候",
            "哪天",
            "星期几",
            "几号",
            "今天",
            "明天",
            "昨天",
            "前天",
            "后天",
            "多少天",
        )
        is_date_query = any(kw in text for kw in _date_keywords)

        for phrase, intent in _PHRASE_TO_INTENT.items():
            # Skip calculator phrases for date/time queries
            if intent == "calculator" and is_date_query:
                continue
            if phrase.lower() in text_lower:
                # Longer phrases get higher weight (more specific)
                weight = 1.0 + len(phrase) * 0.1
                scores[intent] += weight

        # Boost for arithmetic operators (calculator indicator)
        if any(op in text for op in ("*", "/", "+", "-")):
            import re

            if re.search(r"\d+\s*[\*\/+\-]\s*\d+", text):
                scores["calculator"] += 2.0

        # Date queries are search/knowledge intent
        if is_date_query:
            scores["search"] += 2.0

        return scores

    def _embedding_scores(self, text: str) -> dict[str, float] | None:
        """Tier 2: Score intents using sentence embeddings."""
        if not self._model_available or self._model is None:
            return None

        try:
            import numpy as np  # type: ignore[import-untyped]

            # Encode the query
            query_emb = self._model.encode(text, normalize_embeddings=True)

            # Encode intent prototype descriptions
            prototypes = {
                "search": "搜索查询比较信息事实证据核实验证",
                "calculator": "计算数值预算价格成本算术数学",
                "code_executor": "代码编程程序执行调试开发Python",
                "bigmodel_proxy": "综合总结归纳正式结论管理层报告",
            }

            scores: dict[str, float] = {}
            for intent, proto_text in prototypes.items():
                proto_emb = self._model.encode(proto_text, normalize_embeddings=True)
                sim = float(np.dot(query_emb, proto_emb))
                # Convert from [-1,1] to [0,1]
                scores[intent] = max(0.0, (sim + 1.0) / 2.0)

            return scores
        except Exception:
            return None

    def _ngram_similarity(self, text_a: str, text_b: str) -> float:
        """Fallback similarity using character n-gram Jaccard."""

        def _char_ngrams(text: str, n: int = 2) -> set[str]:
            return {text[i : i + n] for i in range(max(0, len(text) - n + 1))}

        ngrams_a = _char_ngrams(text_a) | _char_ngrams(text_a, 3)
        ngrams_b = _char_ngrams(text_b) | _char_ngrams(text_b, 3)

        if not ngrams_a or not ngrams_b:
            return 0.0

        intersection = ngrams_a & ngrams_b
        union = ngrams_a | ngrams_b
        return len(intersection) / len(union)

    def _load_cache(self) -> None:
        if self._cache_path and self._cache_path.exists():
            try:
                data = json.loads(self._cache_path.read_text(encoding="utf-8"))
                self._cache.update(data)
            except (json.JSONDecodeError, OSError):
                pass

    def save_cache(self) -> None:
        if self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
