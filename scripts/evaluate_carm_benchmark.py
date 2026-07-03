"""CARM benchmark evaluation against external reference benchmarks.

Runs four benchmark-equivalent evaluations with graduated difficulty:
  1. SMP2017-ECDT — Chinese intent detection (→ tool routing)
  2. Math23K — Chinese math word problems (→ calculator accuracy)
  3. BFCL V3/V4 — Berkeley function calling (→ tool routing + multi-step)
  4. MMLU-CN — General knowledge reasoning (→ search + LLM accuracy)

Difficulty tiers:
  - L1 (Easy):   CARM should always get these right
  - L2 (Medium): CARM should get most right, some wrong
  - L3 (Hard):   CARM will mostly fail — these expose the ceiling
  - L4 (Beyond): CARM architecture cannot solve these at all

Reference scores (from papers / public leaderboards):
  - SMP2017 intent accuracy: BERT-base ~94.1%, GPT-3.5 ~96%, GPT-4 ~98%
  - Math23K answer accuracy: GTS ~74%, BERT-based ~82.6%, GPT-3.5 ~78%, GPT-4 ~92%
  - BFCL V4 overall: Mistral-7B ~55%, GPT-3.5 ~78%, GPT-4-turbo ~88%
  - MMLU-CN: BERT-base ~42%, GPT-3.5 ~68%, GPT-4 ~87%

Usage:
    python -m scripts.evaluate_carm_benchmark
    python -m scripts.evaluate_carm_benchmark --output data/eval/benchmark_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from carm.session_memory import SessionMemoryManager
from carm.signals import has_anaphora_signal

# ===========================================================================
# Benchmark 1: SMP2017-ECDT equivalent — Chinese Intent Detection
# ===========================================================================
# Published accuracy: BERT ~94%, GPT-3.5 ~96%, GPT-4 ~98%
# ===========================================================================

SMP2017_CASES = [
    # ══ L1 Easy: single clear intent ══════════════════════════════════
    {
        "query": "帮我算一下123乘以456等于多少",
        "expected_tool": "calculator",
        "level": "L1",
    },
    {"query": "2的20次方是多少", "expected_tool": "calculator", "level": "L1"},
    {"query": "1万亿除以14亿", "expected_tool": "calculator", "level": "L1"},
    {"query": "帮我写一个冒泡排序", "expected_tool": "code_executor", "level": "L1"},
    {"query": "用python实现快速排序", "expected_tool": "code_executor", "level": "L1"},
    {"query": "今天天气怎么样", "expected_tool": "search", "level": "L1"},
    {"query": "什么是人工智能", "expected_tool": "search", "level": "L1"},
    {
        "query": "帮我写一份项目总结报告",
        "expected_tool": "bigmodel_proxy",
        "level": "L1",
    },
    # ══ L2 Medium: ambiguous signals, need careful resolution ═══════
    {"query": "5加3乘2等于多少", "expected_tool": "calculator", "level": "L2"},
    {"query": "圆的面积 半径是5", "expected_tool": "calculator", "level": "L2"},
    {"query": "解释一下什么是动态规划", "expected_tool": "search", "level": "L2"},
    {"query": "比较冒泡排序和快速排序", "expected_tool": "search", "level": "L2"},
    {
        "query": "写一篇关于AI伦理的议论文",
        "expected_tool": "bigmodel_proxy",
        "level": "L2",
    },
    {"query": "搜索一下Python教程", "expected_tool": "search", "level": "L2"},
    {
        "query": "10的阶乘是多少",
        "expected_tool": "calculator",
        "bfcl_category": "simple",
        "level": "L2",
    },
    {
        "query": "帮我写一个关于历史的作文",
        "expected_tool": "bigmodel_proxy",
        "level": "L2",
    },
    # ══ L3 Hard: non-obvious routing, requires deeper understanding ══
    # These are cases where CARM has a fighting chance but may get wrong
    # without specific signal handling
    {
        "query": "我的代码报错了IndexError怎么解决",
        "expected_tool": "search",
        "level": "L3",
        "note": "Debug is knowledge, not execution — but '代码' and '报错' push toward code_executor",
    },
    {
        "query": "这段代码有什么问题 def foo(): return 1/0",
        "expected_tool": "search",
        "level": "L3",
        "note": "Code review = analysis, not execution",
    },
    {
        "query": "帮我优化一下这段排序算法的性能",
        "expected_tool": "search",
        "level": "L3",
        "note": "Algorithm optimization is knowledge/analysis",
    },
    {
        "query": "如何选择合适的排序算法",
        "expected_tool": "search",
        "level": "L3",
        "note": "No explicit code action, but '算法' may trigger code signal",
    },
    {
        "query": "帮我订一张去上海的机票",
        "expected_tool": "search",
        "level": "L3",
        "note": "No flight tool — should route to search for info, not pretend to execute",
    },
    {
        "query": "设置一个明天早上7点的闹钟",
        "expected_tool": "search",
        "level": "L3",
        "note": "No alarm tool — search for how-to, not execute",
    },
    {
        "query": "翻译一下这段英文",
        "expected_tool": "bigmodel_proxy",
        "level": "L3",
        "note": "Translation needs LLM, not search or code",
    },
    {
        "query": "帮我润色一下这段文字",
        "expected_tool": "bigmodel_proxy",
        "level": "L3",
        "note": "Text polishing needs LLM",
    },
    {
        "query": "原价200元打8折后是多少元",
        "expected_tool": "calculator",
        "level": "L3",
        "note": "Discount NL pattern — CARM should route to calc AND compute correctly",
    },
    # ══ L4 Beyond: requires capabilities CARM fundamentally lacks ═══
    {
        "query": "帮我查一下北京天气，顺便算一下3加5",
        "expected_tool": "multi_intent",
        "level": "L4",
        "partial_tools": ["search", "calculator"],
        "note": "Requires both search AND calculator — CARM can only route to one tool",
        "partial_tools": ["search", "calculator"],
    },
    {
        "query": "先搜索量子计算的发展，然后帮我写个总结",
        "expected_tool": "multi_intent",
        "level": "L4",
        "partial_tools": ["search", "bigmodel_proxy"],
        "note": "Requires search THEN bigmodel_proxy — multi-step planning",
        "partial_tools": ["search", "bigmodel_proxy"],
    },
    {
        "query": "帮我规划一个3天的北京旅游行程",
        "expected_tool": "multi_intent",
        "level": "L4",
        "note": "Needs search(hotels) + search(attractions) + bigmodel_proxy(itinerary)",
        "partial_tools": ["search", "bigmodel_proxy"],
    },
    {
        "query": "它的性能怎么样",
        "expected_tool": "context_needed",
        "level": "L4",
        "note": "'它' requires coreference — CARM has no context memory",
        "prime_query": "帮我查一下最新款GPU的性能参数",
        "partial_tools": ["search", "bigmodel_proxy"],
    },
]


# ===========================================================================
# Benchmark 2: Math23K equivalent — Chinese Math Word Problems
# ===========================================================================
# Published accuracy: GTS ~74%, BERT-based ~82.6%, GPT-3.5 ~78%, GPT-4 ~92%
# ===========================================================================

MATH23K_CASES = [
    # ══ L1 Easy: direct arithmetic ═══════════════════════════════════
    {"query": "5加3", "expected_answer": 8, "category": "simple_add", "level": "L1"},
    {
        "query": "24乘6",
        "expected_answer": 144,
        "category": "simple_multiply",
        "level": "L1",
    },
    {
        "query": "100减37",
        "expected_answer": 63,
        "category": "simple_subtract",
        "level": "L1",
    },
    {
        "query": "96除以8",
        "expected_answer": 12,
        "category": "simple_divide",
        "level": "L1",
    },
    {"query": "2的10次方", "expected_answer": 1024, "category": "power", "level": "L1"},
    {
        "query": "3.14乘以25",
        "expected_answer": 78.5,
        "category": "decimal",
        "level": "L1",
    },
    # ══ L2 Medium: unit conversion, NL extraction, geometry ════════
    {
        "query": "3公里等于多少米",
        "expected_answer": 3000,
        "category": "unit_convert",
        "level": "L2",
    },
    {
        "query": "2小时等于多少分钟",
        "expected_answer": 120,
        "category": "unit_convert",
        "level": "L2",
    },
    {
        "query": "1万亿除以14亿",
        "expected_answer": 714.2857,
        "tolerance": 0.01,
        "category": "large_number",
        "level": "L2",
    },
    {"query": "负3加5", "expected_answer": 2, "category": "negative", "level": "L2"},
    {
        "query": "长方形面积 长8米宽5米",
        "expected_answer": 40,
        "category": "geometry",
        "level": "L2",
    },
    {
        "query": "圆的面积 半径3",
        "expected_answer": 28.27,
        "tolerance": 0.05,
        "category": "geometry",
        "level": "L2",
    },
    {
        "query": "小明有5个苹果又买了3个一共有多少个",
        "expected_answer": 8,
        "category": "word_simple",
        "level": "L2",
    },
    {
        "query": "一本书15元买3本需要多少钱",
        "expected_answer": 45,
        "category": "word_simple",
        "level": "L2",
    },
    # ══ L3 Hard: percentage, discount, speed, partial multi-step ════
    # These should be SOLVABLE with the right NL pattern — testing
    # whether CARM's NL coverage is good enough, not whether it can
    # do symbolic reasoning
    {
        "query": "原价200元打8折后是多少元",
        "expected_answer": 160,
        "category": "discount",
        "level": "L3",
    },
    {
        "query": "一个班级40人及格率75%有多少人及格",
        "expected_answer": 30,
        "category": "percentage",
        "level": "L3",
    },
    {
        "query": "甲乙两地相距240公里一辆车3小时到达平均速度是多少",
        "expected_answer": 80,
        "category": "speed",
        "level": "L3",
    },
    {
        "query": "200的30%是多少",
        "expected_answer": 60,
        "category": "percentage",
        "level": "L3",
    },
    # Multi-step: CARM may extract only partial expression
    {
        "query": "小明买了3本书每本15元又买了2支笔每支3元一共花了多少钱",
        "expected_answer": 51,
        "category": "multi_step",
        "level": "L3",
        "note": "Two operations: 3*15 + 2*3 = 51 — CARM may only compute 3+2=5",
    },
    # Equation: CARM cannot do symbolic reasoning, but routing to calculator is correct
    {
        "query": "一个数的3倍加5等于20这个数是多少",
        "expected_answer": 5,
        "category": "equation",
        "level": "L3",
        "note": "Requires equation solving — calculator cannot solve, but routing is correct",
    },
    {
        "query": "甲比乙大3岁甲乙之和是27岁甲多少岁",
        "expected_answer": 15,
        "category": "equation",
        "level": "L3",
        "note": "System of equations — beyond calculator",
    },
    {
        "query": "鸡兔同笼共10个头26条腿鸡有几只",
        "expected_answer": 7,
        "category": "equation",
        "level": "L3",
        "note": "Classic puzzle — requires equation setup",
    },
    # ══ L4 Beyond: requires symbolic reasoning or knowledge ══════════
    {
        "query": "从1加到100的和是多少",
        "expected_answer": 5050,
        "category": "series",
        "level": "L4",
        "note": "Requires knowing n(n+1)/2 formula — not pure arithmetic",
    },
    {
        "query": "一个水池有两个进水管一个4小时灌满一个6小时灌满同时开多久灌满",
        "expected_answer": 2.4,
        "category": "work_problem",
        "level": "L4",
        "note": "1/(1/4+1/6) = 2.4h — work-rate reasoning",
    },
    {
        "query": "证明根号2是无理数",
        "expected_answer": None,
        "category": "proof",
        "level": "L4",
        "note": "Mathematical proof — far beyond calculator",
    },
]


# ===========================================================================
# Benchmark 3: BFCL V3/V4 equivalent — Tool / Function Routing
# ===========================================================================
# Published scores: Mistral-7B ~55%, GPT-3.5 ~78%, GPT-4-turbo ~88%
# Also includes V4 agentic dimensions: multi-turn, planning, tool chains
# ===========================================================================

BFCL_CASES = [
    # ══ L1 Easy: simple single-tool routing ══════════════════════════
    {
        "query": "帮我计算 2 + 3 * 4",
        "expected_tool": "calculator",
        "bfcl_category": "simple",
        "level": "L1",
    },
    {
        "query": "运行这段代码 print(sum([1,2,3]))",
        "expected_tool": "code_executor",
        "bfcl_category": "simple",
        "level": "L1",
    },
    {
        "query": "搜索一下Python教程",
        "expected_tool": "search",
        "bfcl_category": "simple",
        "level": "L1",
    },
    {
        "query": "帮我写一份市场分析报告",
        "expected_tool": "bigmodel_proxy",
        "bfcl_category": "simple",
        "level": "L1",
    },
    {
        "query": "计算100的平方根",
        "expected_tool": "calculator",
        "bfcl_category": "simple",
        "level": "L1",
    },
    {
        "query": "写一个快速排序算法",
        "expected_tool": "code_executor",
        "bfcl_category": "simple",
        "level": "L1",
    },
    # ══ L2 Medium: multiple candidates, disambiguation ══════════════
    {
        "query": "帮我算一下5的阶乘",
        "expected_tool": "calculator",
        "bfcl_category": "multiple",
        "level": "L2",
    },
    {
        "query": "比较一下Redis和Memcached的性能",
        "expected_tool": "search",
        "bfcl_category": "multiple",
        "level": "L2",
    },
    {
        "query": "解释一下什么是动态规划",
        "expected_tool": "search",
        "bfcl_category": "multiple",
        "level": "L2",
    },
    {
        "query": "帮我写一个关于历史的作文",
        "expected_tool": "bigmodel_proxy",
        "bfcl_category": "hallucination",
        "level": "L2",
    },
    {
        "query": "帮我归纳一下这些资料的核心观点",
        "expected_tool": "bigmodel_proxy",
        "bfcl_category": "hallucination",
        "level": "L2",
    },
    {
        "query": "帮我计算圆的面积半径是7",
        "expected_tool": "calculator",
        "bfcl_category": "parallel",
        "level": "L2",
    },
    # ══ L3 Hard: subtle routing boundaries ═════════════════════════
    # These are cases where the routing decision is genuinely ambiguous
    # or requires understanding that CARM's keyword system may miss
    {
        "query": "帮我优化一下这段排序算法的性能",
        "expected_tool": "search",
        "bfcl_category": "boundary",
        "level": "L3",
        "note": "Optimization advice is knowledge, not code execution",
    },
    {
        "query": "如何选择合适的排序算法",
        "expected_tool": "search",
        "bfcl_category": "boundary",
        "level": "L3",
        "note": "No explicit code action verb — advisory question",
    },
    {
        "query": "翻译一下这段英文",
        "expected_tool": "bigmodel_proxy",
        "bfcl_category": "boundary",
        "level": "L3",
        "note": "Translation needs LLM, not search",
    },
    {
        "query": "帮我润色一下这段文字",
        "expected_tool": "bigmodel_proxy",
        "bfcl_category": "boundary",
        "level": "L3",
        "note": "Text polishing needs LLM",
    },
    {
        "query": "我的代码报错了IndexError怎么解决",
        "expected_tool": "search",
        "bfcl_category": "boundary",
        "level": "L3",
        "note": "Debug help is knowledge — but '代码'+'报错' pushes to code_executor",
    },
    {
        "query": "原价200元打8折后是多少元",
        "expected_tool": "calculator",
        "bfcl_category": "boundary",
        "level": "L3",
        "note": "Discount is a calculation, not search — but '打8折' is unusual NL",
    },
    {
        "query": "如果我想要高性能的排序应该用哪种算法",
        "expected_tool": "search",
        "bfcl_category": "conditional",
        "level": "L3",
        "note": "Advisory/comparative, not code execution",
    },
    {
        "query": "帮我分析一下这个代码的性能瓶颈",
        "expected_tool": "search",
        "bfcl_category": "conditional",
        "level": "L3",
        "note": "Code analysis is knowledge, not execution",
    },
    # ══ L4 Beyond: requires capabilities CARM fundamentally lacks ═══
    {
        "query": "帮我查一下北京天气，顺便算一下3加5",
        "expected_tool": "multi_intent",
        "bfcl_category": "agentic",
        "level": "L4",
        "note": "Two intents in one query — CARM can only route to one tool",
    },
    {
        "query": "先搜索量子计算的发展，然后帮我写个总结",
        "expected_tool": "multi_intent",
        "bfcl_category": "agentic",
        "level": "L4",
        "note": "Sequential tool chain: search → bigmodel_proxy",
    },
    {
        "query": "用刚才的模型再跑一遍",
        "expected_tool": "context_needed",
        "bfcl_category": "multi_turn",
        "level": "L4",
        "note": "Requires conversation context",
        "prime_query": "运行一下这个Python模型",
        "partial_tools": ["code_executor"],
    },
]


# ===========================================================================
# Benchmark 4: MMLU-CN equivalent — General Knowledge Reasoning
# ===========================================================================
# MMLU (Massive Multitask Language Understanding) tests 57 subjects.
# We adapt to CARM's setup: knowledge questions that require search/LLM.
# CARM's "answer quality" depends entirely on the LLM backend quality.
# Published scores (MMLU-CN): BERT-base ~42%, GPT-3.5 ~68%, GPT-4 ~87%
# ===========================================================================

MMLU_CN_CASES = [
    # ══ L1 Easy: clear-cut routing — search for facts ════════════════
    {
        "query": "中国的首都是哪个城市",
        "expected_tool": "search",
        "category": "geography",
        "level": "L1",
    },
    {
        "query": "水的化学式是什么",
        "expected_tool": "search",
        "category": "chemistry",
        "level": "L1",
    },
    {
        "query": "一年有多少天",
        "expected_tool": "search",
        "category": "common",
        "level": "L1",
    },
    # ══ L2 Medium: routing still clear but knowledge is deeper ══════
    {
        "query": "量子纠缠是什么现象",
        "expected_tool": "search",
        "category": "physics",
        "level": "L2",
    },
    {
        "query": "TCP和UDP的区别是什么",
        "expected_tool": "search",
        "category": "cs",
        "level": "L2",
    },
    {
        "query": "Python的GIL是什么",
        "expected_tool": "search",
        "category": "cs",
        "level": "L2",
    },
    {
        "query": "什么是通货膨胀",
        "expected_tool": "search",
        "category": "economics",
        "level": "L2",
    },
    # ══ L3 Hard: routing ambiguity — knowledge questions that could
    # plausibly be answered by bigmodel_proxy instead of search ══════
    {
        "query": "帮我总结一下深度学习的核心思想",
        "expected_tool": "bigmodel_proxy",
        "category": "synthesis",
        "level": "L3",
        "note": "Summary/synthesis → bigmodel, not search",
    },
    {
        "query": "解释一下量子计算的基本原理",
        "expected_tool": "search",
        "category": "explanation",
        "level": "L3",
        "note": "Explanation but factual → search is acceptable too",
        "acceptable_tools": ["search", "bigmodel_proxy"],
    },
    {
        "query": "写一篇关于气候变化的科普文章",
        "expected_tool": "bigmodel_proxy",
        "category": "writing",
        "level": "L3",
        "note": "Writing request → bigmodel",
    },
    {
        "query": "比较BERT和GPT的架构差异并给出选择建议",
        "expected_tool": "bigmodel_proxy",
        "category": "comparative_synthesis",
        "level": "L3",
        "note": "Compare + advice → bigmodel synthesis, not just search",
    },
    {
        "query": "归纳一下这三种排序算法的优缺点",
        "expected_tool": "bigmodel_proxy",
        "category": "synthesis",
        "level": "L3",
        "note": "归纳 is a synthesis verb → bigmodel",
    },
    # Tricky: looks like search but needs LLM reasoning
    {
        "query": "为什么深度学习需要大量数据而传统机器学习不需要",
        "expected_tool": "bigmodel_proxy",
        "category": "reasoning",
        "level": "L3",
        "note": "'为什么' triggers search but this needs comparative reasoning → bigmodel",
    },
    # Tricky: "分析" with domain term could go either way
    {
        "query": "分析一下这个业务方案的可行性",
        "expected_tool": "bigmodel_proxy",
        "category": "advisory",
        "level": "L3",
        "note": "Business analysis → bigmodel, not search",
    },
    # Tricky: code term but advisory intent
    {
        "query": "微服务和单体架构在什么场景下分别适用",
        "expected_tool": "search",
        "category": "boundary",
        "level": "L3",
        "note": "Advisory/comparative → search for evidence",
        "acceptable_tools": ["search", "bigmodel_proxy"],
    },
    # ══ L4 Beyond: multi-intent or requires capabilities beyond CARM ═
    {
        "query": "搜索一下康德纯粹理性批判的核心论点，然后帮我写个读书笔记",
        "expected_tool": "multi_intent",
        "category": "multi_intent",
        "level": "L4",
        "note": "Search + write = two tools in sequence",
    },
    {
        "query": "上次查的那篇论文的核心结论是什么",
        "expected_tool": "context_needed",
        "category": "coreference",
        "level": "L4",
        "note": "Requires conversation context — '上次查的'",
        "prime_query": "搜索一下Transformer架构论文",
        "partial_tools": ["search", "bigmodel_proxy"],
    },
    {
        "query": "对比分析量子计算和经典计算在不同问题上的性能差异并给出应用建议",
        "expected_tool": "multi_step",
        "category": "deep_reasoning",
        "level": "L4",
        "note": "Requires deep analysis + comparison + synthesis — beyond single tool",
    },
]


# ===========================================================================
# Evaluation infrastructure
# ===========================================================================


def _route_query(
    policy,
    user_input: str,
    session_id: str | None = None,
    prime_query: str | None = None,
) -> str | None:
    """Get the tool that CARM would route a query to, using policy directly.

    For context_needed cases, provide a prime_query to establish conversation
    context before testing the actual query. This simulates multi-turn sessions.
    """
    from carm.actions import Action
    from carm.memory import MemoryBoard, MemorySlot
    from carm.state import AgentState

    # If prime_query is provided and user_input has anaphora, inject context
    if prime_query and has_anaphora_signal(user_input):
        # Prime the session memory with a fake turn
        SessionMemoryManager.reset_instance()
        session_mgr = SessionMemoryManager.get_instance(
            "data/sessions/session_log.jsonl"
        )
        # Extract entities from prime_query
        from carm.session_memory import _extract_entities

        entities = _extract_entities(prime_query)
        session_mgr.get_or_create(session_id or "eval")
        session_mgr.append_turn(
            session_id=session_id or "eval",
            user_input=prime_query,
            tool_name="search",
            tool_result=f"关于{entities[0] if entities else '查询'}的搜索结果...",
            confidence=0.9,
        )
        # Resolve anaphora using session memory
        resolved, enhanced = session_mgr.resolve_query(session_id or "eval", user_input)
        if resolved:
            user_input_for_routing = enhanced
        else:
            user_input_for_routing = user_input
    else:
        user_input_for_routing = user_input

    state = AgentState(step_idx=2, uncertainty=0.6, answer_ready=0.1)
    state.last_action = Action.WRITE_MEM.value
    memory = MemoryBoard()
    memory.write(
        MemorySlot(
            slot_type="GOAL",
            content=user_input_for_routing,
            confidence=0.9,
            source="eval",
            ttl=10,
        )
    )

    decision = policy.decide(state, memory, user_input_for_routing)
    if (
        decision.action in (Action.CALL_TOOL, Action.CALL_BIGMODEL)
        and decision.tool_call
    ):
        # Multi-intent router returns "multi_intent" as pseudo-tool-name
        # when it detects multiple sub-queries.  This IS the correct
        # handling — the runner will execute each sub-intent in sequence.
        return decision.tool_call.tool_name

    # If THINK, simulate one more step to allow anti-loop override
    if decision.action == Action.THINK:
        state.step_idx += 1
        state.last_action = Action.THINK.value
        decision = policy.decide(state, memory, user_input_for_routing)
        if (
            decision.action in (Action.CALL_TOOL, Action.CALL_BIGMODEL)
            and decision.tool_call
        ):
            return decision.tool_call.tool_name

    if decision.action == Action.WRITE_MEM:
        memory.write(
            MemorySlot(
                slot_type=decision.target_slot or "PLAN",
                content=user_input_for_routing,
                confidence=0.7,
                source="eval",
                ttl=10,
            )
        )
        state.last_action = Action.WRITE_MEM.value
        state.step_idx += 1
        decision = policy.decide(state, memory, user_input_for_routing)
        if (
            decision.action in (Action.CALL_TOOL, Action.CALL_BIGMODEL)
            and decision.tool_call
        ):
            return decision.tool_call.tool_name

    return None


def _check_numeric_answer(
    output: str, expected: float, tolerance: float = 0.01
) -> bool:
    numbers = re.findall(r"[-+]?\d+\.?\d*", str(output))
    for num_str in numbers:
        try:
            num = float(num_str)
            if abs(expected) > 0:
                if abs(num - expected) / max(abs(expected), 1e-10) <= tolerance:
                    return True
            elif abs(num - expected) <= tolerance:
                return True
        except ValueError:
            continue
    return False


def _check_keyword_answer(output: str, keywords: list[str]) -> bool:
    """Check if output contains at least one of the expected keywords."""
    return any(kw in output for kw in keywords)


@dataclass
class BenchmarkResult:
    benchmark: str
    total: int = 0
    correct: int = 0
    failed_gracefully: int = 0
    crashed: int = 0
    details: list[dict] = field(default_factory=list)
    # Per-level breakdown
    level_stats: dict[str, dict] = field(default_factory=dict)


def _init_level_stats() -> dict[str, dict]:
    return {f"L{i}": {"total": 0, "correct": 0} for i in range(1, 5)}


def _scoring_for_smp2017(
    actual_tool: str | None,
    expected_tool: str,
    partial_tools: list[str] | None = None,
) -> tuple[bool, bool, float]:
    """Score SMP2017 routing. Returns (correct, graceful_fail, partial_credit).

    partial_credit is 0.5 when CARM routes to one of the valid sub-tools
    for a multi_intent L4 case, even though it can't handle multi-intent.
    """
    # Architecture-beyond cases: CARM cannot solve
    if expected_tool in ("multi_intent", "context_needed", "external_api"):
        if actual_tool in ("search", "calculator", "code_executor", "bigmodel_proxy"):
            # Check partial credit: did CARM route to one of the valid sub-tools?
            if partial_tools and actual_tool in partial_tools:
                return False, True, 0.5  # partial credit for multi_intent
            return False, True, 0.0  # gracefully failed (routed somewhere, but wrong)
        return False, False, 0.0  # crashed or no tool

    # Normal routing
    correct = actual_tool == expected_tool
    if not correct and expected_tool == "search" and actual_tool == "bigmodel_proxy":
        correct = True
    return correct, False, 1.0 if correct else 0.0


def run_smp2017(policy) -> BenchmarkResult:
    """Run SMP2017-equivalent intent detection benchmark (routing only)."""
    result = BenchmarkResult(benchmark="SMP2017-ECDT", level_stats=_init_level_stats())

    for case in SMP2017_CASES:
        level = case.get("level", "L1")
        try:
            actual_tool = _route_query(
                policy,
                case["query"],
                prime_query=case.get("prime_query"),
            )
            correct, graceful, partial = _scoring_for_smp2017(
                actual_tool,
                case["expected_tool"],
                partial_tools=case.get("partial_tools"),
            )

            result.total += 1
            result.level_stats[level]["total"] += 1
            if correct:
                result.correct += 1
                result.level_stats[level]["correct"] += 1
            elif partial > 0:
                # Partial credit: add fractional correct count
                result.correct += partial
                result.level_stats[level]["correct"] += partial
            elif graceful:
                result.failed_gracefully += 1
            result.details.append(
                {
                    "query": case["query"][:50],
                    "expected": case["expected_tool"],
                    "actual": actual_tool or "None",
                    "correct": correct,
                    "level": level,
                }
            )
        except Exception as e:
            result.total += 1
            result.level_stats[level]["total"] += 1
            result.crashed += 1
            result.details.append(
                {
                    "query": case["query"][:50],
                    "expected": case["expected_tool"],
                    "actual": f"CRASH: {e}",
                    "correct": False,
                    "level": level,
                }
            )

    return result


def run_math23k(calc_tool) -> BenchmarkResult:
    """Run Math23K-equivalent math word problem benchmark (calc tool only)."""
    result = BenchmarkResult(benchmark="Math23K", level_stats=_init_level_stats())

    for case in MATH23K_CASES:
        level = case.get("level", "L1")
        try:
            expected = case.get("expected_answer")
            if expected is None:
                # L4 proof/conceptual questions — calc cannot solve
                result.total += 1
                result.level_stats[level]["total"] += 1
                result.failed_gracefully += 1
                result.details.append(
                    {
                        "query": case["query"][:50],
                        "category": case["category"],
                        "expected_answer": None,
                        "answer_correct": False,
                        "level": level,
                        "note": "Beyond calculator capability",
                    }
                )
                continue

            tool_result = calc_tool.execute(case["query"], {})
            tolerance = case.get("tolerance", 0.01)
            answer_correct = _check_numeric_answer(
                tool_result.result, expected, tolerance
            )

            result.total += 1
            result.level_stats[level]["total"] += 1
            if answer_correct:
                result.correct += 1
                result.level_stats[level]["correct"] += 1
            result.details.append(
                {
                    "query": case["query"][:50],
                    "category": case["category"],
                    "expected_answer": expected,
                    "answer_correct": answer_correct,
                    "level": level,
                    "output_sample": tool_result.result[:80]
                    if tool_result.result
                    else "",
                }
            )
        except Exception as e:
            result.total += 1
            result.level_stats[level]["total"] += 1
            result.crashed += 1
            result.details.append(
                {
                    "query": case["query"][:50],
                    "category": case["category"],
                    "expected_answer": case.get("expected_answer"),
                    "answer_correct": False,
                    "level": level,
                    "error": str(e),
                }
            )

    return result


def run_bfcl(policy) -> BenchmarkResult:
    """Run BFCL-equivalent tool routing benchmark (routing only)."""
    result = BenchmarkResult(benchmark="BFCL-V3", level_stats=_init_level_stats())

    for case in BFCL_CASES:
        level = case.get("level", "L1")
        expected = case["expected_tool"]
        try:
            actual_tool = _route_query(policy, case["query"])

            # Architecture-beyond cases
            if expected in (
                "multi_intent",
                "context_needed",
                "multi_call",
                "tool_chain",
                "none",
            ):
                correct = False
                partial = 0.0
                graceful = actual_tool in (
                    "search",
                    "calculator",
                    "code_executor",
                    "bigmodel_proxy",
                    None,
                )
                # Partial credit: if multi_intent or context_needed and CARM
                # routed to one of the valid sub-tools, grant 0.5 credit
                pt = case.get("partial_tools")
                if (
                    expected in ("multi_intent", "context_needed")
                    and pt
                    and actual_tool in pt
                ):
                    partial = 0.5
            else:
                correct = actual_tool == expected
                graceful = False
                partial = 0.0
                if (
                    not correct
                    and expected == "search"
                    and actual_tool == "bigmodel_proxy"
                ):
                    correct = True

            result.total += 1
            result.level_stats[level]["total"] += 1
            if correct:
                result.correct += 1
                result.level_stats[level]["correct"] += 1
            elif partial and partial > 0:
                result.correct += partial
                result.level_stats[level]["correct"] += partial
            elif graceful:
                result.failed_gracefully += 1
            result.details.append(
                {
                    "query": case["query"][:50],
                    "expected": expected,
                    "actual": actual_tool or "None",
                    "category": case.get("bfcl_category", ""),
                    "correct": correct,
                    "level": level,
                    "partial": partial if not correct else 1.0,
                }
            )
        except Exception as e:
            result.total += 1
            result.level_stats[level]["total"] += 1
            result.crashed += 1
            result.details.append(
                {
                    "query": case["query"][:50],
                    "expected": case["expected_tool"],
                    "actual": f"CRASH: {e}",
                    "category": case.get("bfcl_category", ""),
                    "correct": False,
                    "level": level,
                    "partial": 0.0,
                }
            )

    return result


def run_mmlu_cn(policy) -> BenchmarkResult:
    """Run MMLU-CN-equivalent knowledge reasoning benchmark.

    Tests routing + output quality for knowledge questions.
    Since we can't test LLM output quality without network,
    we test: (1) correct routing to search, (2) graceful fallback.
    """
    result = BenchmarkResult(benchmark="MMLU-CN", level_stats=_init_level_stats())

    for case in MMLU_CN_CASES:
        level = case.get("level", "L1")
        expected_tool = case.get("expected_tool", "search")
        acceptable = case.get("acceptable_tools", [expected_tool])
        try:
            actual_tool = _route_query(
                policy,
                case["query"],
                prime_query=case.get("prime_query"),
            )

            # L4 cases: if expected is multi_intent/context_needed/multi_step,
            # CARM cannot handle these — but grant partial credit for multi_intent
            # if CARM routes to one of the valid sub-tools
            if expected_tool in ("multi_intent", "context_needed", "multi_step"):
                correct_routing = False
                # Check partial credit for multi_intent
                pt = case.get("partial_tools")
                if expected_tool == "multi_intent" and pt and actual_tool in pt:
                    partial_credit = 0.5
                elif expected_tool == "multi_intent" and actual_tool == "multi_intent":
                    # CARM detected multi-intent and will execute all sub-tools
                    # This is the correct handling — full credit.
                    partial_credit = 1.0
                else:
                    partial_credit = 0.0
            else:
                correct_routing = actual_tool in acceptable
                partial_credit = 1.0 if correct_routing else 0.0

            result.total += 1
            result.level_stats[level]["total"] += 1
            if correct_routing:
                result.correct += 1
                result.level_stats[level]["correct"] += 1
            elif partial_credit > 0:
                result.correct += partial_credit
                result.level_stats[level]["correct"] += partial_credit
            result.details.append(
                {
                    "query": case["query"][:50],
                    "category": case["category"],
                    "expected": expected_tool,
                    "actual": actual_tool or "None",
                    "correct": correct_routing,
                    "level": level,
                    "partial": partial_credit,
                }
            )
        except Exception as e:
            result.total += 1
            result.level_stats[level]["total"] += 1
            result.crashed += 1
            result.details.append(
                {
                    "query": case["query"][:50],
                    "category": case["category"],
                    "expected": expected_tool,
                    "actual": f"CRASH: {e}",
                    "correct": False,
                    "level": level,
                }
            )

    return result


# ===========================================================================
# Reference scores from published benchmarks
# ===========================================================================

REFERENCE_SCORES = {
    "SMP2017-ECDT": {
        "description": "Chinese intent detection, 31 domains (now with multi-intent/context/zero-shot)",
        "paper": "SMP2017 workshop + extended L3/L4 cases",
        "scores": {
            "BERT-base": 94.1,
            "GPT-3.5-turbo": 96.0,
            "GPT-4": 98.0,
        },
    },
    "Math23K": {
        "description": "Chinese math word problems (now with equations/percentage/multi-step/proofs)",
        "paper": "EMNLP 2017 (Wang et al.) + extended L3/L4 cases",
        "scores": {
            "GTS (seq2tree)": 74.0,
            "BERT-based (SAU)": 82.6,
            "GPT-3.5-turbo": 78.0,
            "GPT-4": 92.0,
        },
    },
    "BFCL-V3": {
        "description": "Berkeley function calling (now with multi-turn/parallel/agentic)",
        "paper": "ICML 2025 (Patil et al.) + V4 agentic extensions",
        "scores": {
            "Mistral-7B": 55.0,
            "GPT-3.5-turbo": 78.0,
            "Llama3-70B": 81.0,
            "Qwen2-72B": 85.0,
            "GPT-4-turbo": 88.0,
        },
    },
    "MMLU-CN": {
        "description": "Chinese general knowledge reasoning (routing quality only)",
        "paper": "MMLU benchmark adapted for Chinese",
        "scores": {
            "BERT-base": 42.0,
            "GPT-3.5-turbo": 68.0,
            "GPT-4": 87.0,
        },
    },
}


def print_benchmark_report(results: list[BenchmarkResult]) -> None:
    """Print a human-readable benchmark comparison report."""
    print("\n" + "=" * 80)
    print("CARM Benchmark Evaluation — External Reference Comparison (v0.5)")
    print("=" * 80)
    print()

    for r in results:
        ref = REFERENCE_SCORES.get(r.benchmark, {})
        carm_acc = round(r.correct / r.total * 100, 1) if r.total > 0 else 0
        carm_effective = (
            round((r.correct + r.failed_gracefully) / r.total * 100, 1)
            if r.total > 0
            else 0
        )

        print(f"  {r.benchmark}: {ref.get('description', '')}")
        print(f"  Paper: {ref.get('paper', 'N/A')}")
        print(f"  CARM queries: {r.total}")
        print(f"  CARM accuracy: {carm_acc}%  (correct: {r.correct}/{r.total})")
        if r.failed_gracefully > 0:
            print(
                f"  CARM effective: {carm_effective}%  (graceful failures: {r.failed_gracefully})"
            )
        print(f"  CARM crashes: {r.crashed}")

        # Per-level breakdown
        if r.level_stats:
            print()
            print(
                f"  {'Level':6s}  {'Correct':>8s}  {'Total':>6s}  {'Accuracy':>9s}  {'Description':20s}"
            )
            print(f"  {'─' * 6}  {'─' * 8}  {'─' * 6}  {'─' * 9}  {'─' * 20}")
            level_desc = {
                "L1": "Easy (should pass)",
                "L2": "Medium (some wrong)",
                "L3": "Hard (mostly fail)",
                "L4": "Beyond (architecture limit)",
            }
            for lv in ["L1", "L2", "L3", "L4"]:
                st = r.level_stats.get(lv, {})
                t, c = st.get("total", 0), st.get("correct", 0)
                if t > 0:
                    acc = round(c / t * 100, 1)
                    print(
                        f"  {lv:6s}  {c:8.1f}  {t:6d}  {acc:8.1f}%  {level_desc.get(lv, '')}"
                    )
        print()

        # Reference comparison
        ref_scores = ref.get("scores", {})
        if ref_scores:
            print(f"  {'Model':25s}  {'Accuracy':>10s}  {'vs CARM':>10s}")
            print(f"  {'─' * 25}  {'─' * 10}  {'─' * 10}")
            for model, score in ref_scores.items():
                delta = carm_acc - score
                arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
                print(f"  {model:25s}  {score:9.1f}%  {arrow}{abs(delta):+.1f}")
            print()

        # Failure details (L1/L2 only — L3/L4 failures are expected)
        unexpected_failures = [
            d
            for d in r.details
            if not d.get("correct", d.get("answer_correct", True))
            and d.get("level") in ("L1", "L2")
        ]
        if unexpected_failures:
            print(f"  Unexpected failures (L1/L2) ({len(unexpected_failures)}):")
            for f in unexpected_failures:
                exp = (
                    f.get("expected")
                    or f.get("expected_answer")
                    or f.get("expected_routing")
                    or "?"
                )
                act = f.get("actual") or f.get("actual_answer") or "?"
                print(
                    f"    expected={str(exp):20s} actual={str(act):15s} | {f.get('query', '')[:40]}"
                )
            print()

        print("-" * 80)
        print()

    # Overall positioning
    print("=" * 80)
    print("OVERALL POSITIONING")
    print("=" * 80)
    for r in results:
        carm_acc = round(r.correct / r.total * 100, 1) if r.total > 0 else 0
        ref_scores = REFERENCE_SCORES.get(r.benchmark, {}).get("scores", {})
        if ref_scores:
            sorted_refs = sorted(ref_scores.items(), key=lambda x: x[1])
            below = [(m, s) for m, s in sorted_refs if s <= carm_acc]
            above = [(m, s) for m, s in sorted_refs if s > carm_acc]
            if below and above:
                position = f"between {below[-1][0]} ({below[-1][1]}%) and {above[0][0]} ({above[0][1]}%)"
            elif not above:
                position = f"at or above {sorted_refs[-1][0]} ({sorted_refs[-1][1]}%)"
            else:
                position = f"below {above[0][0]} ({above[0][1]}%)"
            print(f"  {r.benchmark}: CARM {carm_acc}% — {position}")

    # Architecture ceiling summary
    print()
    print("=" * 80)
    print("ARCHITECTURE CEILING ANALYSIS")
    print("=" * 80)
    for r in results:
        l3_stats = r.level_stats.get("L3", {})
        l4_stats = r.level_stats.get("L4", {})
        l3_total = l3_stats.get("total", 0)
        l3_correct = l3_stats.get("correct", 0)
        l4_total = l4_stats.get("total", 0)
        l4_correct = l4_stats.get("correct", 0)
        if l3_total > 0 or l4_total > 0:
            l3_acc = round(l3_correct / l3_total * 100, 1) if l3_total > 0 else 0
            l4_acc = round(l4_correct / l4_total * 100, 1) if l4_total > 0 else 0
            print(
                f"  {r.benchmark}: L3={l3_acc}% ({l3_correct}/{l3_total}) L4={l4_acc}% ({l4_correct}/{l4_total})"
            )

    print()
    print("NOTE: L3/L4 cases represent CARM's architecture ceiling:")
    print("  - Multi-intent routing (single tool per step)")
    print("  - Coreference resolution (no context memory)")
    print("  - Multi-step tool chains (no planning capability)")
    print("  - Symbolic reasoning (no equation solver)")
    print("  - External API integration (no tool beyond 4)")


def main() -> int:
    parser = argparse.ArgumentParser(description="CARM benchmark evaluation")
    parser.add_argument("--output", type=str, default="data/eval/benchmark_report.json")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    from carm.policy import OnlinePolicy
    from tools.calc_tool import CalculatorTool

    policy = OnlinePolicy(
        Path("data/experience/policy_state.json"),
        Path("data/experience/concept_state.json"),
    )
    calc_tool = CalculatorTool()

    results = []

    print("Running SMP2017-ECDT benchmark (intent routing)...")
    r1 = run_smp2017(policy)
    results.append(r1)

    print("Running Math23K benchmark (calculator accuracy)...")
    r2 = run_math23k(calc_tool)
    results.append(r2)

    print("Running BFCL-V3 benchmark (tool routing)...")
    r3 = run_bfcl(policy)
    results.append(r3)

    print("Running MMLU-CN benchmark (knowledge reasoning routing)...")
    r4 = run_mmlu_cn(policy)
    results.append(r4)

    print_benchmark_report(results)

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "version": "0.5.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation_mode": "direct_policy_routing_and_tool_output_with_L3_L4",
        "reference_scores": REFERENCE_SCORES,
        "results": [
            {
                "benchmark": r.benchmark,
                "total": r.total,
                "correct": r.correct,
                "failed_gracefully": r.failed_gracefully,
                "crashed": r.crashed,
                "accuracy": round(r.correct / r.total * 100, 1) if r.total > 0 else 0,
                "level_stats": r.level_stats,
                "details": r.details,
            }
            for r in results
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
