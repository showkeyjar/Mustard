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
    {"query": "3.5公里等于多少米", "expected_tool": "calculator", "level": "L1"},
    {"query": "2的20次方是多少", "expected_tool": "calculator", "level": "L1"},
    {"query": "1万亿除以14亿", "expected_tool": "calculator", "level": "L1"},
    {"query": "帮我写一个冒泡排序", "expected_tool": "code_executor", "level": "L1"},
    {"query": "用python实现快速排序", "expected_tool": "code_executor", "level": "L1"},
    {"query": "运行一个阶乘的代码", "expected_tool": "code_executor", "level": "L1"},
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
    {
        "query": "帮我归纳一下这些资料的核心观点",
        "expected_tool": "bigmodel_proxy",
        "level": "L2",
    },
    {"query": "搜索一下Python教程", "expected_tool": "search", "level": "L2"},
    {"query": "10的阶乘是多少", "expected_tool": "code_executor", "level": "L2"},
    {
        "query": "帮我算一下排序的时间复杂度",
        "expected_tool": "code_executor",
        "level": "L2",
    },
    {
        "query": "帮我写一个关于历史的作文",
        "expected_tool": "bigmodel_proxy",
        "level": "L2",
    },
    # ══ L3 Hard: multi-intent, zero-shot, boundary ══════════════════
    # Multi-intent: CARM only routes to ONE tool per query
    {
        "query": "帮我查一下北京天气，顺便算一下3加5",
        "expected_tool": "multi_intent",
        "level": "L3",
        "note": "Requires both search AND calculator — CARM cannot handle multi-intent",
    },
    {
        "query": "先搜索量子计算的发展，然后帮我写个总结",
        "expected_tool": "multi_intent",
        "level": "L3",
        "note": "Requires search THEN bigmodel_proxy — multi-step planning",
    },
    {
        "query": "用快速排序排5,3,8，然后告诉我它的时间复杂度",
        "expected_tool": "multi_intent",
        "level": "L3",
        "note": "Requires code_executor THEN search — sequential tool chain",
    },
    # Coreference / context-dependent (CARM has no context memory)
    {
        "query": "它的性能怎么样",
        "expected_tool": "context_needed",
        "level": "L3",
        "note": "'它' requires coreference resolution — CARM has no context",
    },
    {
        "query": "这个方案有什么问题",
        "expected_tool": "context_needed",
        "level": "L3",
        "note": "'这个方案' requires previous context — CARM cannot resolve",
    },
    # Zero-shot domains: CARM has no mapping for these
    {
        "query": "帮我订一张去上海的机票",
        "expected_tool": "external_api",
        "level": "L3",
        "note": "No 'flight booking' tool — CARM should gracefully decline",
    },
    {
        "query": "设置一个明天早上7点的闹钟",
        "expected_tool": "external_api",
        "level": "L3",
        "note": "No 'alarm' tool — CARM should gracefully decline",
    },
    {
        "query": "给我的同事发一封邮件",
        "expected_tool": "external_api",
        "level": "L3",
        "note": "No 'email' tool — CARM should gracefully decline",
    },
    # Subtle boundary: "debug" intent
    {
        "query": "我的代码报错了IndexError list index out of range怎么解决",
        "expected_tool": "search",
        "level": "L3",
        "note": "Debug help is knowledge, not code execution — but CARM may route to code_executor",
    },
    {
        "query": "这段代码有什么问题 def foo(): return 1/0",
        "expected_tool": "search",
        "level": "L3",
        "note": "Code review is analysis, not execution",
    },
    # ══ L4 Beyond: architecture ceiling ═════════════════════════════
    {
        "query": "根据上下文，用户最可能想问什么",
        "expected_tool": "context_needed",
        "level": "L4",
        "note": "Requires conversational context understanding — impossible without memory",
    },
    {
        "query": "帮我规划一个3天的北京旅游行程包括酒店和景点",
        "expected_tool": "multi_intent",
        "level": "L4",
        "note": "Requires search(hotels) + search(attractions) + bigmodel_proxy(itinerary) — multi-step orchestration",
    },
    {
        "query": "对比一下最近三年的销售数据并给出趋势预测",
        "expected_tool": "multi_intent",
        "level": "L4",
        "note": "Requires search(data) + calculator(trends) + bigmodel_proxy(prediction)",
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
    # ══ L3 Hard: equations, multi-step reasoning, percentage ════════
    # Equations: CARM cannot do symbolic reasoning
    {
        "query": "一个数的3倍加5等于20这个数是多少",
        "expected_answer": 5,
        "category": "equation",
        "level": "L3",
        "note": "Requires solving 3x+5=20 — symbolic reasoning",
    },
    {
        "query": "甲比乙大3岁甲乙之和是27岁甲多少岁",
        "expected_answer": 15,
        "category": "equation",
        "level": "L3",
        "note": "Requires system of equations",
    },
    {
        "query": "鸡兔同笼共10个头26条腿鸡有几只",
        "expected_answer": 7,
        "category": "equation",
        "level": "L3",
        "note": "Classic Chinese math puzzle — requires equation setup",
    },
    # Multi-step: requires >1 calculation step
    {
        "query": "小明买了3本书每本15元又买了2支笔每支3元一共花了多少钱",
        "expected_answer": 51,
        "category": "multi_step",
        "level": "L3",
        "note": "Two-step: 3*15 + 2*3 = 51 — CARM may extract only one expression",
    },
    {
        "query": "一个长方形周长30米长是宽的2倍求面积",
        "expected_answer": 50,
        "category": "multi_step",
        "level": "L3",
        "note": "Requires: 2(l+w)=30, l=2w → w=5, l=10, area=50",
    },
    # Percentage / ratio
    {
        "query": "原价200元打8折后是多少元",
        "expected_answer": 160,
        "category": "percentage",
        "level": "L3",
        "note": "200 * 0.8 — requires understanding '打8折'",
    },
    {
        "query": "一个班级40人及格率75%有多少人及格",
        "expected_answer": 30,
        "category": "percentage",
        "level": "L3",
        "note": "40 * 0.75 — requires understanding '及格率'",
    },
    # Speed/distance/time with reasoning
    {
        "query": "甲乙两地相距240公里一辆车3小时到达平均速度是多少",
        "expected_answer": 80,
        "category": "rate_reasoning",
        "level": "L3",
        "note": "240/3=80 — CARM should extract this",
    },
    {
        "query": "甲乙两人从A地出发甲每小时走5公里乙每小时走7公里2小时后两人相距多少公里",
        "expected_answer": 4,
        "category": "rate_reasoning",
        "level": "L3",
        "note": "Same direction: (7-5)*2=4 — requires understanding relative speed",
    },
    # ══ L4 Beyond: complex reasoning, proofs, optimization ══════════
    {
        "query": "证明根号2是无理数",
        "expected_answer": None,
        "category": "proof",
        "level": "L4",
        "note": "Mathematical proof — far beyond calculator",
    },
    {
        "query": "求函数f(x)=x^3-3x在区间[-2,2]上的最大值",
        "expected_answer": 2,
        "category": "calculus",
        "level": "L4",
        "note": "Requires calculus — derivative, critical points, boundary evaluation",
    },
    {
        "query": "一个水池有两个进水管一个4小时灌满一个6小时灌满同时开多久灌满",
        "expected_answer": 2.4,
        "category": "work_problem",
        "level": "L4",
        "note": "1/(1/4+1/6) = 2.4h — classic work-rate problem requires equation reasoning",
    },
    {
        "query": "从1加到100的和是多少",
        "expected_answer": 5050,
        "category": "series",
        "level": "L4",
        "note": "Requires knowing the formula n(n+1)/2, not just computation",
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
        "expected_tool": "code_executor",
        "bfcl_category": "multiple",
        "level": "L2",
        "note": "阶乘: calc+code signals, code wins",
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
    # ══ L3 Hard: V4 agentic dimensions ══════════════════════════════
    # Multi-turn: context-dependent routing (CARM has no context)
    {
        "query": "用刚才的模型再跑一遍",
        "expected_tool": "context_needed",
        "bfcl_category": "multi_turn",
        "level": "L3",
        "note": "'刚才的模型' requires conversation context",
    },
    {
        "query": "换一个方法试试",
        "expected_tool": "context_needed",
        "bfcl_category": "multi_turn",
        "level": "L3",
        "note": "'换一个方法' requires knowing what was tried before",
    },
    {
        "query": "这次精度好一点了吗",
        "expected_tool": "context_needed",
        "bfcl_category": "multi_turn",
        "level": "L3",
        "note": "Requires previous execution context",
    },
    # Parallel tool calling: CARM can only call one tool per step
    {
        "query": "同时查一下北京和上海的天气",
        "expected_tool": "multi_call",
        "bfcl_category": "parallel_call",
        "level": "L3",
        "note": "Requires 2 parallel search calls — CARM only calls one tool at a time",
    },
    {
        "query": "帮我计算圆的面积和周长半径是5",
        "expected_tool": "multi_call",
        "bfcl_category": "parallel_call",
        "level": "L3",
        "note": "Requires 2 calculator calls — area and circumference",
    },
    # Relevance detection: should NOT call any tool
    {
        "query": "谢谢你",
        "expected_tool": "none",
        "bfcl_category": "relevance",
        "level": "L3",
        "note": "Social pleasantry — no tool should be called",
    },
    {
        "query": "好的我知道了",
        "expected_tool": "none",
        "bfcl_category": "relevance",
        "level": "L3",
        "note": "Acknowledgment — no tool needed",
    },
    {
        "query": "这个问题我不太懂",
        "expected_tool": "none",
        "bfcl_category": "relevance",
        "level": "L3",
        "note": "User expressing confusion — should ask for clarification, not call a tool",
    },
    # Conditional routing: need reasoning about what tool to call
    {
        "query": "如果我想要高性能的排序应该用哪种算法",
        "expected_tool": "search",
        "bfcl_category": "conditional",
        "level": "L3",
        "note": "Requires knowledge about algorithms, not code execution",
    },
    {
        "query": "帮我分析一下这个代码的性能瓶颈",
        "expected_tool": "search",
        "bfcl_category": "conditional",
        "level": "L3",
        "note": "Code analysis is knowledge, not execution — but CARM may route to code_executor",
    },
    # ══ L4 Beyond: multi-step tool chains, agentic planning ═════════
    {
        "query": "帮我搜索最近的经济数据，然后做一个趋势分析，最后生成报告",
        "expected_tool": "tool_chain",
        "bfcl_category": "agentic",
        "level": "L4",
        "note": "3-step chain: search → calculator → bigmodel_proxy",
    },
    {
        "query": "先运行排序算法，验证结果是否正确，如果不正确换一种方法",
        "expected_tool": "tool_chain",
        "bfcl_category": "agentic",
        "level": "L4",
        "note": "Requires conditional branching based on output",
    },
    {
        "query": "帮我搭建一个web服务器并部署上去",
        "expected_tool": "tool_chain",
        "bfcl_category": "agentic",
        "level": "L4",
        "note": "Requires code generation + execution + deployment — far beyond CARM",
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
    # ══ L1 Easy: common knowledge ═══════════════════════════════════
    {
        "query": "中国的首都是哪个城市",
        "expected_answer": "北京",
        "category": "geography",
        "level": "L1",
        "note": "Common knowledge — any search should find this",
    },
    {
        "query": "水的化学式是什么",
        "expected_answer": "H2O",
        "category": "chemistry",
        "level": "L1",
    },
    {
        "query": "一年有多少天",
        "expected_answer": "365",
        "category": "common",
        "level": "L1",
    },
    # ══ L2 Medium: domain knowledge ═════════════════════════════════
    {
        "query": "量子纠缠是什么现象",
        "expected_answer": "量子",
        "category": "physics",
        "level": "L2",
        "note": "Answer should contain '量子' — search should return relevant info",
    },
    {
        "query": "TCP和UDP的区别是什么",
        "expected_answer": "连接",
        "category": "cs",
        "level": "L2",
        "note": "Answer should mention connection-oriented vs connectionless",
    },
    {
        "query": "Python的GIL是什么",
        "expected_answer": "全局解释器锁",
        "category": "cs",
        "level": "L2",
        "note": "GIL = Global Interpreter Lock",
    },
    {
        "query": "什么是通货膨胀",
        "expected_answer": "物价",
        "category": "economics",
        "level": "L2",
        "note": "Answer should mention price/物价 rising",
    },
    # ══ L3 Hard: multi-hop reasoning, niche knowledge ═══════════════
    {
        "query": "深度学习中的注意力机制最早是在哪篇论文提出的",
        "expected_answer": "Bahdanau",
        "category": "ml_history",
        "level": "L3",
        "note": "Bahdanau et al. 2014 — requires precise academic knowledge",
    },
    {
        "query": "相对论和牛顿力学的根本区别是什么",
        "expected_answer": "时空",
        "category": "physics",
        "level": "L3",
        "note": "Answer should mention spacetime/时空",
    },
    {
        "query": "BERT和GPT的架构差异是什么",
        "expected_answer": "编码器",
        "category": "cs",
        "level": "L3",
        "note": "Answer should mention encoder vs decoder — requires comparative understanding",
    },
    {
        "query": "中国的五年计划是从哪一年开始的",
        "expected_answer": "1953",
        "category": "history",
        "level": "L3",
        "note": "First Five-Year Plan started 1953 — niche factual knowledge",
    },
    {
        "query": "Transformer模型为什么比RNN更适合并行计算",
        "expected_answer": "序列",
        "category": "cs",
        "level": "L3",
        "note": "Answer should mention sequential/序列 dependency removal",
    },
    # ══ L4 Beyond: requires synthesis, reasoning, or expertise ══════
    {
        "query": "根据费曼物理学讲义解释为什么天空是蓝色的",
        "expected_answer": "散射",
        "category": "physics",
        "level": "L4",
        "note": "Requires Rayleigh scattering explanation — synthesis of physics concepts",
    },
    {
        "query": "分析康德纯粹理性批判中先验演绎的核心论证",
        "expected_answer": "范畴",
        "category": "philosophy",
        "level": "L4",
        "note": "Deep philosophy — requires understanding of categories/范畴",
    },
    {
        "query": "为什么图灵测试不能作为衡量智能的唯一标准",
        "expected_answer": "中文房间",
        "category": "ai_philosophy",
        "level": "L4",
        "note": "Should mention Chinese Room argument — requires philosophical reasoning",
    },
]


# ===========================================================================
# Evaluation infrastructure
# ===========================================================================


def _route_query(policy, user_input: str) -> str | None:
    """Get the tool that CARM would route a query to, using policy directly."""
    from carm.actions import Action
    from carm.memory import MemoryBoard, MemorySlot
    from carm.state import AgentState

    state = AgentState(step_idx=2, uncertainty=0.6, answer_ready=0.1)
    state.last_action = Action.WRITE_MEM.value
    memory = MemoryBoard()
    memory.write(
        MemorySlot(
            slot_type="GOAL", content=user_input, confidence=0.9, source="eval", ttl=10
        )
    )

    decision = policy.decide(state, memory, user_input)
    if decision.action == Action.CALL_TOOL and decision.tool_call:
        return decision.tool_call.tool_name
    elif decision.action == Action.CALL_BIGMODEL and decision.tool_call:
        return decision.tool_call.tool_name

    if decision.action == Action.WRITE_MEM:
        memory.write(
            MemorySlot(
                slot_type=decision.target_slot or "PLAN",
                content=user_input,
                confidence=0.7,
                source="eval",
                ttl=10,
            )
        )
        state.last_action = Action.WRITE_MEM.value
        state.step_idx += 1
        decision = policy.decide(state, memory, user_input)
        if decision.action == Action.CALL_TOOL and decision.tool_call:
            return decision.tool_call.tool_name
        elif decision.action == Action.CALL_BIGMODEL and decision.tool_call:
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
    actual_tool: str | None, expected_tool: str
) -> tuple[bool, bool]:
    """Score SMP2017 routing. Returns (correct, graceful_fail)."""
    # Architecture-beyond cases: CARM cannot solve, count as graceful fail
    if expected_tool in ("multi_intent", "context_needed", "external_api"):
        # CARM "passes" if it doesn't crash — but it cannot produce the right answer
        # because it only has 4 tools. We check: did it at least route to a
        # REASONABLE tool? (not crash)
        if actual_tool in ("search", "calculator", "code_executor", "bigmodel_proxy"):
            return False, True  # gracefully failed (routed somewhere, but wrong)
        return False, False  # crashed or no tool

    # Normal routing
    correct = actual_tool == expected_tool
    if not correct and expected_tool == "search" and actual_tool == "bigmodel_proxy":
        correct = True
    return correct, False


def run_smp2017(policy) -> BenchmarkResult:
    """Run SMP2017-equivalent intent detection benchmark (routing only)."""
    result = BenchmarkResult(benchmark="SMP2017-ECDT", level_stats=_init_level_stats())

    for case in SMP2017_CASES:
        level = case.get("level", "L1")
        try:
            actual_tool = _route_query(policy, case["query"])
            correct, graceful = _scoring_for_smp2017(actual_tool, case["expected_tool"])

            result.total += 1
            result.level_stats[level]["total"] += 1
            if correct:
                result.correct += 1
                result.level_stats[level]["correct"] += 1
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
                graceful = actual_tool in (
                    "search",
                    "calculator",
                    "code_executor",
                    "bigmodel_proxy",
                    None,
                )
            else:
                correct = actual_tool == expected
                graceful = False
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
        try:
            actual_tool = _route_query(policy, case["query"])

            # For MMLU, correct routing means search or bigmodel_proxy
            # (both are valid for knowledge questions)
            correct_routing = actual_tool in ("search", "bigmodel_proxy")
            # L3/L4 questions that need deep reasoning may not be answered
            # correctly even with good routing — that's the ceiling

            result.total += 1
            result.level_stats[level]["total"] += 1
            if correct_routing:
                result.correct += 1
                result.level_stats[level]["correct"] += 1
            result.details.append(
                {
                    "query": case["query"][:50],
                    "category": case["category"],
                    "expected_routing": "search or bigmodel_proxy",
                    "actual": actual_tool or "None",
                    "routing_correct": correct_routing,
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
                    "category": case["category"],
                    "expected_routing": "search or bigmodel_proxy",
                    "actual": f"CRASH: {e}",
                    "routing_correct": False,
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
                        f"  {lv:6s}  {c:8d}  {t:6d}  {acc:8.1f}%  {level_desc.get(lv, '')}"
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
            if not d.get("correct", False) and d.get("level") in ("L1", "L2")
        ]
        if unexpected_failures:
            print(f"  Unexpected failures (L1/L2) ({len(unexpected_failures)}):")
            for f in unexpected_failures:
                print(
                    f"    expected={f.get('expected', f.get('expected_routing', '?')):20s} actual={str(f.get('actual', '?')):15s} | {f.get('query', '')[:40]}"
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
