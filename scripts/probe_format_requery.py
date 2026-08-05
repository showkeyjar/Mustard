#!/usr/bin/env python3
"""实测（不是估计）：定向重问补格式，模型到底能补对多少？

背景：`diag_documented_format.py` 给出一个代理指标——模型自发按格式输出时
命中率 96.9%。**那个数字不能用**：那 295 例是模型有把握才自发加的后缀，
不合规的 89 例恰恰是它没把握的那批。用有把握样本估计干预成功率是幸存者偏差。

所以这里不做离线镜像，直接对真实模型发同样的重问 prompt，逐例比对 GT。
89 次调用换一个真数，比任何模拟都便宜。

prompt 里只给三样东西，全部来自现场，不含任何我们自己编的答案：
  1. 用户 query 原文
  2. 该参数 description 原文（格式要求就写在里面）
  3. 模型当前给出的值

用法:
    python scripts/probe_format_requery.py v22
    python scripts/probe_format_requery.py v22 --limit 10     # 先小样试跑
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag_documented_format import (      # noqa: E402
    CATEGORIES, declares_comma_format, gt_values_for, lit, shape,
)

DIAG = Path(__file__).resolve().parent.parent / "data" / "eval" / "diag"
OLLAMA_BASE_URL = "http://192.168.31.20:11434"
OLLAMA_MODEL = "qwen3-coder"

PROMPT = """The parameter `{param}` of function `{func}` is documented as:

{desc}

User request: {query}

The value currently produced for `{param}` is: {value!r}

Rewrite that value so it satisfies the documented format. Use only information
from the user request and the documentation above. Reply with the value alone,
no quotes, no explanation, no extra words."""


def collect(tag: str) -> list[dict]:
    """导出所有'声明了格式但生成值不合规'的实例。"""
    from eval_bfcl_v4_fast import build_messages, load_bfcl_data

    out = []
    for cat in CATEGORIES:
        path = DIAG / f"{cat}_{tag}.jsonl"
        if not path.exists():
            continue
        fmap_by_id = {}
        try:
            for item in load_bfcl_data(cat):
                _, fs = build_messages(item)
                fmap_by_id[item.get("id", "")] = {f.get("name"): f for f in fs
                                                  if isinstance(f, dict)}
        except Exception:
            continue
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            pred = lit(row.get("pred")) or []
            gt = lit(row.get("gt")) or []
            fmap = fmap_by_id.get(row.get("id", ""), {})
            for call in pred if isinstance(pred, list) else []:
                if not isinstance(call, dict):
                    continue
                fname = call.get("name")
                props = ((fmap.get(fname) or {}).get("parameters") or {}).get("properties") or {}
                for pname, pv in (call.get("arguments") or {}).items():
                    if not isinstance(pv, str):
                        continue
                    spec = props.get(pname) or {}
                    ok, _ = declares_comma_format(spec)
                    if not ok or shape(pv).startswith("COMMA"):
                        continue
                    desc = str(spec.get("description") or "")
                    items = spec.get("items")
                    if isinstance(items, dict):
                        desc += " " + str(items.get("description") or "")
                    gtv = gt_values_for(gt, fname, pname)
                    out.append({
                        "id": row.get("id"), "cat": cat, "func": fname,
                        "param": pname, "pred": pv, "gt": gtv,
                        "desc": desc.strip(), "query": row.get("query") or "",
                        "correct": str(row.get("correct")) == "True",
                    })
    return out


def ask(prompt: str) -> str:
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={"model": OLLAMA_MODEL,
              "messages": [{"role": "user", "content": prompt}],
              "stream": False,
              "options": {"temperature": 0.0}},
        timeout=300)
    r.raise_for_status()
    txt = (r.json().get("message") or {}).get("content") or ""
    # 模型偶尔会加思考段或引号，取最后一行非空内容并剥掉包裹符号
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
    lines = [ln.strip() for ln in txt.strip().splitlines() if ln.strip()]
    val = lines[-1] if lines else ""
    return val.strip().strip("`").strip("'\"").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cases = collect(args.tag)
    if args.limit:
        cases = cases[:args.limit]
    print(f"待重问实例 {len(cases)} 个\n")

    stat = collections.Counter()
    results = []
    t0 = time.time()
    for i, c in enumerate(cases, 1):
        try:
            new = ask(PROMPT.format(param=c["param"], func=c["func"],
                                    desc=c["desc"], query=c["query"],
                                    value=c["pred"]))
        except Exception as exc:
            print(f"  [{i}/{len(cases)}] {c['id']} 调用失败: {exc}")
            stat["error"] += 1
            continue

        gtv = c["gt"]
        if gtv is None:
            # gt 里没这个参数：改不改都不影响判定（参数多余本就判错）
            verdict = "no_gt"
        elif any(new == g for g in gtv):
            verdict = "now_hit"
        elif any(c["pred"] == g for g in gtv):
            verdict = "broke"          # 原来就对，重问改坏了
        else:
            verdict = "still_miss"
        stat[verdict] += 1
        c2 = dict(c, requery=new, verdict=verdict)
        results.append(c2)
        flag = {"now_hit": "+", "broke": "-", "still_miss": ".",
                "no_gt": "o"}[verdict]
        print(f"  [{i}/{len(cases)}] {flag} {c['id']:<26} "
              f"{c['pred']!r} -> {new!r}   gt={gtv!r}")

    dt = time.time() - t0
    n = sum(stat.values())
    print(f"\n===== 实测结果（{n} 例，{dt / 60:.1f} 分钟）=====")
    for k in ("now_hit", "still_miss", "broke", "no_gt", "error"):
        if stat[k]:
            print(f"  {k:<12} {stat[k]:>4}")

    scored = stat["now_hit"] + stat["still_miss"] + stat["broke"]
    if scored:
        print(f"\n  重问命中率 = {stat['now_hit']}/{scored} = "
              f"{stat['now_hit'] / scored:.1%}")

    # 只有"原本判错的样本被翻正"才是真收益；样本级去重，
    # 因为一个样本可能有多个参数被改，全对才算翻正。
    by_sample = collections.defaultdict(list)
    for r in results:
        by_sample[r["id"]].append(r)
    gain = loss = 0
    for sid, rs in by_sample.items():
        was = rs[0]["correct"]
        # 该样本涉及的所有重问参数都命中，才可能翻正
        all_hit = all(r["verdict"] in ("now_hit", "no_gt") for r in rs)
        any_broke = any(r["verdict"] == "broke" for r in rs)
        if not was and all_hit and any(r["verdict"] == "now_hit" for r in rs):
            gain += 1
        if was and any_broke:
            loss += 1
    print(f"\n  样本级（仅这些参数看）: 可能翻正 {gain} / 确定改坏 {loss}")
    print("  注意：'可能翻正'仍需该样本其他参数与函数集合本就正确，")
    print("        这是上界不是实得——真实收益要看上线后的逐样本对账。")

    out = Path(args.out or DIAG / f"format_requery_{args.tag}.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n  明细已写入 {out}")


if __name__ == "__main__":
    main()
