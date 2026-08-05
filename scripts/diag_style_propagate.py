#!/usr/bin/env python3
"""Change I 候选：list 参数内部的风格一致性传播。—— 已否决，勿部署。

============================ REJECTED 2026-08-05 ============================
v21 实测：GAINED 0 / LOST 0 / 9 neutral。假设不成立，不要再拿这条线立项。

为什么不成立：BFCL 的 gt 值本身在同一个 list 里就是混合风格的
（['Opening hours', 'ticket prices'] 这种 gt 真实存在），所以"schema 词表命中项
的风格可以外推给其他项"这个前提是错的。

顺带暴露一个实现缺陷（保留在下面代码里当反例）：单词型元素（'humidity'）
推断不出分隔符，apply_style 用空格兜底，结果把
['temperature_high','temperature_low','humidity','precipitation']
反向改成了空格分隔（simple_python_333、parallel_17）。判分虽未变差，
但方向是错的 —— 说明 detect_style 在无分隔符样本上不可信。

结论：参数值这条线上，格式规范化的空间已经被 Change H 吃干净了。
剩下的 arg_wrong_value 主要是语义失配（semantic 55 / unreachable 72），
不是书写风格问题。
=============================================================================


动机
----
Change H 只在"去掉内部分隔符和大小写后完全相等"时才吸附。结果在 list 参数上
经常只吸附了一部分元素：
    ['working hours', 'ticket price'] -> ['working hours', 'ticket_price']
    ['protein', 'calories', 'carbs']  -> ['Protein', 'Calories', 'carbs']
既然同一个 list 里已经有元素确认了 schema 的书写风格（分隔符 + 大小写），
把这个风格套到剩余元素上是从 **schema** 推断的，不是从 gt 泄漏。

这个脚本只量化机会大小，不改生产代码。
和 diag_vocab_snap.py 一样内建保真度自检：离线判分复现不了线上判定的样本
一律排除出结论。

用法:
    python scripts/diag_style_propagate.py v21
    python scripts/diag_style_propagate.py v21 --show gained
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag_vocab_snap import (  # noqa: E402
    CATEGORIES,
    judged_correct,
    norm,
    param_vocab,
    snap_value,
)

DIAG = Path(__file__).resolve().parent.parent / "data" / "eval" / "diag"


def detect_style(s: str):
    """从一个已确认属于 schema 的词表项推断书写风格。

    返回 (sep, case)：
      sep  — 词间分隔符，'_' / '-' / ' ' / ''（单词就没有分隔符，返回 None）
      case — 'lower' / 'upper' / 'title' / 'title_first' / None（无法判断）
    """
    sep = None
    for cand in ("_", "-", " "):
        if cand in s:
            sep = cand
            break
    words = re.split(r"[_\- ]+", s) if sep else [s]
    words = [w for w in words if w]
    if not words:
        return None, None

    def wcase(w):
        if w.isupper() and len(w) > 1:
            return "upper"
        if w[:1].isupper() and w[1:].islower():
            return "title"
        if w.islower():
            return "lower"
        return "mixed"

    cases = [wcase(w) for w in words]
    if any(c == "mixed" for c in cases):
        return sep, None
    if all(c == "lower" for c in cases):
        return sep, "lower"
    if all(c == "upper" for c in cases):
        return sep, "upper"
    if all(c == "title" for c in cases):
        return sep, "title"
    if cases[0] == "title" and all(c == "lower" for c in cases[1:]):
        return sep, "title_first"
    return sep, None


def apply_style(s: str, sep, case):
    """把推断出的风格套到一个未吸附的值上。"""
    if sep is None and case is None:
        return s
    words = [w for w in re.split(r"[_\- ]+", str(s)) if w]
    if not words:
        return s
    if case == "lower":
        words = [w.lower() for w in words]
    elif case == "upper":
        words = [w.upper() for w in words]
    elif case == "title":
        words = [w[:1].upper() + w[1:].lower() for w in words]
    elif case == "title_first":
        words = [words[0][:1].upper() + words[0][1:].lower()] + [
            w.lower() for w in words[1:]
        ]
    use_sep = sep if sep is not None else ("" if len(words) == 1 else " ")
    return use_sep.join(words)


def snap_list_with_style(pv, vocab):
    """先做 H 的吸附，再把命中项的风格传播给未命中项。"""
    if not isinstance(pv, list) or not vocab:
        return snap_value(pv, vocab)
    snapped = snap_value(pv, vocab)
    if not isinstance(snapped, list):
        return snapped
    # 哪些元素被 H 真正吸附了（= 确认属于 schema 词表）
    styles = []
    for orig, new in zip(pv, snapped):
        if isinstance(new, str) and new != orig:
            styles.append(detect_style(new))
        elif isinstance(new, str) and new in vocab:
            styles.append(detect_style(new))
    if not styles:
        return snapped
    # 只有当所有确认项风格一致时才传播 —— 有分歧就不猜
    uniq = {s for s in styles if s != (None, None)}
    if len(uniq) != 1:
        return snapped
    sep, case = next(iter(uniq))
    out = []
    for orig, new in zip(pv, snapped):
        if not isinstance(new, str) or new != orig:
            out.append(new)
            continue
        out.append(apply_style(new, sep, case))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--show", choices=["gained", "lost", "touched"])
    ap.add_argument("--limit", type=int, default=25)
    args = ap.parse_args()

    from eval_bfcl_v4_fast import build_messages, load_bfcl_data

    gained, lost, touched = [], [], []
    per_cat = collections.defaultdict(lambda: [0, 0, 0])
    n_seen = n_touched = baseline_mismatch = 0

    for cat in CATEGORIES:
        path = DIAG / f"{cat}_{args.tag}.jsonl"
        if not path.exists():
            continue
        fmap_by_id = {}
        try:
            for item in load_bfcl_data(cat):
                _, fs = build_messages(item)
                fmap_by_id[item.get("id", "")] = {
                    f.get("name"): f for f in fs if isinstance(f, dict)
                }
        except Exception:
            continue

        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            pred = norm(row.get("pred")) or []
            gt = norm(row.get("gt")) or []
            if not pred:
                continue
            n_seen += 1
            was_correct = str(row.get("correct")) == "True"
            if judged_correct(pred, gt) != was_correct:
                baseline_mismatch += 1
                continue

            fmap = fmap_by_id.get(row.get("id", ""), {})
            # 基线是 **Change H 之后** 的状态，因为 I 是叠加在 H 上的
            h_pred, i_pred, changed = [], [], False
            for call in pred:
                fname = call.get("name")
                func = fmap.get(fname) or {}
                hargs, iargs = {}, {}
                for k, v in (call.get("arguments") or {}).items():
                    vocab = param_vocab(func, k)
                    hv = snap_value(v, vocab)
                    iv = snap_list_with_style(v, vocab)
                    hargs[k] = hv
                    iargs[k] = iv
                    if iv != hv:
                        changed = True
                h_pred.append({"name": fname, "arguments": hargs})
                i_pred.append({"name": fname, "arguments": iargs})
            if not changed:
                continue
            n_touched += 1
            h_correct = judged_correct(h_pred, gt)
            i_correct = judged_correct(i_pred, gt)
            if i_correct and not h_correct:
                gained.append((cat, row.get("id"), h_pred, i_pred))
                per_cat[cat][0] += 1
            elif h_correct and not i_correct:
                lost.append((cat, row.get("id"), h_pred, i_pred))
                per_cat[cat][1] += 1
            else:
                touched.append((cat, row.get("id"), h_pred, i_pred))
                per_cat[cat][2] += 1

    if args.show:
        pool = {"gained": gained, "lost": lost, "touched": touched}[args.show]
        print(f"=== {args.show} ({len(pool)}) ===")
        for cat, iid, old, new in pool[: args.limit]:
            print(f"\n[{cat}] {iid}")
            for o, n in zip(old, new):
                for k in o["arguments"]:
                    if o["arguments"][k] != n["arguments"][k]:
                        print(f"    {o['name']}.{k}: "
                              f"{o['arguments'][k]} -> {n['arguments'][k]}")
        return

    print(f"STYLE PROPAGATION (Change I, stacked on H)  tag={args.tag}")
    print("=" * 78)
    print(f"  samples replayed        : {n_seen}")
    print(f"  excluded (fidelity)     : {baseline_mismatch} "
          f"({100 * (1 - baseline_mismatch / max(n_seen, 1)):.1f}% agreement)")
    print(f"  values changed beyond H : {n_touched}")
    print(f"      -> GAINED           : {len(gained)}")
    print(f"      -> LOST             : {len(lost)}")
    print(f"      -> neutral          : {len(touched)}")
    print()
    if per_cat:
        print(f"{'category':<26}{'gain':>6}{'lost':>6}{'neut':>6}")
        print("-" * 44)
        for cat, (g, l, t) in sorted(per_cat.items(), key=lambda x: -x[1][0]):
            print(f"{cat:<26}{g:>6}{l:>6}{t:>6}")
        print()
    for label, rows in (("GAINED", gained), ("LOST", lost)):
        if not rows:
            continue
        print(f"{label}:")
        for cat, iid, old, new in rows:
            for o, n in zip(old, new):
                for k in o["arguments"]:
                    if o["arguments"][k] != n["arguments"][k]:
                        print(f"  [{cat}] {iid}: {o['name']}.{k}: "
                              f"{o['arguments'][k]} -> {n['arguments'][k]}")
        print()


if __name__ == "__main__":
    main()
