"""Track A 判别性分析：到底存不存在能区分「该删」与「该留」的 GT-free 信号？

不再逐个猜策略。直接给每个调用打上标签，再看候选特征能否分开两类：
  MUST_DELETE : 出现在 GT⊆PRED 的 over 样本里、且不在 GT 匹配集中的冗余调用
  MUST_KEEP   : 出现在当前判对样本里的调用（删任何一个都会造成 LOSS）

若某特征在两类上的分布高度重叠，则该特征做不出安全的删除规则——
这是"规则不存在"的证据，而不是"还没调好参数"。

用法：PYTHONPATH=scripts python scripts/diag_trackA_separability.py
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict

from diag_weakroot_v24 import load_gt, score, WEAK, _params_match
from probe_track_a import (
    PARALLEL_CATS, func_trigger_positions, arg_positions, signal_tokens,
)


def feats(call, calls, query, all_names):
    """GT-free 特征。"""
    fn = call.get("name", "")
    args = call.get("arguments", {}) or {}
    trig = func_trigger_positions(fn, all_names, query)
    ap = arg_positions(args, query)
    n_args = len([v for v in args.values() if v not in (None, "", [])])
    same_name = sum(1 for c in calls if c.get("name") == fn)
    return {
        # 函数名在 query 里有无词法证据
        "lex": bool(trig),
        # 参数字面值在 query 里有几个能落地
        "grounded": (len(ap) / n_args) if n_args else 1.0,
        # 参数与函数触发点的最小距离
        "dist": (min(abs(a - p) for a in ap for p in trig) if (ap and trig) else -1),
        # 是否同名多调
        "dup": same_name > 1,
        "n_args": n_args,
    }


def main():
    delete_rows, keep_rows = [], []

    for cat in PARALLEL_CATS:
        path = WEAK.get(cat)
        if not path:
            continue
        gtm = load_gt(cat)
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        for r in rows:
            if r.get("error"):
                continue
            gt = gtm.get(r["id"])
            pred = r["pred"]
            query = r.get("query", "")
            names = r.get("func_names", [])

            if r.get("correct"):
                # 判对样本里的每个调用都是 MUST_KEEP
                for c in pred:
                    keep_rows.append(feats(c, pred, query, names))
                continue

            if score(pred, gt, cat) != "COUNT" or len(pred) <= len(gt or []):
                continue
            # over 样本：贪心匹配出 GT 占用的调用，剩下的就是 MUST_DELETE
            gi = []
            for it in (gt or []):
                if isinstance(it, dict):
                    for f, p in it.items():
                        gi.append((f, p if isinstance(p, dict) else {}))
            used = [False] * len(pred)
            allmatched = True
            for f, gp in gi:
                hit = False
                for i, c in enumerate(pred):
                    if used[i] or c.get("name") != f:
                        continue
                    if _params_match(c.get("arguments", {}), gp):
                        used[i] = True
                        hit = True
                        break
                if not hit:
                    allmatched = False
                    break
            if not allmatched:
                continue
            for i, c in enumerate(pred):
                if not used[i]:
                    delete_rows.append(feats(c, pred, query, names))

    print("=" * 70)
    print(f"MUST_DELETE 调用 : {len(delete_rows)}    MUST_KEEP 调用 : {len(keep_rows)}")
    print("=" * 70)

    def dist(rows, key, buckets):
        c = Counter()
        for r in rows:
            c[buckets(r[key])] += 1
        n = max(len(rows), 1)
        return {k: (v, v / n * 100) for k, v in c.items()}

    def show(title, key, buckets, order):
        d1 = dist(delete_rows, key, buckets)
        d2 = dist(keep_rows, key, buckets)
        print(f"\n--- 特征: {title} ---")
        print(f"    {'桶':<14} {'MUST_DELETE':>18} {'MUST_KEEP':>18}   判别力")
        for b in order:
            a = d1.get(b, (0, 0.0))
            k = d2.get(b, (0, 0.0))
            gap = a[1] - k[1]
            flag = "强" if abs(gap) > 40 else ("弱" if abs(gap) > 15 else "无")
            print(f"    {str(b):<14} {a[0]:6d} ({a[1]:5.1f}%) {k[0]:6d} ({k[1]:5.1f}%)   {gap:+6.1f}pp {flag}")

    show("函数名有词法证据", "lex", lambda v: v, [True, False])
    show("同名多调", "dup", lambda v: v, [True, False])
    show("参数落地率", "grounded",
         lambda v: "0" if v == 0 else ("(0,0.5)" if v < 0.5 else ("[0.5,1)" if v < 1 else "1.0")),
         ["0", "(0,0.5)", "[0.5,1)", "1.0"])
    show("参数-函数距离", "dist",
         lambda v: "无" if v < 0 else ("<50" if v < 50 else ("50-150" if v < 150 else ">=150")),
         ["无", "<50", "50-150", ">=150"])

    # 组合规则的最优可能：任取一个特征桶做删除规则，最好能达到什么
    print("\n" + "=" * 70)
    print("单特征删除规则的最优上界（precision = 删中的确实该删）")
    print("=" * 70)
    rules = {
        "lex==False": lambda r: not r["lex"],
        "grounded==0": lambda r: r["grounded"] == 0,
        "lex==False AND grounded==0": lambda r: (not r["lex"]) and r["grounded"] == 0,
        "dup AND grounded==0": lambda r: r["dup"] and r["grounded"] == 0,
        "dup AND dist>=150": lambda r: r["dup"] and r["dist"] >= 150,
    }
    for name, f in rules.items():
        tp = sum(1 for r in delete_rows if f(r))
        fp = sum(1 for r in keep_rows if f(r))
        prec = tp / (tp + fp) * 100 if (tp + fp) else 0.0
        rec = tp / max(len(delete_rows), 1) * 100
        print(f"   {name:30s} 命中该删 {tp:3d} / 误伤该留 {fp:4d}  ->  precision {prec:5.1f}%  recall {rec:5.1f}%")


if __name__ == "__main__":
    main()
