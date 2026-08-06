"""Track A 可行性诊断（不改代码，只量证据）。

回答两个决定性问题：
  U 侧 (under/missing_func)：缺失函数的 GT 参数，能否从"已有兄弟调用的参数"机械复制出来？
       若能复制 -> 后处理有机会补全；若不能 -> 需要重新做参数抽取，后处理不可行。
  O 侧 (over/*)：多出来的调用，是否能用"GT-free 的机械信号"识别？
       统计多出的调用与保留调用的关系（同名重复 / schema 外 / 负值对冲）。

用法：python scripts/diag_trackA_feasibility.py
"""
from __future__ import annotations
import json
import re
from collections import Counter

from diag_weakroot_v24 import load_gt, score, WEAK, _values_match


def gt_items(gt):
    """展开 GT 为 [(fname, param_dict)] 列表。"""
    out = []
    for it in gt or []:
        if isinstance(it, dict):
            for fn, pr in it.items():
                out.append((fn, pr if isinstance(pr, dict) else {}))
    return out


def args_copyable(gt_params: dict, sibling_args_list: list) -> str:
    """判断 GT 参数能否从某个兄弟调用的参数里复制得到。

    返回 'exact'   : 存在兄弟，其参数在 GT 要求的每个键上都命中
         'partial' : 存在兄弟命中部分键（>=1 且非全部）
         'none'    : 没有任何键可复制
         'trivial' : GT 无必填键（空参数），天然可补
    """
    keys = [k for k, v in gt_params.items() if isinstance(v, list) and v]
    if not keys:
        return "trivial"
    best = 0
    for sib in sibling_args_list:
        hit = 0
        for k in keys:
            pv = sib.get(k)
            if pv is None:
                continue
            if any(_values_match(pv, x) for x in gt_params[k]):
                hit += 1
        best = max(best, hit)
    if best == len(keys):
        return "exact"
    if best > 0:
        return "partial"
    return "none"


def name_tokens(fname: str):
    return [t for t in re.split(r"[._\-]|(?<=[a-z])(?=[A-Z])", fname) if len(t) > 2]


def lexical_evidence(fname: str, query: str) -> bool:
    q = query.lower()
    toks = [t.lower() for t in name_tokens(fname)]
    if not toks:
        return False
    return sum(1 for t in toks if t in q) >= max(1, len(toks) // 2)


def main():
    u_copy = Counter()
    u_lex = Counter()
    o_shape = Counter()
    u_examples, o_examples = [], []

    for cat, path in WEAK.items():
        gtm = load_gt(cat)
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        for r in rows:
            if r.get("error") or r.get("correct"):
                continue
            gt = gtm.get(r["id"])
            if score(r["pred"], gt, cat) != "COUNT":
                continue
            gi = gt_items(gt)
            gnames = [n for n, _ in gi]
            pnames = [c.get("name", "") for c in r["pred"]]
            pargs = [c.get("arguments", {}) for c in r["pred"]]
            gc, pc = Counter(gnames), Counter(pnames)

            if len(pnames) < len(gnames):
                # under：找出缺失的 (fname, params)
                missing = []
                seen = Counter()
                for fn, pr in gi:
                    seen[fn] += 1
                    if seen[fn] > pc.get(fn, 0):
                        missing.append((fn, pr))
                for fn, pr in missing:
                    verdict = args_copyable(pr, pargs)
                    u_copy[verdict] += 1
                    u_lex[(verdict, lexical_evidence(fn, r.get("query", "")))] += 1
                    if verdict in ("exact", "trivial") and len(u_examples) < 8:
                        u_examples.append((r["id"], fn, pr, pargs, r.get("query", "")[:130]))
            else:
                # over：多出来的调用是什么形态
                extra_names = [f for f in pc if pc[f] > gc.get(f, 0)]
                for f in extra_names:
                    if f not in gc:
                        o_shape["extra_func(schema内但GT不要)"] += pc[f] - gc.get(f, 0)
                    else:
                        o_shape["same_func_repeat(同名多调)"] += pc[f] - gc[f]
                if len(o_examples) < 8:
                    o_examples.append((r["id"], gnames, pnames, r.get("query", "")[:130]))

    print("=" * 72)
    print("U 侧：缺失函数的 GT 参数能否从兄弟调用复制？（决定后处理可行性）")
    print("=" * 72)
    tot_u = sum(u_copy.values())
    for k, v in u_copy.most_common():
        print(f"   {k:9s} : {v:3d}  ({v/max(tot_u,1)*100:.1f}%)")
    print(f"   合计缺失函数实例 : {tot_u}")
    print("\n   交叉：可复制性 × 查询里有函数名词法证据")
    for (verdict, lex), v in sorted(u_lex.items()):
        print(f"     {verdict:9s} lexical={str(lex):5s} : {v}")

    print("\n   可机械补全的样例（exact/trivial）:")
    for sid, fn, pr, pargs, q in u_examples:
        print(f"     [{sid}] 缺 {fn}")
        print(f"        GT参数   = {pr}")
        print(f"        兄弟参数 = {pargs}")
        print(f"        query    = {q}")

    print("\n" + "=" * 72)
    print("O 侧：多出来的调用形态")
    print("=" * 72)
    for k, v in o_shape.most_common():
        print(f"   {k:28s} : {v}")
    print("\n   样例:")
    for sid, gn, pn, q in o_examples:
        print(f"     [{sid}]")
        print(f"        GT   = {gn}")
        print(f"        PRED = {pn}")
        print(f"        query= {q}")


if __name__ == "__main__":
    main()
