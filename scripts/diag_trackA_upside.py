"""Track A 真实上行空间（按样本计，不按调用实例计）。

一个样本只有"全部调用都对"才得分，所以必须按样本算可修复性。

U 侧：样本内 *所有* 缺失函数都能从兄弟调用机械复制参数 -> 才算可补
O 侧：GT 的每个 (name, args) 都能在 PRED 里找到匹配 -> 退化为"纯删除问题"
      （只删不加就能得分，规则可以做得很保守）

用法：PYTHONPATH=scripts python scripts/diag_trackA_upside.py
"""
from __future__ import annotations
import json
from collections import Counter

from diag_weakroot_v24 import load_gt, score, WEAK, _params_match
from diag_trackA_feasibility import gt_items, args_copyable


def gt_subset_of_pred(gi, pred):
    """GT 的每个 (fname, params) 是否都能在 pred 里找到一个未被占用的匹配调用。

    贪心匹配即可：只要存在一个完美匹配，就说明"只删不加"可解。
    """
    used = [False] * len(pred)
    for fn, gp in gi:
        hit = False
        for i, c in enumerate(pred):
            if used[i] or c.get("name") != fn:
                continue
            if _params_match(c.get("arguments", {}), gp):
                used[i] = True
                hit = True
                break
        if not hit:
            return False, 0
    return True, len(pred) - len(gi)


def main():
    u_all_copyable = Counter()
    o_pure_delete = Counter()
    o_delete_examples = []
    u_ok_examples = []
    per_cat = Counter()

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
            pred = r["pred"]
            pargs = [c.get("arguments", {}) for c in pred]
            pc = Counter(c.get("name", "") for c in pred)

            if len(pred) < len(gi):
                verdicts = []
                seen = Counter()
                for fn, pr in gi:
                    seen[fn] += 1
                    if seen[fn] > pc.get(fn, 0):
                        verdicts.append(args_copyable(pr, pargs))
                allok = verdicts and all(v in ("exact", "trivial") for v in verdicts)
                u_all_copyable["可补(全部exact/trivial)" if allok else "不可补"] += 1
                per_cat[(cat, "under", "可补" if allok else "不可补")] += 1
                if allok and len(u_ok_examples) < 6:
                    u_ok_examples.append((r["id"], r.get("gt_names"), r.get("pred_names")))
            else:
                ok, nextra = gt_subset_of_pred(gi, pred)
                key = "纯删除可解(GT⊆PRED)" if ok else "删不出来(GT⊄PRED)"
                o_pure_delete[key] += 1
                per_cat[(cat, "over", "纯删除可解" if ok else "删不出来")] += 1
                if ok and len(o_delete_examples) < 10:
                    o_delete_examples.append(
                        (r["id"], nextra, r.get("gt_names"), r.get("pred_names"), r.get("query", "")[:110])
                    )

    print("=" * 74)
    print("Track A 真实上行空间（按样本计）")
    print("=" * 74)
    print("\n[U 侧] under / 需要补函数")
    tu = sum(u_all_copyable.values())
    for k, v in u_all_copyable.most_common():
        print(f"   {k:24s} : {v:3d}  ({v/max(tu,1)*100:.1f}%)")
    print(f"   小计 : {tu}")
    for sid, gn, pn in u_ok_examples:
        print(f"     [{sid}] GT={gn}")
        print(f"        PRED={pn}")

    print("\n[O 侧] over / 需要删调用")
    to = sum(o_pure_delete.values())
    for k, v in o_pure_delete.most_common():
        print(f"   {k:24s} : {v:3d}  ({v/max(to,1)*100:.1f}%)")
    print(f"   小计 : {to}")
    print("\n   纯删除可解的样例（只需删掉 N 个冗余调用即可得分）:")
    for sid, nextra, gn, pn, q in o_delete_examples:
        print(f"     [{sid}] 需删 {nextra} 个")
        print(f"        GT   = {gn}")
        print(f"        PRED = {pn}")
        print(f"        query= {q}")

    print("\n" + "=" * 74)
    print("分类别明细")
    print("=" * 74)
    for (cat, d, v), n in sorted(per_cat.items()):
        print(f"   {cat:24s} {d:6s} {v:12s} : {n}")

    ceiling = o_pure_delete.get("纯删除可解(GT⊆PRED)", 0) + u_all_copyable.get("可补(全部exact/trivial)", 0)
    print(f"\n>>> Track A 理论天花板（后处理形态）: {ceiling} 个样本")
    print(f"    其中纯删除 {o_pure_delete.get('纯删除可解(GT⊆PRED)',0)} + 可补全 {u_all_copyable.get('可补(全部exact/trivial)',0)}")


if __name__ == "__main__":
    main()
