"""Track A 机制确认：COUNT 失败到底发生在哪一层？

从 trace 里抽出三个阶段的产物，判定失败归属：
  SELECT_WRONG   : LLM selected 的函数集合本身就 != GT 函数集合（选择层）
  EXTRACT_OVER   : 选择层正确，但某函数的参数抽取产出了多于 GT 的实例（抽取层过度笛卡尔积）
  EXTRACT_UNDER  : 选择层正确，但某函数的参数抽取产出少于 GT（抽取层漏抽）
  POST_DROP      : 抽取层数量正确，但最终输出被后处理改变（后处理层）

这决定了修复应该落在哪一层——上游 prompt 还是后处理。

用法：PYTHONPATH=scripts python scripts/diag_trackA_mechanism.py
"""
from __future__ import annotations
import json
import re
from collections import Counter

from diag_weakroot_v24 import load_gt, score, WEAK

PARALLEL_CATS = ["parallel_multiple", "live_parallel", "live_parallel_multiple"]
SEL_RE = re.compile(r"LLM selected: (\[.*\])")
PARAM_RE = re.compile(r"^\s+([A-Za-z0-9_.]+) params: (\{.*\})\s*$")


def parse_trace(trace):
    """返回 (llm_selected 或 None, {fname: 抽取实例数})。"""
    selected = None
    extracted = Counter()
    for t in trace or []:
        m = SEL_RE.search(t)
        if m:
            try:
                selected = json.loads(m.group(1).replace("'", '"'))
            except Exception:  # noqa: BLE001
                selected = None
        m2 = PARAM_RE.match(t)
        if m2:
            extracted[m2.group(1)] += 1
    return selected, extracted


def main():
    tally = Counter()
    examples = {}

    for cat in PARALLEL_CATS:
        path = WEAK.get(cat)
        if not path:
            continue
        gtm = load_gt(cat)
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        for r in rows:
            if r.get("error") or r.get("correct"):
                continue
            gt = gtm.get(r["id"])
            if score(r["pred"], gt, cat) != "COUNT":
                continue

            gnames = Counter(r.get("gt_names") or [])
            pnames = Counter(r.get("pred_names") or [])
            selected, extracted = parse_trace(r.get("trace"))

            if selected is None:
                tally["NO_TRACE(快路径/无LLM选择)"] += 1
                continue

            # 选择层是否正确：选中的函数集合 == GT 用到的函数集合（去重后比）
            if set(selected) != set(gnames):
                tally["SELECT_WRONG(选择层)"] += 1
                examples.setdefault("SELECT_WRONG(选择层)", []).append(
                    (r["id"], sorted(set(selected)), sorted(set(gnames)))
                )
                continue

            if not extracted:
                tally["NO_EXTRACT_TRACE"] += 1
                continue

            over = [f for f in extracted if extracted[f] > gnames.get(f, 0)]
            under = [f for f in extracted if extracted[f] < gnames.get(f, 0)]
            if over and not under:
                k = "EXTRACT_OVER(抽取层·过度笛卡尔积)"
            elif under and not over:
                k = "EXTRACT_UNDER(抽取层·漏抽)"
            elif over and under:
                k = "EXTRACT_MIXED(抽取层·错配)"
            else:
                k = "POST_DROP(后处理层)" if pnames != extracted else "OTHER"
            tally[k] += 1
            examples.setdefault(k, []).append(
                (r["id"], dict(extracted), dict(gnames))
            )

    print("=" * 72)
    print("COUNT 失败的层级归属（决定修复该落在哪一层）")
    print("=" * 72)
    tot = sum(tally.values())
    for k, v in tally.most_common():
        print(f"   {k:36s} : {v:3d}  ({v/max(tot,1)*100:.1f}%)")
    print(f"   合计 : {tot}")

    for k in ["EXTRACT_OVER(抽取层·过度笛卡尔积)", "EXTRACT_UNDER(抽取层·漏抽)",
              "EXTRACT_MIXED(抽取层·错配)", "SELECT_WRONG(选择层)"]:
        ex = examples.get(k, [])
        if not ex:
            continue
        print(f"\n--- {k} 样例 ---")
        for sid, e, g in ex[:6]:
            print(f"   [{sid}]")
            print(f"      抽取产出 = {e}")
            print(f"      GT 需要  = {g}")


if __name__ == "__main__":
    main()
