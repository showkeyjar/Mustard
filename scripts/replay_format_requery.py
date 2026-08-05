#!/usr/bin/env python3
"""把实测的重问结果代回原预测，做整样本级重放判分，产出承诺清单。

为什么这一步是保真的：重问只替换**已生成参数的字符串值**，
不改变函数选择、不改变切分、不新增或删除调用——LLM 的决策序列完全不动。
性质与 `diag_vocab_snap.py`（Change H）相同，属于纯后处理离线重放。
（对比：切分类改动会改变后续 LLM 调用，那种就不能离线重放。）

护栏：先用离线判分器复现线上判定，不能复现的样本直接剔除，
不让它进入结论——这条是 Change H 那轮定下来的，别删。

用法:
    python scripts/replay_format_requery.py v22
    python scripts/replay_format_requery.py v22 --promise data/eval/diag/promise_v23.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag_documented_format import CATEGORIES, lit          # noqa: E402
from diag_vocab_snap import judged_correct, norm            # noqa: E402

DIAG = Path(__file__).resolve().parent.parent / "data" / "eval" / "diag"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--requery", default=None)
    ap.add_argument("--promise", default=None)
    args = ap.parse_args()

    rq_path = Path(args.requery or DIAG / f"format_requery_{args.tag}.json")
    requeries = json.loads(rq_path.read_text(encoding="utf-8"))

    # (sample_id, func, param) -> 新值
    patch: dict[tuple, str] = {}
    for r in requeries:
        patch[(r["id"], r["func"], r["param"])] = r["requery"]
    touched_ids = {r["id"] for r in requeries}
    print(f"重问补丁 {len(patch)} 条，覆盖样本 {len(touched_ids)} 个")

    gain, loss, neutral = [], [], []
    unreproducible = 0
    n_seen = 0

    for cat in CATEGORIES:
        path = DIAG / f"{cat}_{args.tag}.jsonl"
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            sid = row.get("id")
            if sid not in touched_ids:
                continue
            pred = lit(row.get("pred")) or []
            gt = lit(row.get("gt")) or []
            was = str(row.get("correct")) == "True"
            n_seen += 1

            # 护栏：线上判定必须能被离线判分器复现
            if judged_correct(pred, gt) != was:
                unreproducible += 1
                continue

            newpred, changed = [], False
            for call in pred:
                if not isinstance(call, dict):
                    newpred.append(call)
                    continue
                fname = call.get("name")
                nargs = dict(call.get("arguments") or {})
                for pname in list(nargs):
                    nv = patch.get((sid, fname, pname))
                    if nv is not None and nv != nargs[pname]:
                        nargs[pname] = nv
                        changed = True
                newpred.append({"name": fname, "arguments": nargs})
            if not changed:
                continue

            now = judged_correct(newpred, gt)
            rec = {"id": sid, "cat": cat, "was": was, "now": now,
                   "pred": [c.get("arguments") for c in pred],
                   "new": [c.get("arguments") for c in newpred]}
            if now and not was:
                gain.append(rec)
            elif was and not now:
                loss.append(rec)
            else:
                neutral.append(rec)

    print(f"扫到受影响样本 {n_seen}，其中离线判分无法复现线上判定 "
          f"{unreproducible} 个（已剔除，不计入结论）")
    print()
    print("===== 重放结果（整样本判分）=====")
    print(f"  GAIN    {len(gain):>4}")
    print(f"  LOSS    {len(loss):>4}")
    print(f"  NEUTRAL {len(neutral):>4}")
    print(f"  净收益  {len(gain) - len(loss):+d}")
    print()

    bycat = collections.Counter(r["cat"] for r in gain)
    if bycat:
        print("  GAIN 分布: " + " ".join(f"{c}={n}" for c, n in bycat.most_common()))
    if loss:
        print("\n  --- 承诺的回退样本（上线后必须逐个复核）---")
        for r in loss:
            print(f"    {r['id']:<28} {r['pred']} -> {r['new']}")
    if neutral:
        print(f"\n  --- 改了值但判定不变的 {len(neutral)} 个（前 8）---")
        for r in neutral[:8]:
            print(f"    [{'OK ' if r['was'] else 'ERR'}] {r['id']:<26} "
                  f"{r['pred']} -> {r['new']}")

    if args.promise:
        out = {"change": "documented_value_format_requery",
               "base_tag": args.tag,
               "gain": [r["id"] for r in gain],
               "loss": [r["id"] for r in loss],
               "neutral": [r["id"] for r in neutral]}
        Path(args.promise).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  承诺清单已写入 {args.promise}")


if __name__ == "__main__":
    main()
