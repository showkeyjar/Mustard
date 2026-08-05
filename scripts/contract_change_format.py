#!/usr/bin/env python3
"""Change M (documented-value-format requery) 的契约测试 + 三态对账。

比对基线 v22 与上线 v23 在 live_multiple 上的逐样本结果，用 trace 里模型
原始的 `NAME params:` 生成签名区分三种结论：

  kept              承诺方向实现，且生成签名相同（被证实）
  broken            生成签名相同，但承诺方向没实现（真违约）
  untestable_drift  生成本身漂移（重跑噪声），承诺无法在此验证（不是违约）

同时量承诺外样本的翻转率作为噪声标尺，并做收支闭合：
  承诺集合内实测净值 + 承诺外噪声净漂移 = 全部可比样本的总差值

为什么用生成签名：Change M 是生成后处理，trace 的 `params:` 行在
requery 之前记录（见服务端 Change M 注释）。所以即使 v23 多了 requery 的
LLM 调用日志，PARAMS_LINE 匹配到的原始生成签名在 v22/v23 之间仍一致——
这正是三态判定能成立的前提。

用法:
    python scripts/contract_change_format.py --base v22 --verify v23
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag_gate_counterfactual import PARAMS_LINE  # noqa: E402

DIAG = Path(__file__).resolve().parent.parent / "data" / "eval" / "diag"
CAT = "live_multiple"


def load_rows(tag: str):
    path = DIAG / f"{CAT}_{tag}.jsonl"
    if not path.exists():
        raise SystemExit(f"no such file: {path}")
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            yield json.loads(line)


def gen_signature(row: dict):
    """trace 里模型这一轮实际生成的参数。Change M 不在此处留痕，所以签名稳定。"""
    sig = []
    for ln in row.get("trace") or []:
        m = PARAMS_LINE.match(str(ln))
        if m:
            try:
                args = ast.literal_eval(m.group(2))
            except Exception:
                args = m.group(2)
            sig.append((m.group(1), repr(args)))
    return tuple(sig) or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="v22")
    ap.add_argument("--verify", default="v23")
    ap.add_argument("--promise", default=str(DIAG / "promise_v23_format.json"))
    args = ap.parse_args()

    promise = json.loads(Path(args.promise).read_text(encoding="utf-8"))
    gain_ids = set(promise.get("gain", []))
    loss_ids = set(promise.get("loss", []))
    neutral_ids = set(promise.get("neutral", []))
    promised_ids = gain_ids | loss_ids | neutral_ids

    base_rows = {r["id"]: r for r in load_rows(args.base)}
    verify_rows = {r["id"]: r for r in load_rows(args.verify)}
    base_sig = {sid: gen_signature(r) for sid, r in base_rows.items()}
    verify_sig = {sid: gen_signature(r) for sid, r in verify_rows.items()}
    base_ok = {sid: bool(r.get("correct")) for sid, r in base_rows.items()}
    verify_ok = {sid: bool(r.get("correct")) for sid, r in verify_rows.items()}

    print(f"CONTRACT  Change M  base={args.base}  verify={args.verify}  cat={CAT}")
    print("=" * 78)
    print(f"承诺清单: gain {len(gain_ids)} / loss {len(loss_ids)} / "
          f"neutral {len(neutral_ids)}  (净 {len(gain_ids)-len(loss_ids):+d})")

    kept = broken = drift = unexpected = 0
    broken_gain = broken_loss = 0
    drift_gain = drift_loss = 0
    realized_gain = realized_loss = 0
    unexpected_rows = []

    for sid in sorted(promised_ids):
        if sid not in verify_rows:
            print(f"  [skip] {sid}: not in {args.verify}")
            continue
        was = base_ok.get(sid)
        now = verify_ok.get(sid)
        same_gen = (
            base_sig.get(sid) is not None
            and base_sig.get(sid) == verify_sig.get(sid)
        )
        if sid in gain_ids:
            direction, predicted = "gain", True
        elif sid in loss_ids:
            direction, predicted = "loss", False
        else:
            direction, predicted = "neutral", was

        if now == predicted:
            kept += 1
            if same_gen and direction == "gain" and now and not was:
                realized_gain += 1
            if same_gen and direction == "loss" and was and not now:
                realized_loss += 1
        elif same_gen and now:
            # 偏离承诺，但方向有利：承诺说会错、实际判对。
            # 这不是违约（没有引入风险），而是离线判分比线上严格导致的低估。
            # 单列出来而不是计入 broken —— 但也不能藏起来：它同样是预测失准，
            # 说明离线镜像和线上判分器之间有保真度缺口，必须逐个解释清楚。
            unexpected += 1
            unexpected_rows.append((sid, direction, predicted))
            print(f"  [意外+ ] {sid} ({direction}): same generation, "
                  f"promised correct={predicted} actual=True  <-- 离线判分偏严")
        elif same_gen:
            broken += 1
            if direction == "gain":
                broken_gain += 1
            elif direction == "loss":
                broken_loss += 1
            print(f"  [BROKEN] {sid} ({direction}): same generation, "
                  f"promised correct={predicted} actual={now}")
        else:
            drift += 1
            if direction == "gain":
                drift_gain += 1
            elif direction == "loss":
                drift_loss += 1
            print(f"  [drift ] {sid} ({direction}): generation drifted, "
                  f"contract not testable here")

    print(f"\n  分态: kept={kept}  broken={broken} "
          f"(gain {broken_gain}/loss {broken_loss})  "
          f"untestable_drift={drift} (gain {drift_gain}/loss {drift_loss})  "
          f"unexpected_favorable={unexpected}")
    print(f"  兑现 (同生成): gain {realized_gain} / loss {realized_loss}")
    if unexpected_rows:
        print(f"  ! {unexpected} 个有利方向的预测失准 —— 不计违约，但说明离线判分器"
              f"比线上严格，收益被系统性低估。每个都要能解释。")

    # 噪声标尺：承诺外样本
    both = (set(base_ok) & set(verify_ok)) - promised_ids
    up = sum(1 for i in both if verify_ok[i] and not base_ok[i])
    down = sum(1 for i in both if base_ok[i] and not verify_ok[i])
    if both:
        print(f"\n  噪声标尺（承诺外 {len(both)} 样本）: 变好 {up}  变坏 {down}  "
              f"净 {up-down:+d}  翻转率 {(up+down)/len(both):.2%}")

    # 收支闭合
    promised_measured = sum(
        (1 if (verify_ok[s] and not base_ok[s]) else
         -1 if (base_ok[s] and not verify_ok[s]) else 0)
        for s in promised_ids if s in verify_rows and s in base_rows
    )
    total_measured = sum(
        1 if verify_ok[s] and not base_ok[s] else
        -1 if base_ok[s] and not verify_ok[s] else 0
        for s in (set(base_ok) & set(verify_ok))
    )
    closed = promised_measured + (up - down) == total_measured
    print(f"\n  收支闭合:")
    print(f"    承诺净收益(预测)      = {len(gain_ids)-len(loss_ids):+d}")
    print(f"    承诺集合内实测净值    = {promised_measured:+d}")
    print(f"    承诺外噪声净漂移      = {up-down:+d}")
    print(f"    全部可比样本总差值    = {total_measured:+d}")
    print(f"    闭合: {closed}  "
          f"({promised_measured} + {up-down} = {total_measured})")

    # 准确率对比
    base_correct = sum(1 for v in base_ok.values() if v)
    verify_correct = sum(1 for v in verify_ok.values() if v)
    n = len(base_ok)
    print(f"\n  live_multiple 准确率: {args.base} {base_correct}/{n} = "
          f"{base_correct/n:.2%}  ->  {args.verify} {verify_correct}/{len(verify_ok)} "
          f"= {verify_correct/len(verify_ok):.2%}  "
          f"({(verify_correct-base_correct):+d}, "
          f"{(verify_correct/len(verify_ok)-base_correct/n):+.2%})")

    if drift:
        print("\n  注意：drift 不是违约，但表示这些承诺没被证实。"
              "若 gain/loss 上的 drift 集中，收益结论就缺少直接证据。")
    if broken == 0:
        print("\n  契约通过：无不利方向的违约 (broken=0)。")
    else:
        print(f"\n  契约失败：{broken} 个不利方向的违约。")


if __name__ == "__main__":
    main()
