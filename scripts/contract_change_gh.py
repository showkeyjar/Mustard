#!/usr/bin/env python3
"""Change G/H 的契约测试 + 事前承诺清单导出。

为什么需要这个脚本
------------------
反事实脚本从生产代码 import 函数，它的"基线"会随生产代码状态漂移。
如果承诺清单不是在**真正要上线的那个配置**下导出，清单本身就是错的。
这个项目已经在这上面栽过三次，所以把它固化成可重跑的契约。

它做三件事：
  1. 断言 Change G 的安全不变量（irrelevance 的保护规则必须仍然触发）
  2. 断言 Change H 的生产实现与离线回放实现逐样本等价
  3. 在当前生产代码状态下导出承诺清单（受影响样本 + 预期翻转方向）

用法:
    python scripts/contract_change_gh.py --base v21 --out v22
    python scripts/contract_change_gh.py --base v21 --verify v22   # 评测后核对
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import carm_bfcl_server_optimized as prod  # noqa: E402
import diag_vocab_snap as vocab_ref  # noqa: E402
from diag_gate_counterfactual import PARAMS_LINE, calls_match_gt, norm  # noqa: E402

DIAG = Path(__file__).resolve().parent.parent / "data" / "eval" / "diag"

# Change G 的安全不变量：这些样本靠"空 list 算退化"这条规则保住正确答案，
# G2 只放宽空 dict，所以它们必须继续被门控抑制。
GATE_MUST_STILL_FIRE = {
    "irrelevance_130": "empty list in required arg",
    "irrelevance_218": "empty list in required arg",
    "live_relevance_6-6-0": "empty string in required arg",
}


def load_rows(tag: str):
    for path in sorted(glob.glob(str(DIAG / f"*_{tag}.jsonl"))):
        cat = os.path.basename(path)[: -len(f"_{tag}.jsonl")]
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                yield cat, json.loads(line)
            except json.JSONDecodeError:
                continue


def gen_signature(row: dict):
    """门控/吸附之前，模型这一轮实际生成了什么。

    Change G 和 Change H 都是**生成后处理**：给定同一份模型输出，它们的效果
    是确定的。契约回放正是建立在"模型输出不变"这个前提上。

    但重跑时生成本身有噪声（采样温度 + 判分器 temperature=0.001）。如果直接拿
    correct 标志位对账，模型换了个输出导致的翻转会被记成"违约"，反过来也可能
    掩盖真正的违约。所以对账前必须先确认前提成立。

    返回 None 表示 trace 里没有可比的生成行 —— 此时只能降级为标志位比对，
    并如实标注证据不足，而不是假装验证过了。
    """
    sig = []
    for ln in row.get("trace") or []:
        m = PARAMS_LINE.match(ln)
        if m:
            try:
                args = ast.literal_eval(m.group(2))
            except Exception:
                args = m.group(2)
            sig.append((m.group(1), repr(args)))
    return tuple(sig) or None


def schema_maps(cat: str):
    """(gate_schema_map, raw function list) per sample id."""
    from eval_bfcl_v4_fast import build_messages, load_bfcl_data

    gate, raw = {}, {}
    for item in load_bfcl_data(cat):
        _, funcs = build_messages(item)
        sid = item.get("id", "")
        gate[sid] = prod._gate_schema_map(funcs)
        raw[sid] = funcs
    return gate, raw


# ---------------------------------------------------------------------------
# Contract 1 — Change G invariants
# ---------------------------------------------------------------------------
def check_gate(base: str):
    failures, still_fires, released = [], {}, []
    cats = {c for c, _ in load_rows(base)}
    for cat in sorted(cats):
        try:
            gate_sm, _ = schema_maps(cat)
        except Exception as exc:
            print(f"  [warn] {cat}: schema load failed ({exc})")
            continue
        for c, row in load_rows(base):
            if c != cat:
                continue
            lines = [str(x) for x in (norm(row.get("trace")) or [])]
            if "Degenerate-argument gate" not in "\n".join(lines):
                continue
            sid = row.get("id", "")
            sch = gate_sm.get(sid, {})
            query = str(row.get("query", ""))
            gen = []
            for ln in lines:
                m = PARAMS_LINE.match(ln)
                if m:
                    try:
                        gen.append((m.group(1), ast.literal_eval(m.group(2))))
                    except (ValueError, SyntaxError):
                        pass
            fires = any(
                prod._is_degenerate_call(n, a, sch.get(n, {}), query)
                for n, a in gen
                if sch.get(n)
            )
            if fires:
                still_fires[sid] = cat
            else:
                gt = norm(row.get("gt")) or []
                was = str(row.get("correct")) == "True"
                now = calls_match_gt(gen, gt)
                released.append(
                    {
                        "id": sid,
                        "category": cat,
                        "was_correct": was,
                        "predicted_correct": now,
                        "direction": "gain" if (now and not was)
                        else "loss" if (was and not now)
                        else "neutral",
                    }
                )
    for sid, why in GATE_MUST_STILL_FIRE.items():
        if sid not in still_fires:
            failures.append(f"INVARIANT BROKEN: {sid} no longer gated ({why})")
    return failures, released


# ---------------------------------------------------------------------------
# Contract 2 — Change H equivalence with the offline replay
# ---------------------------------------------------------------------------
def check_snap(base: str):
    failures, affected = [], []
    compared = 0
    cats = {c for c, _ in load_rows(base)}
    for cat in sorted(cats):
        try:
            _, raw_sm = schema_maps(cat)
        except Exception:
            continue
        for c, row in load_rows(base):
            if c != cat:
                continue
            sid = row.get("id", "")
            funcs = raw_sm.get(sid) or []
            pred = norm(row.get("pred")) or []
            # pred rows are {"name": ..., "arguments": {...}} — NOT {fname: args}.
            # Getting this wrong made an earlier revision of this contract
            # report "0 values rewritten" while the offline replay saw 15.
            calls = []
            for call in pred:
                if isinstance(call, dict) and "name" in call:
                    calls.append((call.get("name"), call.get("arguments") or {}))
            if not calls:
                continue
            compared += 1

            prod_out = prod.snap_calls_to_schema_vocab(calls, funcs)

            by_name = {f["name"]: f for f in funcs
                       if isinstance(f, dict) and f.get("name")}
            ref_out = []
            for n, a in calls:
                func = by_name.get(n)
                if not func or not isinstance(a, dict):
                    ref_out.append((n, a))
                    continue
                na = dict(a)
                for k, v in a.items():
                    na[k] = vocab_ref.snap_value(v, vocab_ref.param_vocab(func, k))
                ref_out.append((n, na))

            if prod_out != ref_out:
                failures.append(
                    f"SNAP MISMATCH {cat}/{sid}: prod={prod_out} ref={ref_out}"
                )
            if prod_out != calls:
                gt = norm(row.get("gt")) or []
                was = str(row.get("correct")) == "True"
                off_before = vocab_ref.judged_correct(pred, gt)
                now = vocab_ref.judged_correct(
                    [{"name": n, "arguments": a} for n, a in prod_out], gt
                )
                basis = "replay"
                if off_before != was:
                    # The offline judge disagrees with the live verdict on this
                    # sample, so its absolute verdict is not evidence. But the
                    # *delta* still can be. Measured on all 2757 v21 rows the
                    # offline judge is strictly stricter: offline-correct with
                    # live-wrong happens exactly once (live_relevance_6-6-0, a
                    # gate case), while the reverse happens 125 times. So:
                    #
                    #   offline_correct(after) == True  =>  live correct
                    #
                    # holds at 2008/2009. Concretely these are all case-only
                    # rewrites ('seafood' -> 'Seafood') where GT is 'Seafood':
                    # the live judge is case-insensitive and already scored
                    # them correct, the offline one is not. After snapping the
                    # value equals GT exactly, which is the strictest possible
                    # match — it cannot lose a point it already held.
                    if now:
                        basis = "exact_match_after"
                    elif off_before == now:
                        # Snapping does not move the offline verdict either.
                        # Weaker, but still no mechanism for a regression.
                        basis = "delta_invariant"
                    else:
                        # offline goes correct -> wrong. This is the only shape
                        # that can hide a real loss. Surface it, never drop it.
                        failures.append(
                            f"UNPREDICTED {cat}/{sid}: offline verdict flips "
                            f"{off_before}->{now} while live said {was}; "
                            f"before={str(calls)[:120]} after={str(prod_out)[:120]}"
                        )
                        continue
                affected.append(
                    {
                        "id": sid,
                        "category": cat,
                        "was_correct": was,
                        "predicted_correct": now if basis == "replay" else was,
                        "direction": "gain" if (basis == "replay" and now and not was)
                        else "loss" if (basis == "replay" and was and not now)
                        else "neutral",
                        "basis": basis,
                        "before": str(calls)[:200],
                        "after": str(prod_out)[:200],
                    }
                )
    return failures, affected, compared


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="v21")
    ap.add_argument("--out", default="v22")
    ap.add_argument("--verify", help="tag of the post-deploy run to check against")
    args = ap.parse_args()

    promise_path = DIAG / f"promise_{args.out}.json"

    if args.verify:
        if not promise_path.exists():
            raise SystemExit(f"no promise file at {promise_path}")
        promise = json.loads(promise_path.read_text(encoding="utf-8"))
        base_sig = {}
        for _cat, row in load_rows(promise.get("base", args.base)):
            base_sig[row.get("id", "")] = gen_signature(row)
        actual, new_sig = {}, {}
        for _cat, row in load_rows(args.verify):
            sid = row.get("id", "")
            actual[sid] = str(row.get("correct")) == "True"
            new_sig[sid] = gen_signature(row)
        print(f"VERIFY {args.out} promises against run {args.verify}")
        print("=" * 78)
        ok = miss = drift = weak = 0
        drifted, broken = [], []
        for entry in promise["affected"]:
            sid = entry["id"]
            if sid not in actual:
                print(f"  [skip] {sid}: not in {args.verify}")
                continue
            same_gen = (
                base_sig.get(sid) is not None
                and base_sig.get(sid) == new_sig.get(sid)
            )
            if actual[sid] == entry["predicted_correct"]:
                ok += 1
                # A promise that came true on a *different* generation is a
                # coincidence, not a confirmation. Count it, don't hide it.
                if not same_gen:
                    weak += 1
                continue
            # Prediction did not hold. Split real contract breaches from
            # generation noise -- the post-processing contract only claims
            # anything when the model produced the same calls.
            if same_gen:
                miss += 1
                broken.append(entry)
                print(
                    f"  [BROKEN] {sid} ({entry['change']}/{entry['direction']}): "
                    f"same generation, promised correct="
                    f"{entry['predicted_correct']} actual={actual[sid]}"
                )
            else:
                drift += 1
                drifted.append(entry)
                print(
                    f"  [drift ] {sid} ({entry['change']}/{entry['direction']}): "
                    f"model generated different calls, contract not testable here"
                )
        print(
            f"\n  kept={ok} (of which {weak} on a drifted generation)  "
            f"broken={miss}  untestable_drift={drift}"
        )
        if drift:
            print(
                "  注意：drift 不是违约，但它意味着这些样本没有验证到承诺。"
                "若 drift 集中在 gain/loss 上，收益结论就没有被证实。"
            )
        for grp, name in ((broken, "broken"), (drifted, "drift")):
            g = sum(1 for e in grp if e["direction"] == "gain")
            l_ = sum(1 for e in grp if e["direction"] == "loss")
            if grp:
                print(f"    {name}: gain {g} / loss {l_} / neutral {len(grp)-g-l_}")
        raise SystemExit(0 if miss == 0 else 1)

    print(f"CONTRACT CHECK  base={args.base}  (production code as it stands now)")
    print("=" * 78)

    gate_fail, released = check_gate(args.base)
    print(f"\n[Change G] gate released on {len(released)} sample(s)")
    for e in released:
        print(f"    {e['direction']:<8} {e['category']:<20} {e['id']}")
    for f in gate_fail:
        print(f"    !! {f}")

    snap_fail, affected, compared = check_snap(args.base)
    print(f"\n[Change H] compared {compared} sample(s), "
          f"{len(affected)} value(s) rewritten")
    for e in affected:
        tag = "" if e.get("basis") == "replay" else f"   [{e['basis']}]"
        print(f"    {e['direction']:<8} {e['category']:<20} {e['id']}{tag}")
    basis_n: dict[str, int] = {}
    for e in affected:
        basis_n[e.get("basis", "replay")] = basis_n.get(e.get("basis", "replay"), 0) + 1
    if set(basis_n) - {"replay"}:
        print("\n  证据基础分布（非 replay 的说明见 check_snap 注释）:")
        for b, n in sorted(basis_n.items()):
            note = {
                "replay": "离线判分复现了线上判定，直接回放",
                "exact_match_after": "吸附后与 GT 完全相等，严格判分器也判对",
                "delta_invariant": "吸附不改变离线判定，无回退机制",
            }.get(b, "")
            print(f"    {b:<20}{n:>4}   {note}")
    for f in snap_fail[:5]:
        print(f"    !! {f}")

    failures = gate_fail + snap_fail
    all_affected = (
        [dict(e, change="G") for e in released]
        + [dict(e, change="H") for e in affected]
    )
    counts = {"gain": 0, "loss": 0, "neutral": 0}
    for e in all_affected:
        counts[e["direction"]] += 1

    print("\n" + "=" * 78)
    print(f"  promised gains   : {counts['gain']}")
    print(f"  promised losses  : {counts['loss']}")
    print(f"  promised neutral : {counts['neutral']}")
    print(f"  contract failures: {len(failures)}")

    if failures:
        print("\nCONTRACT FAILED — do not deploy.")
        raise SystemExit(1)

    promise_path.write_text(
        json.dumps(
            {
                "base": args.base,
                "target": args.out,
                "counts": counts,
                "invariants_checked": sorted(GATE_MUST_STILL_FIRE),
                "affected": all_affected,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nCONTRACT PASSED. promise written to {promise_path}")


if __name__ == "__main__":
    main()
