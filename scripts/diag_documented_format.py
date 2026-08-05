#!/usr/bin/env python3
"""测量（不修复）：schema description 里声明了取值格式的参数，生成端遵守了吗？

这是 `enforce_documented_value_formats_declared_in_parameter_descriptions.md`
提案的第一道前置检验，目的是**先给提案一次被证伪的机会**：

  提案假设 76 例 live_multiple 失败是"模型没遵守 description 里写死的格式"。
  如果多数声明了格式的参数其实已经合规，那这 76 例另有共因，提案的归因是错的。

口径故意收窄到提案证据的那一族——"逗号两段式"格式声明：

    "in the format 'City, State' or 'City, Country'"
    "in the format of 'City, State', such as 'Los Angeles, CA'"

判定"声明了格式"的唯一依据是 description 里的**带引号示例**，
且要求示例全部含逗号。不内置任何城市/州名映射表——
一旦需要映射表才能判定，那就不是从 schema 推出来的，属于刷榜（提案已否决）。

四象限是这个脚本的产出重点：

    pred 合规 / gt 也要格式   → 本来就对，无关
    pred 不合规 / gt 要格式   → 可修复上限
    pred 不合规 / gt 不要格式 → **修了会破坏**（风险）
    pred 合规 / gt 不要格式   → 已经在被这个格式坑（反向证据）

用法:
    python scripts/diag_documented_format.py v22
    python scripts/diag_documented_format.py v22 --show fixable
    python scripts/diag_documented_format.py v22 --show risk
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DIAG = Path(__file__).resolve().parent.parent / "data" / "eval" / "diag"

CATEGORIES = [
    "simple_python", "simple_java", "simple_javascript",
    "multiple", "parallel", "parallel_multiple",
    "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple",
]

# description 里被引号包住的示例值。长度上限防止把整句话吞进来。
QUOTED = re.compile(r"['\"`]([^'\"`\n]{2,60})['\"`]")
# 只有出现这些引导词时，引号内容才当成"格式示例"，否则可能只是随口举例。
FORMAT_CUE = re.compile(
    r"\b(in the format|format of|formatted as|should be in|must be in|"
    r"such as|for example|e\.g\.)", re.I)


def lit(x, default=None):
    if x is None:
        return default
    if not isinstance(x, str):
        return x
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return default


def shape(s) -> str:
    """把字符串抽象成结构签名。只区分"有没有逗号分段"和尾段形态。

    不认识具体地名，只认识结构——这是它不会退化成映射表的原因。
    """
    t = str(s).strip()
    if "," not in t:
        return "NOCOMMA"
    tail = t.rpartition(",")[2].strip()
    if re.fullmatch(r"[A-Z]{2}", tail):
        return "COMMA_UPPER2"
    if re.fullmatch(r"[A-Za-z][A-Za-z .]{1,30}", tail):
        return "COMMA_WORD"
    return "COMMA_OTHER"


def declares_comma_format(spec: dict) -> tuple[bool, list[str]]:
    """该参数的 description 是否声明了"值必须是逗号两段式"？

    条件（全部满足才算）:
      1. description 出现格式引导词
      2. 至少 2 个带引号示例（1 个太弱，可能只是举个值）
      3. **所有**示例都含逗号——只要有一个裸值示例，就说明裸值也被接受
    """
    if not isinstance(spec, dict):
        return False, []
    desc = str(spec.get("description") or "")
    items = spec.get("items")
    if isinstance(items, dict):
        desc += " " + str(items.get("description") or "")
    if not FORMAT_CUE.search(desc):
        return False, []
    examples = [e.strip() for e in QUOTED.findall(desc)]
    # 过滤掉明显不是"值"的引号内容：整句、含冒号的说明、
    # 以及纯标点——`music_theory.key_signature` 的描述里引了一个 ', '
    # 当分隔符说明，早期版本把它当成了"格式示例"，凭空造出一个假阳性位点。
    examples = [e for e in examples
                if len(e.split()) <= 6 and ":" not in e
                and len(e) >= 3 and re.search(r"[A-Za-z]", e)]
    if len(examples) < 2:
        return False, examples
    if not all(shape(e).startswith("COMMA") for e in examples):
        return False, examples
    return True, examples


def gt_values_for(gt: list, fname: str, pname: str) -> list | None:
    """gt 对该函数该参数的可接受值列表；参数不在 gt 里返回 None。"""
    for call in gt or []:
        if not isinstance(call, dict):
            continue
        gname, gargs = next(iter(call.items()))
        if gname != fname or not isinstance(gargs, dict):
            continue
        if pname in gargs:
            acc = gargs[pname]
            return acc if isinstance(acc, list) else [acc]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--show", choices=["fixable", "risk", "compliant", "declared"])
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    from eval_bfcl_v4_fast import build_messages, load_bfcl_data

    n_rows = n_params = 0
    declared_params: set[tuple[str, str]] = set()   # (func, param) 声明了格式的
    all_params: set[tuple[str, str]] = set()
    quad = collections.Counter()
    per_cat = collections.defaultdict(collections.Counter)
    # 位点级 GT 自洽性：同一个 func.param 上，gt 是不是始终要求同一种格式？
    site_gt = collections.defaultdict(collections.Counter)
    fixable, risk, compliant_rows = [], [], []
    # 触发面：一次"定向重问"会打到多少个样本
    would_requery_ids: set[str] = set()
    requery_by_cat = collections.Counter()

    for cat in CATEGORIES:
        path = DIAG / f"{cat}_{args.tag}.jsonl"
        if not path.exists():
            continue
        fmap_by_id = {}
        try:
            for item in load_bfcl_data(cat):
                _, fs = build_messages(item)
                fmap_by_id[item.get("id", "")] = {f.get("name"): f for f in fs
                                                  if isinstance(f, dict)}
        except Exception as exc:            # 数据缺失就跳过，不静默假装扫过
            print(f"  [warn] {cat}: schema 加载失败 {exc}", file=sys.stderr)
            continue

        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            pred = lit(row.get("pred")) or []
            gt = lit(row.get("gt")) or []
            if not pred:
                continue
            n_rows += 1
            was_correct = str(row.get("correct")) == "True"
            fmap = fmap_by_id.get(row.get("id", ""), {})

            for call in pred:
                if not isinstance(call, dict):
                    continue
                fname = call.get("name")
                func = fmap.get(fname) or {}
                props = (func.get("parameters") or {}).get("properties") or {}
                for pname, pv in (call.get("arguments") or {}).items():
                    if not isinstance(pv, str):
                        continue
                    n_params += 1
                    all_params.add((fname, pname))
                    ok, examples = declares_comma_format(props.get(pname) or {})
                    if not ok:
                        continue
                    declared_params.add((fname, pname))

                    pred_ok = shape(pv).startswith("COMMA")
                    gtv = gt_values_for(gt, fname, pname)
                    if gtv is None:
                        gt_kind = "absent"
                    elif all(shape(g).startswith("COMMA") for g in gtv if isinstance(g, str)):
                        gt_kind = "needs_format"
                    elif any(shape(g).startswith("COMMA") for g in gtv if isinstance(g, str)):
                        gt_kind = "either"
                    else:
                        gt_kind = "bare_ok"

                    key = ("pred_ok" if pred_ok else "pred_bad", gt_kind)
                    quad[key] += 1
                    per_cat[cat][key] += 1
                    if gt_kind != "absent":
                        site_gt[(fname, pname)][gt_kind] += 1

                    rec = {
                        "id": row.get("id"), "cat": cat, "func": fname,
                        "param": pname, "pred": pv, "gt": gtv,
                        "correct": was_correct, "examples": examples[:3],
                    }
                    if not pred_ok:
                        would_requery_ids.add(row.get("id"))
                        requery_by_cat[cat] += 1
                        if gt_kind in ("needs_format", "either"):
                            fixable.append(rec)
                        else:
                            risk.append(rec)
                    else:
                        compliant_rows.append(rec)

    print(f"===== DOCUMENTED-FORMAT COMPLIANCE ({args.tag}) =====")
    print(f"扫描样本 {n_rows}，字符串参数实例 {n_params}")
    print(f"参数位点 (func.param) 共 {len(all_params)} 个，"
          f"其中声明了逗号格式的 {len(declared_params)} 个")
    print()

    total = sum(quad.values())
    if not total:
        print("没有任何参数实例命中'声明了逗号两段式格式'。")
        print("→ 提案的归因不成立，或识别规则太严。先看 --show declared 再决定。")
        return

    print(f"命中实例 {total}：")
    header = f"  {'':<10} {'gt需格式':>10} {'gt两可':>8} {'gt裸值可':>10} {'gt无此参':>10}"
    print(header)
    for pk, label in (("pred_ok", "pred合规"), ("pred_bad", "pred不合规")):
        cells = [quad[(pk, g)] for g in ("needs_format", "either", "bare_ok", "absent")]
        print(f"  {label:<8} {cells[0]:>12} {cells[1]:>8} {cells[2]:>11} {cells[3]:>11}")
    print()

    n_bad = sum(quad[("pred_bad", g)] for g in
                ("needs_format", "either", "bare_ok", "absent"))
    n_ok = total - n_bad
    print(f"总体合规率: {n_ok}/{total} = {n_ok / total:.1%}")
    print()

    print("--- 修复上限与风险 ---")
    fix_fail = [r for r in fixable if not r["correct"]]
    fix_pass = [r for r in fixable if r["correct"]]
    risk_pass = [r for r in risk if r["correct"]]
    risk_fail = [r for r in risk if not r["correct"]]
    print(f"  可修复 (pred不合规 & gt要格式) : {len(fixable):>4}"
          f"   其中样本当前判错 {len(fix_fail)} / 判对 {len(fix_pass)}")
    print(f"  会破坏 (pred不合规 & gt接受裸值): {len(risk):>4}"
          f"   其中样本当前判对 {len(risk_pass)} / 判错 {len(risk_fail)}")
    print(f"  → 收益上限 +{len(fix_fail)}（判错的才可能翻正），"
          f"风险下限 -{len(risk_pass)}（判对的会被改坏）")
    print()

    # 定向重问的收益完全取决于"模型补出来的后缀对不对"。
    # 不用假设，也不用离线镜像——真实系统已经自发按格式输出过 292 次，
    # 那批输出的正确率就是干预成功率最诚实的代理指标。
    print("--- 代理指标：模型自发按格式输出时，值对了吗？---")
    hit = miss = 0
    miss_examples = []
    for r in compliant_rows:
        gtv = r["gt"]
        if not gtv:
            continue
        if any(r["pred"] == g for g in gtv):
            hit += 1
        else:
            miss += 1
            if len(miss_examples) < 8:
                miss_examples.append(r)
    tot = hit + miss
    if tot:
        print(f"  pred 已合规且 gt 有该参数: {tot} 例，"
              f"值命中 {hit} / 不命中 {miss} → **{hit / tot:.1%}**")
        print(f"  这就是'定向重问'能达到的上界估计。"
              f"按此估计净收益 ≈ {len([r for r in fixable if not r['correct']]) * hit / tot:.0f}"
              f" - {len([r for r in risk if r['correct']])}"
              f" = {len([r for r in fixable if not r['correct']]) * hit / tot - len([r for r in risk if r['correct']]):.0f}")
        for r in miss_examples:
            print(f"    [miss] {r['id']:<26} pred={r['pred']!r} gt={r['gt']!r}")
    print()

    print("--- GT 自洽性：同一参数位点上，GT 的格式要求一致吗？---")
    consistent = inconsistent = 0
    incon_sites = []
    for site, kinds in site_gt.items():
        if kinds.get("needs_format") and (kinds.get("bare_ok") or kinds.get("either")):
            inconsistent += 1
            incon_sites.append((site, kinds))
        else:
            consistent += 1
    print(f"  有 GT 数据的位点 {len(site_gt)} 个："
          f"要求一致 {consistent} / **自相矛盾 {inconsistent}**")
    for (f, p), kinds in sorted(incon_sites,
                                key=lambda x: -sum(x[1].values()))[:12]:
        detail = " ".join(f"{k}={v}" for k, v in kinds.most_common())
        print(f"    {f}.{p:<18} {detail}")
    if inconsistent:
        print("  → 同一个 schema 声明下 GT 两种都收，'照 description 补格式'")
        print("     在这些位点上是掷硬币，不是修复。")
    print()

    print("--- 触发面（一次定向重问会打到多少样本）---")
    print(f"  受影响样本 {len(would_requery_ids)} 个，新增 LLM 调用 "
          f"{sum(requery_by_cat.values())} 次")
    for cat, n in requery_by_cat.most_common():
        print(f"    {cat:<24} {n}")
    print()

    if args.show:
        pool = {"fixable": fixable, "risk": risk,
                "compliant": compliant_rows, "declared": None}[args.show]
        if args.show == "declared":
            print(f"--- 声明了逗号格式的参数位点 ({len(declared_params)}) ---")
            for f, p in sorted(declared_params):
                print(f"  {f}.{p}")
            return
        print(f"--- {args.show} ({len(pool)}) ---")
        for r in pool[:args.limit]:
            mark = "OK " if r["correct"] else "ERR"
            print(f"  [{mark}] {r['id']:<28} {r['func']}.{r['param']}")
            print(f"         pred={r['pred']!r}  gt={r['gt']!r}")
            print(f"         examples={r['examples']}")


if __name__ == "__main__":
    main()
