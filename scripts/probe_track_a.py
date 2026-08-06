"""Track A 离线探针：并行调用数纠正（只删不加）。

规则必须是 GT-free 的（只用 query / schema 名 / 调用本身），GT 只用于打分。

策略：
  d1  : 无词法证据的多余函数删除
        —— 函数名 token 在 query 里完全没有证据的调用，删掉
  d3  : 同名重复的邻近性收敛
        —— 同名函数被调用 k 次，但 query 中该函数的触发 token 只出现 t 次 (t<k)，
           按"参数值在 query 中的位置 与 函数 token 位置"的最小距离排序，只保留 t 个
  d1+d3: 两者串联

三态对账（教训 #9：守差分，不静默丢弃）：
  UNCHANGED     策略未改变调用集合 -> 无风险，不计入
  GAIN          离线 FAIL -> OK，且线上 correct=False（真实上行）
  LOSS          离线 OK   -> FAIL，或线上 correct=True 但改动后离线 FAIL（真实下行）
  UNPREDICTABLE 线上 correct=True 但离线判 FAIL，且策略改了调用 -> 无法预测，显式报错

用法：PYTHONPATH=scripts python scripts/probe_track_a.py
"""
from __future__ import annotations
import json
import re
from collections import Counter, defaultdict

from diag_weakroot_v24 import load_gt, score, WEAK

PARALLEL_CATS = ["parallel_multiple", "live_parallel", "live_parallel_multiple"]


# ---------------------------------------------------------------- 词法工具

def name_tokens(fname: str):
    """函数名切成有意义的 token（去掉命名空间噪声）。"""
    parts = re.split(r"[._\-]", fname)
    toks = []
    for p in parts:
        toks += re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|\d+", p)
    return [t.lower() for t in toks if len(t) > 2]


STOP = {"get", "find", "calculate", "info", "data", "value", "result",
        "the", "and", "for", "with", "from", "toolkit", "tool", "api"}


def signal_tokens(fname: str, all_names: list):
    """取该函数区别于同 schema 其他函数的判别性 token。"""
    mine = set(name_tokens(fname)) - STOP
    others = set()
    for n in all_names:
        if n != fname:
            others |= set(name_tokens(n))
    disc = mine - others
    return disc if disc else mine


def token_positions(tok: str, q: str):
    """token 在 query 中出现的位置（宽松匹配：允许词干前缀）。"""
    pos = []
    for m in re.finditer(re.escape(tok[:max(4, len(tok) - 2)]), q):
        pos.append(m.start())
    return pos


def func_trigger_positions(fname: str, all_names: list, q: str):
    """函数在 query 中被触发的位置集合（各判别 token 位置的并集，聚类后去重）。"""
    ql = q.lower()
    pos = set()
    for t in signal_tokens(fname, all_names):
        for p in token_positions(t, ql):
            pos.add(p)
    if not pos:
        return []
    # 邻近位置聚类（30 字符内视为同一次触发）
    sp = sorted(pos)
    clusters = [sp[0]]
    for p in sp[1:]:
        if p - clusters[-1] > 30:
            clusters.append(p)
    return clusters


def arg_positions(args: dict, q: str):
    """调用参数的字面值在 query 中出现的位置。"""
    ql = q.lower()
    pos = []
    for v in (args or {}).values():
        for s in (v if isinstance(v, list) else [v]):
            if isinstance(s, str) and len(s) > 2:
                i = ql.find(s.lower())
                if i >= 0:
                    pos.append(i)
            elif isinstance(s, (int, float)) and not isinstance(s, bool):
                i = ql.find(str(s))
                if i >= 0:
                    pos.append(i)
    return pos


# ---------------------------------------------------------------- 策略

def strat_d1(calls, query, all_names):
    """删掉函数名 token 在 query 里完全没有词法证据的调用。"""
    if len(calls) <= 1:
        return calls
    keep = []
    for c in calls:
        trig = func_trigger_positions(c["name"], all_names, query)
        if trig:
            keep.append(c)
    return keep if keep else calls


def strat_d3(calls, query, all_names):
    """同名重复的邻近性收敛：query 中触发 t 次，就只保留 t 个最贴近的调用。"""
    by_name = defaultdict(list)
    for c in calls:
        by_name[c["name"]].append(c)
    keep = []
    for fn, group in by_name.items():
        if len(group) <= 1:
            keep += group
            continue
        trig = func_trigger_positions(fn, all_names, query)
        t = len(trig)
        if t == 0 or t >= len(group):
            keep += group
            continue
        # 按 (参数位置 与 最近触发点) 的最小距离排序，保留最近的 t 个
        scored = []
        for c in group:
            ap = arg_positions(c.get("arguments", {}), query)
            if not ap:
                d = 10 ** 6
            else:
                d = min(abs(a - p) for a in ap for p in trig)
            scored.append((d, c))
        scored.sort(key=lambda x: x[0])
        keep += [c for _, c in scored[:t]]
    # 保持原顺序
    order = {id(c): i for i, c in enumerate(calls)}
    keep.sort(key=lambda c: order[id(c)])
    return keep


STRATS = {
    "d1": lambda c, q, n: strat_d1(c, q, n),
    "d3": lambda c, q, n: strat_d3(c, q, n),
    "d1+d3": lambda c, q, n: strat_d3(strat_d1(c, q, n), q, n),
}


# ---------------------------------------------------------------- 探针

def sig(calls):
    return json.dumps([[c.get("name"), c.get("arguments")] for c in calls], sort_keys=True)


def run(name, fn):
    tally = Counter()
    gains, losses, unpred = [], [], []

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
            all_names = r.get("func_names", [])
            online_ok = bool(r.get("correct"))

            try:
                rep = fn([dict(c) for c in pred], query, all_names)
            except Exception as e:  # noqa: BLE001
                tally["STRAT_ERROR"] += 1
                continue

            if sig(rep) == sig(pred):
                tally["UNCHANGED"] += 1
                continue

            before = score(pred, gt, cat) == "OK"
            after = score(rep, gt, cat) == "OK"

            # 教训 #9：只有"线上对 + 离线判错 + 调用被改"才是真正无法预测
            if online_ok and not before:
                tally["UNPREDICTABLE"] += 1
                unpred.append((cat, r["id"]))
                continue

            if online_ok and before and not after:
                tally["LOSS"] += 1
                losses.append((cat, r["id"], r.get("gt_names"), r.get("pred_names")))
            elif (not online_ok) and (not before) and after:
                tally["GAIN"] += 1
                gains.append((cat, r["id"], r.get("gt_names"), r.get("pred_names")))
            elif before and not after:
                tally["LOSS"] += 1
                losses.append((cat, r["id"], r.get("gt_names"), r.get("pred_names")))
            else:
                tally["CHANGED_NEUTRAL"] += 1

    net = tally["GAIN"] - tally["LOSS"]
    print(f"\n{'='*66}\n策略 {name}\n{'='*66}")
    for k in ["UNCHANGED", "CHANGED_NEUTRAL", "GAIN", "LOSS", "UNPREDICTABLE", "STRAT_ERROR"]:
        if tally[k]:
            print(f"   {k:16s} : {tally[k]}")
    print(f"   >>> 净收益 = {net:+d}")
    if gains:
        print("   GAIN 明细:")
        for cat, sid, gn, pn in gains[:12]:
            print(f"     + {sid}")
    if losses:
        print("   LOSS 明细:")
        for cat, sid, gn, pn in losses[:12]:
            print(f"     - {sid}  GT={gn}")
            print(f"                PRED={pn}")
    if unpred:
        print(f"   UNPREDICTABLE（线上对/离线判错/被改动）: {[s for _, s in unpred[:10]]}")
    return net


def main():
    print("Track A 离线探针 — 并行调用数纠正（只删不加）")
    print(f"样本范围: {PARALLEL_CATS}")
    results = {}
    for name, fn in STRATS.items():
        results[name] = run(name, fn)
    print(f"\n{'='*66}\n汇总\n{'='*66}")
    for k, v in results.items():
        print(f"   {k:8s} 净收益 {v:+d}")


if __name__ == "__main__":
    main()
