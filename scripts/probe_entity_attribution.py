"""Change N 探针：参数抽取的「实体归属」约束（prompt 级，零额外调用成本）。

方法论要求：
  1. 先做保真度校验 —— 旧 prompt 必须复现出诊断数据里已观测到的错误输出，
     否则这个离线镜像不可用于决策（教训 #5）。
  2. 命中率只在「待干预样本」上实测，禁止用自发正确的样本估计（教训 #14）。
  3. 同时在「当前判对样本」上跑，量 LOSS。

用法：
  PYTHONPATH=scripts python scripts/probe_entity_attribution.py --mode fidelity
  PYTHONPATH=scripts python scripts/probe_entity_attribution.py --mode ab
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from collections import Counter

import httpx

from diag_weakroot_v24 import load_gt, score, WEAK

BFCL_DATA = r"D:\tools\miniconda3\envs\BFCL\Lib\site-packages\bfcl_eval\data"
OLLAMA_URL = os.environ.get("CARM_OLLAMA_URL", "http://192.168.31.20:11434")
OLLAMA_MODEL = os.environ.get("CARM_OLLAMA_MODEL", "qwen3-coder")

# ----------------------------------------------------- 新增的实体归属规则
ATTRIBUTION_RULES = """
32. ENTITY ATTRIBUTION (highest priority — overrides rule 2): create an object ONLY for
    entities the query actually pairs with THIS function. Do NOT apply this function to
    every entity you see just because several entities appear in the text.
    Counter-example (this is the #1 mistake): "Get temperature and humidity for Boston,
    and precipitation for Rome" — for weather_precipitation return ONLY
    [{"location":"Rome"}], NOT Boston as well. Boston belongs to the other functions.
    Counter-example: "overview of the Battle of Waterloo and the signing of the Treaty of
    Tordesillas" — for battle_details return ONLY [{"battle_name":"Battle of Waterloo"}].
    Rule 2's cartesian expansion applies ONLY when the query gives this function's own verb
    or noun a list of entities ("weather in Boston AND San Francisco" — one verb, two cities).
"""


def load_schemas(cat: str) -> dict:
    p = os.path.join(BFCL_DATA, f"BFCL_v4_{cat}.json")
    out = {}
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        it = json.loads(line)
        out[it["id"]] = {f["name"]: f for f in it.get("function", [])}
    return out


def build_param_desc(func: dict) -> str:
    params = func.get("parameters", {})
    props = params.get("properties", {})
    required = params.get("required", [])
    lines = []
    for pname, pinfo in props.items():
        ptype = pinfo.get("type", "any")
        pdesc = pinfo.get("description", "")
        req = "req" if pname in required else "opt"
        enum_vals = pinfo.get("enum")
        enum_str = f" enum={enum_vals}" if enum_vals else ""
        nested = pinfo.get("properties")
        nested_str = ""
        if nested:
            ks = []
            for nn, ni in nested.items():
                ne = ni.get("enum")
                ks.append(f"{nn}({ni.get('type','any')}{f' enum={ne}' if ne else ''})")
            nested_str = f" [nested: {', '.join(ks)}]"
        lines.append(f"  - {pname} ({ptype}, {req}{enum_str}){nested_str}: {pdesc}")
    return "\n".join(lines) if lines else "  (none)"


# 精简复刻服务端 prompt 的骨架（保真度由 --mode fidelity 校验）
BASE_RULES = """CRITICAL RULES:
1. Return JSON array of objects, one per call.
2. If the query mentions MULTIPLE entities (cities, people, items, dates) that each need this function, create one object PER entity.
   Example: "weather in Boston and San Francisco" -> [{"location":"Boston, MA"}, {"location":"San Francisco, CA"}]
3. Do NOT invent extra calls for unrelated functions.
4. Use correct types (int/float/str/bool). Omit missing optional params.
5. Use enum values EXACTLY as listed. Do NOT iterate over enum values - pick ONE based on the query.
6. For location params, use the location string AS IT APPEARS in the query.
7. For boolean params, use JSON true/false.
8. If the query asks for the same thing only once, return exactly ONE object.
12. Do NOT return duplicate objects with the same params.
17. Do NOT add optional params if the query does not mention them.
31. NEVER invent a value that appears in neither the Query nor the Full request.
"""


def make_prompt(func: dict, query: str, with_attribution: bool) -> str:
    rules = BASE_RULES + (ATTRIBUTION_RULES if with_attribution else "")
    return f"""Extract ALL params for: {func.get('name','')}

Schema:
{build_param_desc(func)}

Query: {query}

{rules}
Examples:
Simple: [{{"a":1,"b":2}}]
Multiple (explicit): [{{"a":1,"b":2}},{{"a":3,"b":4}}]"""


def call_llm(prompt: str) -> list[dict]:
    with httpx.Client(timeout=180.0) as client:
        resp = client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": "Output only a JSON array of objects."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.001, "num_predict": 300},
            },
        )
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
    m = re.search(r"\[.*\]", content, re.S)
    if not m:
        return []
    try:
        v = json.loads(m.group(0))
        return [x for x in v if isinstance(x, dict)]
    except Exception:  # noqa: BLE001
        return []


def pick_samples(limit_bad: int, limit_good: int):
    """挑选待干预样本（COUNT/over）与对照样本（当前判对）。"""
    bad, good = [], []
    for cat in ["parallel_multiple", "live_parallel_multiple"]:
        path = WEAK.get(cat)
        if not path:
            continue
        gtm = load_gt(cat)
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("error"):
                continue
            gt = gtm.get(r["id"])
            if r.get("correct"):
                if len(good) < limit_good and len(r.get("pred", [])) > 1:
                    good.append((cat, r, gt))
                continue
            if score(r["pred"], gt, cat) == "COUNT" and len(r["pred"]) > len(gt or []):
                if len(bad) < limit_bad:
                    bad.append((cat, r, gt))
    return bad, good


def run_sample(cat, r, gt, schemas, with_attr):
    """对样本里出现过的每个函数重新抽取参数，拼成新的调用集合。"""
    sch = schemas[cat].get(r["id"], {})
    names = []
    for c in r["pred"]:
        if c.get("name") not in names:
            names.append(c["name"])
    calls = []
    for fn in names:
        f = sch.get(fn)
        if not f:
            return None
        for ps in call_llm(make_prompt(f, r.get("query", ""), with_attr)):
            calls.append({"name": fn, "arguments": ps})
    return calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fidelity", "ab"], default="fidelity")
    ap.add_argument("--bad", type=int, default=12)
    ap.add_argument("--good", type=int, default=12)
    args = ap.parse_args()

    schemas = {c: load_schemas(c) for c in ["parallel_multiple", "live_parallel_multiple"]}
    bad, good = pick_samples(args.bad, args.good)
    print(f"待干预样本 {len(bad)} / 对照样本 {len(good)}   model={OLLAMA_MODEL}")

    if args.mode == "fidelity":
        # 教训 #5：旧 prompt 必须复现出诊断里已观测到的错误（调用数偏多）
        same_count = repro = 0
        for cat, r, gt in bad:
            out = run_sample(cat, r, gt, schemas, with_attr=False)
            if out is None:
                continue
            obs = len(r["pred"])
            gtn = len(gt or [])
            print(f"  [{r['id']}] 观测={obs} 复刻={len(out)} GT={gtn}")
            if len(out) == obs:
                same_count += 1
            if len(out) > gtn:
                repro += 1
        n = len(bad)
        print(f"\n保真度：调用数完全一致 {same_count}/{n}；复现出「多调」形态 {repro}/{n}")
        print("判据：复现率 < 70% 则该镜像不可用于决策，需要改用服务端实跑。")
        return

    # A/B —— 教训 #6：必须在同一镜像里同时跑 OFF/ON 基线，否则分不清
    # 「归属规则的效应」与「简化 prompt 的镜像偏差」。
    def evaluate(pool, label):
        rows = []
        for cat, r, gt in pool:
            o_off = run_sample(cat, r, gt, schemas, with_attr=False)
            o_on = run_sample(cat, r, gt, schemas, with_attr=True)
            if o_off is None or o_on is None:
                continue
            s_off = score(o_off, gt, cat)
            s_on = score(o_on, gt, cat)
            rows.append((r["id"], len(gt or []), len(o_off), s_off, len(o_on), s_on))
        print(f"\n--- {label} ---")
        print(f"    {'样本':<28}{'GT':>4}{'OFF数':>7}{'OFF判':>10}{'ON数':>7}{'ON判':>10}  差分")
        d = Counter()
        for sid, g, n0, s0, n1, s1 in rows:
            ok0, ok1 = s0 == "OK", s1 == "OK"
            if ok0 and not ok1:
                delta, key = "LOSS", "LOSS"
            elif ok1 and not ok0:
                delta, key = "GAIN", "GAIN"
            else:
                delta, key = "-", "NEUTRAL"
            d[key] += 1
            print(f"    {sid:<28}{g:>4}{n0:>7}{s0:>10}{n1:>7}{s1:>10}  {delta}")
        print(f"    小计: GAIN={d['GAIN']} LOSS={d['LOSS']} NEUTRAL={d['NEUTRAL']}")
        return d

    db = evaluate(bad, "待干预样本（OFF vs ON）")
    dg = evaluate(good, "对照样本（OFF vs ON）")
    net = (db["GAIN"] + dg["GAIN"]) - (db["LOSS"] + dg["LOSS"])
    print(f"\n>>> 归属规则的真实差分效应: GAIN={db['GAIN']+dg['GAIN']} "
          f"LOSS={db['LOSS']+dg['LOSS']} 净={net:+d}")
    print("    注：这是「同一简化镜像内 OFF→ON」的差分，已剔除镜像本身的偏差。")


if __name__ == "__main__":
    main()
