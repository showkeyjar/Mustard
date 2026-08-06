"""Track B 离线探针（保守版 B2：仅 list-of-dict 包裹）。

对预测参数做 schema 驱动的保守修复：
  - 仅当 schema 声明该参数为 object/array/HashMap 类型，且预测值是 dict（非 list）时，
    把该值包裹为 [dict]。
  - 不改键名（B1 暂不做，避免启发式引入 LOSS）、不删键、不碰标量。

然后用 diag_weakroot_v24 的忠实评分器重算，统计：
  GAIN   = 修复前错、修复后对
  LOSS   = 修复前对、修复后错   （应为 0 才安全）
  NEUTRAL= 修复前后都错/都对

用法：python scripts/probe_track_b.py
"""
from __future__ import annotations
import json, os
from collections import Counter
from diag_weakroot_v24 import load_gt, score, IRRELEVANCE_CATS, WEAK

DD = r"D:\tools\miniconda3\envs\BFCL\Lib\site-packages\bfcl_eval\data"
WRAP_TYPES = {"object", "array", "HashMap", "dict", "list"}

WEAK.update({})  # 保持与 diag_weakroot_v24 一致


def load_schemas(cat: str) -> dict:
    """返回 {func_name: {param_name: param_def}}。"""
    out = {}
    p = os.path.join(DD, f"BFCL_v4_{cat}.json")
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        it = json.loads(line)
        for fn in it.get("function", []):
            if isinstance(fn, dict) and fn.get("name"):
                props = fn.get("parameters", {}).get("properties", {})
                out[fn["name"]] = props
    return out


def repair(calls, schemas: dict, strategy: str = "b2"):
    """calls: list of (name, params). 返回修复后的 (name, params)。

    strategy:
      b2   : 任意 object/array/HashMap/dict/list 类型且值为 dict → 包裹 [dict]
      hash  : 仅 BFCL 自定义 HashMap 类型且值为 dict → 包裹 [dict]
      hash+rename : hash + 键名对齐（预测键 k 不在 schema，但 schema 存在以 k 为子串的键 sk，
                   且 pred[k] 为 dict 时 sk 也期望 HashMap）→ 重映射
    """
    out = []
    for name, params in calls:
        if not isinstance(params, dict):
            out.append((name, params))
            continue
        sdef = schemas.get(name, {})
        new_params = dict(params)
        for k, v in list(params.items()):
            pdef = sdef.get(k, {})
            ptype = pdef.get("type") if isinstance(pdef, dict) else None
            if strategy in ("b2", "hash", "hash+rename") and isinstance(v, dict):
                do_wrap = (ptype in WRAP_TYPES) if strategy == "b2" else (ptype == "HashMap")
                if do_wrap:
                    new_params[k] = [v]
            if strategy == "hash+rename" and isinstance(v, dict) and k not in sdef:
                # 找 schema 中 k 为其子串的键
                for sk, sv in sdef.items():
                    st = sv.get("type") if isinstance(sv, dict) else None
                    if k != sk and k in sk and (st == "HashMap" or st in WRAP_TYPES):
                        new_params[sk] = [v]
                        if k in new_params:
                            del new_params[k]
                        break
        out.append((name, new_params))
    return out


def to_calls(pred):
    return [(c.get("name"), c.get("arguments", {})) for c in pred]


def calls_to_dicts(calls):
    """(name, params) 元组 -> {name, arguments} 字典（供 score 使用）。"""
    return [{"name": n, "arguments": p} for n, p in calls]


def evaluate(strategy):
    g = Counter()
    for cat, path in WEAK.items():
        schemas = load_schemas(cat)
        gt = load_gt(cat)
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        for r in rows:
            if r.get("error"):
                continue
            pid = r["id"]
            base = calls_to_dicts(to_calls(r["pred"]))
            before = score(base, gt.get(pid), cat)
            after = score(calls_to_dicts(repair(to_calls(r["pred"]), schemas, strategy)), gt.get(pid), cat)
            if before == "OK":
                if after != "OK":
                    g["LOSS"] += 1
            else:
                if after == "OK":
                    g["GAIN"] += 1
                else:
                    g["NEUTRAL"] += 1
    return g


def main():
    for strat in ("b2", "hash", "hash+rename"):
        g = evaluate(strat)
        print(f"=== strategy={strat} ===  GAIN={g['GAIN']}  LOSS={g['LOSS']}  NEUTRAL={g['NEUTRAL']}  net={g['GAIN']-g['LOSS']:+d}")


if __name__ == "__main__":
    main()
