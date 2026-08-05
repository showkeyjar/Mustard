"""Change M 的 schema 级暴露面扫描。

回答的问题：Change M 全局开启后，**在哪些类别里可能被触发**？

和 diag_documented_format.py 的区别很关键：
  - diag_documented_format 量的是「v22 的具体预测值里，有多少不合规」——
    这是**当前生成**下的触发次数，会随生成漂移变化。
  - 本脚本量的是「schema 里存在多少个声明了逗号格式的参数位点」——
    这是**结构上限**，与生成无关。只要某类别的上限是 0，
    该类别就永远不可能被 Change M 触及，无需重跑即可断言不变。

这个区分是必须的：拿「当前恰好 0 次触发」当「永远不会触发」，
就是把观测值当成了不变量。
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from carm_bfcl_server_optimized import _declares_comma_format  # noqa: E402
from eval_bfcl_v4_fast import build_messages, load_bfcl_data  # noqa: E402

CATEGORIES = [
    "simple_python", "simple_java", "simple_javascript",
    "multiple", "parallel", "parallel_multiple",
    "irrelevance", "live_multiple", "live_irrelevance", "live_relevance",
]


def main() -> None:
    print("Change M schema 级暴露面（结构上限，与当前生成无关）")
    print("=" * 72)
    total_sites: set[tuple[str, str]] = set()
    per_cat_sites: dict[str, set] = defaultdict(set)
    per_cat_samples = Counter()
    per_cat_total = Counter()

    for cat in CATEGORIES:
        try:
            items = list(load_bfcl_data(cat))
        except Exception as exc:
            print(f"  [warn] {cat}: 数据加载失败 {exc}", file=sys.stderr)
            continue
        for item in items:
            per_cat_total[cat] += 1
            _, fs = build_messages(item)
            hit = False
            for f in fs:
                if not isinstance(f, dict):
                    continue
                props = (f.get("parameters") or {}).get("properties") or {}
                for pname, spec in props.items():
                    if _declares_comma_format(spec if isinstance(spec, dict) else {}):
                        site = (f.get("name"), pname)
                        per_cat_sites[cat].add(site)
                        total_sites.add(site)
                        hit = True
            if hit:
                per_cat_samples[cat] += 1

    print(f"{'category':<20}{'样本数':>8}{'含可触发位点的样本':>22}{'不同位点数':>12}")
    for cat in CATEGORIES:
        if cat not in per_cat_total:
            continue
        n = per_cat_total[cat]
        s = per_cat_samples[cat]
        mark = "" if s else "   <-- 结构上不可触及"
        print(f"{cat:<20}{n:>8}{s:>22}{len(per_cat_sites[cat]):>12}{mark}")

    print()
    print(f"全局不同位点数: {len(total_sites)}")
    exposed = [c for c in CATEGORIES if per_cat_samples.get(c)]
    print(f"存在暴露面的类别: {exposed or '(无)'}")
    print()
    print("结论解读：只有上表中『含可触发位点的样本』>0 的类别，")
    print("才需要在 Change M 上线后重新评测；其余类别的历史数字可直接沿用。")


if __name__ == "__main__":
    main()
