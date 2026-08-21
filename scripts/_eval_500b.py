"""Run 500-prompt evaluation and write results to file."""
import json
import sys

sys.path.insert(0, ".")
from carm.router import CARMRouter

d = json.load(open("configs/diverse_eval.json", encoding="utf-8"))
prompts = d.get("prompts", [])
router = CARMRouter()
total = passed = 0
failed = []

for i, item in enumerate(prompts):
    exp = item.get("expected", "")
    if not exp:
        continue
    total += 1
    got = router.route(item["prompt"]).tool_name
    if got == exp:
        passed += 1
    else:
        failed.append((item.get("id", "?"), item["prompt"][:60], exp, got))
    if (i + 1) % 100 == 0:
        with open("scripts/_eval_progress.txt", "w") as f:
            f.write(f"Progress: {i+1}/{len(prompts)}, passed={passed}/{total}\n")

accuracy = passed / total * 100 if total > 0 else 0
results = f"Total={total} Pass={passed} Fail={total-passed} Acc={accuracy:.2f}%\n"
for fid, fp, fe, fg in failed:
    results += f"  {fid}: exp={fe:15s} got={fg:15s} | {fp}\n"

with open("scripts/_eval_results.txt", "w", encoding="utf-8") as f:
    f.write(results)

print(results)
