"""Run 500-prompt evaluation and report accuracy."""
import json
import sys
sys.path.insert(0, ".")

from carm.router import CARMRouter

with open("configs/diverse_eval.json", encoding="utf-8") as f:
    data = json.load(f)

prompts = data if isinstance(data, list) else data.get("prompts", data)

router = CARMRouter()

total = 0
passed = 0
failed_list = []

for item in prompts:
    pid = item.get("id", f"p{total+1}")
    prompt = item["prompt"]
    expected = item.get("expected_tool", item.get("expected", ""))
    if not expected:
        continue
    total += 1
    result = router.route(prompt)
    got = result.tool_name
    if got == expected:
        passed += 1
    else:
        failed_list.append((pid, prompt, expected, got))

accuracy = passed / total * 100 if total > 0 else 0
print(f"\n{'='*60}")
print(f"Total: {total}")
print(f"Passed: {passed}")
print(f"Failed: {total - passed}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"\nFailed ({len(failed_list)}):")
for pid, prompt, exp, got in failed_list:
    print(f"  {pid}: exp={exp:15s} got={got:15s} | {prompt[:80]}")
