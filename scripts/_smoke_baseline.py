"""Resilient, resumable, timeout-guarded local baseline run.

Pinned snapshot (P_before) vs artifact (P_after). Key hardening over the naive
evaluate_isolated_prompts call:
- CHUNK timeout: each chunk runs in a separate process; if it exceeds CHUNK_TIMEOUT
  (default 180s) it is terminated, and the prompts become placeholder (errored)
  rows instead of hanging the whole run forever (runner.run has NO internal guard).
- RESUME: prompts whose id already exists in the output file are skipped, so a
  killed/partial run can be continued without redoing completed work.
- Incremental save after every chunk.
- Never overwrites data/eval/real_prompt_eval_latest.json.

Usage: python -m scripts._smoke_baseline [out.json] [chunk=10] [timeout=180]
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import traceback
from pathlib import Path

from carm.training import load_training_config
from scripts.evaluate_real_prompts import load_eval_prompts


def _worker(batch: list, artifact_dir_str: str, q: "mp.Queue") -> None:
    try:
        from scripts.evaluate_real_prompts import evaluate_isolated_prompts

        res = evaluate_isolated_prompts(batch, artifact_dir=Path(artifact_dir_str))
        q.put(("ok", {"rows": res["rows"], "provenance": res.get("provenance")}))
    except Exception as exc:  # noqa: BLE001
        q.put(("err", repr(exc) + "\n" + traceback.format_exc()))


def _eval_chunk_timed(batch: list, artifact_dir_str: str, timeout: int):
    ctx = mp.get_context("spawn")
    q: "mp.Queue" = ctx.Queue()
    p = ctx.Process(target=_worker, args=(batch, artifact_dir_str, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        try:
            p.kill()
        except Exception:
            pass
        p.join()
        return ("timeout", None)
    if not q.empty():
        return q.get()
    return ("empty", None)


def _aggregate(rows: list[dict], provenance: dict | None = None, errors: list[dict] | None = None) -> dict:
    total = max(1, len(rows))
    bm = sum(1 for r in rows if r.get("baseline_match"))
    pm = sum(1 for r in rows if r.get("pretrained_match"))
    return {
        "delta_tool_match_rate": round((pm - bm) / total, 4),
        "summary": {
            "prompt_count": len(rows),
            "baseline_match_rate": round(bm / total, 4),
            "pretrained_match_rate": round(pm / total, 4),
            "baseline_avg_steps": round(
                sum(len(r.get("baseline_actions", [])) for r in rows) / total, 4
            ),
            "pretrained_avg_steps": round(
                sum(len(r.get("pretrained_actions", [])) for r in rows) / total, 4
            ),
            "errored": sum(1 for r in rows if r.get("errored")),
        },
        # Real worker crash text. Empty => no chunk crashed. Non-empty => PARTIAL run.
        "errors": errors or [],
        "provenance": provenance,
        "rows": rows,
    }


def _placeholders(batch: list) -> list[dict]:
    out = []
    for item in batch:
        out.append({
            "id": str(item.get("id", "")),
            "logic_skill": str(item.get("logic_skill", "")),
            "expected_tool": str(item.get("expected_tool", "")),
            "baseline_used_tool": "",
            "pretrained_used_tool": "",
            "baseline_actions": [],
            "pretrained_actions": [],
            "baseline_match": False,
            "pretrained_match": False,
            "delta": 0,
            "errored": True,
        })
    return out


def main() -> int:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/eval/baseline_run_local.json")
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 180

    prompts = load_eval_prompts(Path("configs/real_prompt_eval.json"))
    training = load_training_config("configs/training.yaml")
    artifact_dir = Path(
        str(training.get("training", {}).get("pretraining", {}).get("artifact_dir", "data/pretrain"))
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: load already-completed rows.
    all_rows: list[dict] = []
    errors: list[dict] = []
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            all_rows = prev.get("rows", [])
            provenance = prev.get("provenance")
            errors = prev.get("errors", []) or []
            print(f"[baseline] resumed {len(all_rows)} existing rows, {len(errors)} prior errors from {out_path}",
                  flush=True)
        except Exception:
            all_rows = []
            provenance = None
            errors = []
    done_ids = {r.get("id") for r in all_rows}

    pending = [p for p in prompts if p.get("id") not in done_ids]
    print(f"[baseline] total={len(prompts)} already_done={len(done_ids)} pending={len(pending)} "
          f"chunk={chunk} timeout={timeout}s artifact={artifact_dir} "
          f"snapshot={Path('data/eval/baseline_snapshot').exists()}", flush=True)

    for i in range(0, len(pending), chunk):
        batch = pending[i : i + chunk]
        status, payload = _eval_chunk_timed(batch, str(artifact_dir), timeout)
        if status == "ok":
            all_rows.extend(payload["rows"])  # type: ignore[arg-type]
            provenance = payload.get("provenance") or provenance  # type: ignore[assignment]
            print(f"[baseline] chunk {i // chunk + 1} OK ({len(batch)}) cum={len(all_rows)}", flush=True)
        else:
            ids = [str(b.get("id", "?")) for b in batch]
            tb = payload if isinstance(payload, str) else f"status={status} (no traceback captured)"
            print(f"[baseline] chunk {i // chunk + 1} {status.upper()} ids={ids}\n{tb}", flush=True)
            errors.append({"chunk": i // chunk + 1, "status": status, "ids": ids, "traceback": tb})
            all_rows.extend(_placeholders(batch))
        out_path.write_text(json.dumps(_aggregate(all_rows, provenance, errors), ensure_ascii=False, indent=2), encoding="utf-8")

    final = _aggregate(all_rows, provenance, errors)
    print(f"[baseline] DONE delta={final['delta_tool_match_rate']} "
          f"baseline={final['summary']['baseline_match_rate']} "
          f"pretrained={final['summary']['pretrained_match_rate']} "
          f"errored={final['summary']['errored']} errors={len(final['errors'])} saved={out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
