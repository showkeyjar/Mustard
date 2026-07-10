"""Run BFCL evaluation for CARM router.

This script:
1. Starts the CARM BFCL API server (OpenAI-compatible)
2. Runs BFCL generate + evaluate for specified test categories
3. Collects and prints results

Usage:
    # In BFCL conda env:
    set OPENAI_API_KEY=dummy
    set OPENAI_BASE_URL=http://localhost:11400/v1
    python scripts/run_bfcl_eval.py --categories simple_python
"""

import subprocess
import sys
import os
import time
import json
import argparse
import signal
from pathlib import Path


def start_carm_server(port=11400):
    """Start CARM BFCL server as subprocess."""
    # Use the system Python (not conda) since it has httpx
    server_proc = subprocess.Popen(
        [sys.executable, "scripts/carm_bfcl_server.py", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
    )
    # Wait for server to be ready
    import httpx

    for _ in range(10):
        try:
            r = httpx.get(f"http://localhost:{port}/health", timeout=2)
            if r.status_code == 200:
                print(f"CARM server ready on port {port}")
                return server_proc
        except Exception:
            time.sleep(1)
    print("ERROR: CARM server failed to start")
    # Print server logs
    server_proc.terminate()
    return None


def run_bfcl_generate(model, categories, result_dir=None):
    """Run BFCL generate command."""
    cmd = [
        "bfcl",
        "generate",
        "--model",
        model,
        "--test-category",
        categories,
    ]
    if result_dir:
        cmd.extend(["--result-dir", result_dir])

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "dummy"
    env["OPENAI_BASE_URL"] = "http://localhost:11400/v1"
    env["PYTHONIOENCODING"] = "utf-8"

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=3600,
    )
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-2000:]}")
    return result.returncode


def run_bfcl_evaluate(model, categories, result_dir=None):
    """Run BFCL evaluate command."""
    cmd = [
        "bfcl",
        "evaluate",
        "--model",
        model,
        "--test-category",
        categories,
    ]
    if result_dir:
        cmd.extend(["--result-dir", result_dir])

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=3600,
    )
    print(result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-2000:]}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run BFCL eval for CARM")
    parser.add_argument(
        "--categories",
        default="simple_python",
        help="BFCL test categories (comma-separated or 'single_turn')",
    )
    parser.add_argument(
        "--model",
        default="carm-router",
        help="Model name registered in BFCL",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=11400,
        help="Port for CARM API server",
    )
    parser.add_argument(
        "--result-dir",
        default=None,
        help="Result directory (default: BFCL default)",
    )
    parser.add_argument(
        "--skip-server",
        action="store_true",
        help="Skip starting CARM server (assume already running)",
    )
    args = parser.parse_args()

    # Start server
    server_proc = None
    if not args.skip_server:
        server_proc = start_carm_server(args.port)
        if server_proc is None:
            sys.exit(1)

    try:
        # Generate
        rc = run_bfcl_generate(args.model, args.categories, args.result_dir)
        if rc != 0:
            print(f"Generate failed with code {rc}")
            sys.exit(1)

        # Evaluate
        rc = run_bfcl_evaluate(args.model, args.categories, args.result_dir)
        if rc != 0:
            print(f"Evaluate failed with code {rc}")
            sys.exit(1)

        print("\n=== BFCL Evaluation Complete ===")

    finally:
        if server_proc:
            server_proc.terminate()
            server_proc.wait()


if __name__ == "__main__":
    main()
