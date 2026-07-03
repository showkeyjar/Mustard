"""Runner script that sets CARM_NO_EMBEDDING before importing evaluate_carm_benchmark."""

import os

os.environ["CARM_NO_EMBEDDING"] = "1"

import sys

sys.argv = [
    "evaluate_carm_benchmark",
    "--output",
    "data/eval/benchmark_report_v0.5.2.json",
]

from scripts.evaluate_carm_benchmark import main

main()
