#!/usr/bin/env python3
"""
Root Benchmark Launcher for SENTIX
Dispatches execution to project/benchmark.py.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ROOT_DIR / "project"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from project.benchmark import run_all_benchmarks

if __name__ == "__main__":
    run_all_benchmarks()

