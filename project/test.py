"""
SENTIX Test Runner
==================
Runs the full automated test suite for SENTIX.
"""
import unittest
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(PROJECT_DIR / "tests"), top_level_dir=str(PROJECT_DIR))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
