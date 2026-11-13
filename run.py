"""
SENTIX Root Application Launcher
================================
Convenience launcher that delegates to the project/ folder.
"""

import sys
import os
from pathlib import Path

# Add project directory to sys.path
PROJECT_DIR = Path(__file__).resolve().parent / "project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Switch working directory to project/ so relative paths resolve cleanly
os.chdir(PROJECT_DIR)

from backend.app import app

if __name__ == "__main__":
    print("=" * 60)
    print("  SENTIX: Multimodal Emotion-Aware AI Companion")
    print("  Launching from root wrapper via project/ package")
    print("  Server running at: http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)
