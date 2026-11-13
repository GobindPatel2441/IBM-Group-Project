"""
SENTIX Application Launcher
===========================
Starts the Sentix Flask backend server on http://localhost:5000.
Serves both the multimodal API and the Liquid Glass frontend interface.
"""

import sys
import os
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

os.chdir(PROJECT_DIR)

from backend.app import app

if __name__ == "__main__":
    print("=" * 60)
    print("  SENTIX: Multimodal Emotion-Aware AI Companion")
    print("  Server running at: http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)
