import os
import sys

# Prevent duplicate libomp initialization abort on macOS when loading Torch + OpenCV/scan_core
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add repository root to sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

