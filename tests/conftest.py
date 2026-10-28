import os

# Prevent duplicate libomp initialization abort on macOS when loading Torch + OpenCV/scan_core
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
