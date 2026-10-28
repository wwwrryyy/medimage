# Embedded ARM Cross-Compilation Guide

This guide details how to cross-compile the C++ `scan_core` zero-copy diagnostic engine (`AnomalyCalculus`, `VideoIngestor`, and pybind11 bridge) for embedded 64-bit ARM Linux targets (e.g., NVIDIA Jetson Orin Nano, Raspberry Pi 4/5 64-bit, or embedded point-of-care medical cart hardware).

---

## Prerequisites (Build Host)

On an x86_64 Ubuntu/Debian host, install the GNU ARM toolchain and multiarch dependencies:

```bash
sudo apt-get update
sudo apt-get install -y \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    cmake \
    python3-dev \
    libopencv-dev
```

For target sysroots or Docker cross-compilation containers:
```bash
# E.g. using multiarch sysroot for ARM64:
sudo dpkg --add-architecture arm64
sudo apt-get update
sudo apt-get install -y \
    libopencv-dev:arm64 \
    python3-dev:arm64
```

---

## Cross-Compiling with CMake

From the `src_cpp/` directory, invoke CMake referencing the toolchain file:

```bash
cd src_cpp
mkdir -p build-arm64 && cd build-arm64

cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchain-aarch64-linux-gnu.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DARM_SYSROOT=/usr/aarch64-linux-gnu

make -j$(nproc)
```

The resulting compiled shared library (`scan_core.so` / `scan_core.cpython-*.so`) is written directly to `src_python/` for deployment.

---

## Deploying to Edge Device

1. Copy the repository or runtime bundle to the ARM edge target:
   ```bash
   rsync -avz --exclude '.git' ./ patient-monitor@jetson.local:~/medical-imaging/
   ```
2. Verify execution on the embedded ARM target:
   ```bash
   python3 -c "import scan_core; engine = scan_core.ScanEngine(); print('Engine initialized:', engine.get_health_status())"
   ```
3. Run the automated integration tests on-device:
   ```bash
   pytest tests/test_scan_core.py tests/test_scan_session.py
   ```
