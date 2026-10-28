# ==============================================================================
# CMake Toolchain File: Cross-compilation for ARM64 (aarch64-linux-gnu)
# Target: NVIDIA Jetson (Nano/Orin), Raspberry Pi (4/5 64-bit), NXP i.MX8, Embedded ARM
# ==============================================================================

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Cross-compilers
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# ARM architecture tuning and SIMD NEON optimizations
set(CMAKE_CXX_FLAGS_INIT "-march=armv8-a+simd -O3 -fPIC -pthread")
set(CMAKE_C_FLAGS_INIT "-march=armv8-a+simd -O3 -fPIC -pthread")

# If an ARM target sysroot is specified via -DARM_SYSROOT=/path/to/sysroot
if(DEFINED ARM_SYSROOT)
    set(CMAKE_SYSROOT ${ARM_SYSROOT})
    set(CMAKE_FIND_ROOT_PATH ${ARM_SYSROOT})
endif()

# Search rules for cross-compiling
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
