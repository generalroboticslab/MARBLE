#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${BALLBOT_PYTHON:-/home/orangepi/repo/micromamba/envs/py312/bin/python}"
VCPKG_ROOT="${VCPKG_ROOT:-/home/orangepi/repo/vcpkg}"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python 3.12 environment not found: $PYTHON" >&2
    echo "Set BALLBOT_PYTHON to a Python 3.12 interpreter." >&2
    exit 1
fi
if [[ ! -f "$ROOT/hardware_bindings/imu/imu_nanobind.cpp" ]]; then
    echo "hardware_bindings/imu/imu_nanobind.cpp is missing; this checkout is incomplete." >&2
    exit 1
fi
if [[ ! -f "$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" ]]; then
    echo "vcpkg toolchain not found: $VCPKG_ROOT" >&2
    echo "Set VCPKG_ROOT to a bootstrapped vcpkg checkout." >&2
    exit 1
fi

"$PYTHON" -m pip install -r "$ROOT/config/requirements-robot.txt"
cmake -S "$ROOT" -B "$ROOT/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
    -DPython_EXECUTABLE="$PYTHON"
cmake --build "$ROOT/build" --target imu_nanobind
"$PYTHON" "$ROOT/scripts/preflight_heuristic.py"
