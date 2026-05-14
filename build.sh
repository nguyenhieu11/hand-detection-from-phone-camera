#!/usr/bin/env bash
# build.sh – one-shot setup + build for Ubuntu 24.04
set -e

# ── 1. System dependencies ────────────────────────────────────────────────────
echo "[1/4] Checking system packages..."

PKGS_NEEDED=()
for pkg in build-essential cmake git libopencv-dev; do
    if dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
        echo "  [ok] $pkg already installed"
    else
        echo "  [--] $pkg missing, will install"
        PKGS_NEEDED+=("$pkg")
    fi
done

if [ ${#PKGS_NEEDED[@]} -gt 0 ]; then
    sudo apt-get update -qq
    sudo apt-get install -y "${PKGS_NEEDED[@]}"
else
    echo "  All system packages already present, skipping apt."
fi

# Check OpenCV version (need >= 4.0)
if pkg-config --exists opencv4 2>/dev/null; then
    OCV_VER=$(pkg-config --modversion opencv4)
    echo "  [ok] OpenCV ${OCV_VER} detected"
elif pkg-config --exists opencv 2>/dev/null; then
    OCV_VER=$(pkg-config --modversion opencv)
    echo "  [ok] OpenCV ${OCV_VER} detected"
fi

# ── 2. ONNX Runtime (CPU, pre-built) ─────────────────────────────────────────
ORT_VERSION="1.20.1"
ORT_ARCH="x64"
ORT_TARBALL="onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}.tgz"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_TARBALL}"

# Helper: parse semver into integer for comparison (1.20.1 → 10020001)
version_int() { echo "$1" | awk -F. '{ printf "%d%04d%04d\n", $1, $2, $3 }'; }

ORT_HEADER=""
for candidate in \
    /usr/local/include/onnxruntime/onnxruntime_cxx_api.h \
    /usr/include/onnxruntime/onnxruntime_cxx_api.h \
    /opt/onnxruntime/include/onnxruntime_cxx_api.h; do
    [ -f "$candidate" ] && ORT_HEADER="$candidate" && break
done

ORT_INSTALLED_VER=""
if [ -n "$ORT_HEADER" ]; then
    # Try to read version from the version header next to it
    VERSION_FILE="$(dirname "$ORT_HEADER")/onnxruntime/core/session/onnxruntime_c_api.h"
    [ ! -f "$VERSION_FILE" ] && VERSION_FILE="$(dirname "$ORT_HEADER")/onnxruntime_c_api.h"
    if [ -f "$VERSION_FILE" ]; then
        MAJOR=$(grep -m1 "ORT_API_VERSION" "$VERSION_FILE" | grep -oP '\d+' | head -1)
    fi
    # Fallback: read from shared lib
    LIB_VER=$(ldconfig -p 2>/dev/null | grep "libonnxruntime.so\." | grep -oP '\d+\.\d+\.\d+' | head -1)
    ORT_INSTALLED_VER="${LIB_VER}"
fi

NEED_ORT=true
if [ -n "$ORT_HEADER" ] && [ -n "$ORT_INSTALLED_VER" ]; then
    if [ "$(version_int "$ORT_INSTALLED_VER")" -ge "$(version_int "$ORT_VERSION")" ]; then
        echo "[2/4] ONNX Runtime ${ORT_INSTALLED_VER} already installed (>= required ${ORT_VERSION}), skipping."
        NEED_ORT=false
    else
        echo "[2/4] ONNX Runtime ${ORT_INSTALLED_VER} found but older than required ${ORT_VERSION}, upgrading..."
    fi
elif [ -n "$ORT_HEADER" ]; then
    echo "[2/4] ONNX Runtime header found at ${ORT_HEADER} (version unknown), assuming compatible, skipping."
    NEED_ORT=false
fi

if $NEED_ORT; then
    echo "[2/4] Downloading ONNX Runtime ${ORT_VERSION}..."
    wget -q --show-progress "${ORT_URL}" -O /tmp/${ORT_TARBALL}
    tar -xf /tmp/${ORT_TARBALL} -C /tmp/
    ORT_DIR="/tmp/onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}"
    sudo cp -r ${ORT_DIR}/include/* /usr/local/include/
    sudo cp -r ${ORT_DIR}/lib/*     /usr/local/lib/
    sudo ldconfig
    echo "[2/4] ONNX Runtime ${ORT_VERSION} installed."
fi

# ── 3. Build ──────────────────────────────────────────────────────────────────
echo "[3/4] Building..."

# Auto-detect ORT install prefix from where the header was found
if [ -n "$ORT_HEADER" ] && $NEED_ORT == false; then
    ORT_PREFIX=$(echo "$ORT_HEADER" | sed 's|/include/.*||')
else
    ORT_PREFIX="/usr/local"
fi

mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT="${ORT_PREFIX}"
cmake --build build -- -j"$(nproc)"

# ── 4. Run ────────────────────────────────────────────────────────────────────
echo "[4/4] Done! Run with:"
echo "      ./build/yolov8_hand best.onnx"
