#!/bin/bash

# Build script for CausalLM Test Application
# This script builds test_api executable only
set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN} $1 ${NC}"
    echo -e "${CYAN}========================================${NC}"
}

log_step() {
    echo -e "\n${YELLOW}[Step $1]${NC} $2"
    echo -e "${YELLOW}----------------------------------------${NC}"
}

# Function to check and fix artifact location
check_artifact() {
    local filename=$1
    local libs_path="libs/arm64-v8a/$filename"
    local obj_path="obj/local/arm64-v8a/$filename"

    if [ -f "$libs_path" ]; then
        size=$(ls -lh "$libs_path" | awk '{print $5}')
        echo -e "  ${GREEN}[OK]${NC} $filename ($size)"
        return 0
    elif [ -f "$obj_path" ]; then
        echo -e "  ${YELLOW}[WARN]${NC} $filename found in obj but not in libs. Copying..."
        mkdir -p "libs/arm64-v8a"
        cp "$obj_path" "$libs_path"
        if [ -x "$obj_path" ]; then
            chmod +x "$libs_path"
        fi
        size=$(ls -lh "$libs_path" | awk '{print $5}')
        echo -e "  ${GREEN}[OK]${NC} $filename ($size) (Copied from obj)"
        return 0
    else
        echo -e "  ${RED}[ERROR]${NC} $filename not found!"
        echo "  Checked paths:"
        echo "    - $libs_path"
        echo "    - $obj_path"
        return 1
    fi
}

# Check if NDK path is set
if [ -z "$ANDROID_NDK" ]; then
    log_error "ANDROID_NDK is not set. Please set it to your Android NDK path."
    echo "Example: export ANDROID_NDK=/path/to/android-ndk-r21d"
    exit 1
fi

# Set NNTRAINER_ROOT
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNTRAINER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export NNTRAINER_ROOT

log_header "Build CausalLM Test Application"
echo "NNTRAINER_ROOT: $NNTRAINER_ROOT"
echo "ANDROID_NDK: $ANDROID_NDK"
echo "Working directory: $(pwd)"

# Check required libraries
log_step "1/2" "Check Dependencies"

MISSING_DEPS=false

# Check Core Lib
if [ -f "$SCRIPT_DIR/jni/libs/arm64-v8a/libquick_ai_core.so" ]; then
    echo -e "  ${GREEN}[OK]${NC} libquick_ai_core.so found"
else
    echo -e "  ${RED}[MISSING]${NC} libquick_ai_core.so"
    echo "    -> Run ./build_android.sh first"
    MISSING_DEPS=true
fi

# Check API Lib
if [ -f "$SCRIPT_DIR/jni/libs/arm64-v8a/libquick_ai_api.so" ]; then
    echo -e "  ${GREEN}[OK]${NC} libquick_ai_api.so found"
else
    echo -e "  ${RED}[MISSING]${NC} libquick_ai_api.so"
    echo "    -> Run ./build_api_lib.sh first"
    MISSING_DEPS=true
fi

if [ "$MISSING_DEPS" = true ]; then
    echo ""
    log_error "Missing dependencies. Please build required libraries first."
    exit 1
fi

log_success "All dependencies found"

# Step 2: Build Test App
log_step "2/2" "Build Test App"

cd "$SCRIPT_DIR/jni"

log_info "Building with ndk-build (builds test_api)..."
if ndk-build NDK_PROJECT_PATH=. NDK_LIBS_OUT=./libs NDK_OUT=./obj APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk test_api -j $(nproc); then
    log_success "Build completed successfully"
else
    log_error "Build failed"
    exit 1
fi

# Verify output
echo ""
echo "Build artifacts:"

check_artifact "test_api" || exit 1

# Summary
log_header "Build Summary"
log_success "Build completed successfully!"
echo ""
echo "Output files are in: $SCRIPT_DIR/jni/libs/arm64-v8a/"
echo ""
echo "Executables:"
echo "  - test_api (API test application)"
echo ""
echo "To install and run:"
echo "  ./install_android.sh"
echo ""
