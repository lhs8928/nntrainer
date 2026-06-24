#!/usr/bin/env bash
set -euo pipefail

# Deploy a pre-built CausalLM binary to an Android device over adb and run
# on-device inference. This is the Android counterpart of
# run_qwen3_0_6b_x86_inference.sh, scoped to deploy + run only.
#
# Building is intentionally out of scope. Produce the artifacts first with:
#   ./tools/package_android.sh            # cross-build libnntrainer (-> builddir/android_build_result)
#   pushd Applications/CausalLM/jni && ndk-build -j"$(nproc)" && popd
# ndk-build writes the nntrainer_causallm executable AND every runtime .so
# (libcausallm_core / libnntrainer / libccapi-nntrainer / libc++_shared) into
# Applications/CausalLM/libs/arm64-v8a/, so pushing that one directory resolves
# all runtime dependencies.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LIBS_DIR="${LIBS_DIR:-${REPO_ROOT}/Applications/CausalLM/libs/arm64-v8a}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/Applications/CausalLM/res/lfm2-tmp/q40/arm}"
DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/nntr_causallm}"
THREADS="${THREADS:-4}"
BINARY_NAME="nntrainer_causallm"
PROMPT="Give me a short introduction to large language model."
DO_PUSH=1

usage() {
  cat <<'USAGE'
Deploy a pre-built nntrainer CausalLM binary to an Android device and run inference.

Usage:
  run_lfm2_android_inference.sh [options] [prompt]

Options:
  --model-dir PATH    Model directory pushed to the device. Default: the ARM Q4_0
                      LFM2 model (Applications/CausalLM/res/lfm2-tmp/q40/arm).
                      Pass .../lfm2-tmp/fp32 to drive the FP32 model.
  --libs-dir PATH     ndk-build output dir (binary + .so). Default:
                      Applications/CausalLM/libs/arm64-v8a
  --device-dir PATH   On-device working dir. Default: /data/local/tmp/nntr_causallm
  --threads N         NNTR_NUM_THREADS value. Default: 4
  --no-push           Skip adb push; reuse what is already on the device.
  -h, --help          Show this help.

Environment overrides:
  MODEL_DIR, LIBS_DIR, DEVICE_DIR, THREADS
  ADB_SERIAL          adb device serial (-s). ADB_IP: remote adb host (-H).

Prerequisite (build first; not done by this script):
  ./tools/package_android.sh
  pushd Applications/CausalLM/jni && ndk-build -j"$(nproc)" && popd
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
  --model-dir) MODEL_DIR="$2"; shift 2 ;;
  --libs-dir) LIBS_DIR="$2"; shift 2 ;;
  --device-dir) DEVICE_DIR="$2"; shift 2 ;;
  --threads) THREADS="$2"; shift 2 ;;
  --no-push) DO_PUSH=0; shift ;;
  -h | --help) usage; exit 0 ;;
  --) shift; PROMPT="$*"; break ;;
  -*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  *) PROMPT="$*"; break ;;
  esac
done

# ---- adb command (honor ADB_IP / ADB_SERIAL like android_test.sh) ----
ADB=(adb)
[[ -n "${ADB_IP:-}" ]] && ADB+=(-H "${ADB_IP}")
[[ -n "${ADB_SERIAL:-}" ]] && ADB+=(-s "${ADB_SERIAL}")

if ! command -v adb >/dev/null 2>&1; then
  echo "adb not found in PATH." >&2
  exit 1
fi

device_count="$("${ADB[@]}" devices | awk 'NR>1 && $2=="device"' | wc -l)"
if [[ "${device_count}" -eq 0 ]]; then
  echo "No Android device detected by adb. Connect a device or set ADB_SERIAL/ADB_IP." >&2
  exit 1
fi
if [[ "${device_count}" -gt 1 && -z "${ADB_SERIAL:-}" ]]; then
  echo "Multiple devices attached; set ADB_SERIAL to pick one:" >&2
  "${ADB[@]}" devices >&2
  exit 1
fi

# ---- validate build artifacts ----
BINARY="${LIBS_DIR}/${BINARY_NAME}"
if [[ ! -x "${BINARY}" ]]; then
  echo "Inference binary not found: ${BINARY}" >&2
  echo "Build it first (see --help): tools/package_android.sh then ndk-build in Applications/CausalLM/jni." >&2
  exit 1
fi

# ---- validate model directory ----
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Model directory not found: ${MODEL_DIR}" >&2
  exit 1
fi
for required in config.json nntr_config.json tokenizer.json; do
  if [[ ! -f "${MODEL_DIR}/${required}" ]]; then
    echo "Missing required model file: ${MODEL_DIR}/${required}" >&2
    exit 1
  fi
done
# The weight blob name lives inside nntr_config.json; verify it exists locally
# so we fail here rather than mid-inference on the device.
MODEL_FILE="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['model_file_name'])" \
  "${MODEL_DIR}/nntr_config.json")"
if [[ ! -f "${MODEL_DIR}/${MODEL_FILE}" ]]; then
  echo "Missing weight file referenced by nntr_config.json: ${MODEL_DIR}/${MODEL_FILE}" >&2
  exit 1
fi

DEV_BIN="${DEVICE_DIR}/bin"
DEV_MODEL="${DEVICE_DIR}/model"

echo "Device       : $("${ADB[@]}" get-serialno 2>/dev/null || echo '?')"
echo "Libs dir     : ${LIBS_DIR}"
echo "Model dir    : ${MODEL_DIR} (weights: ${MODEL_FILE})"
echo "Device dir   : ${DEVICE_DIR}"
echo "Threads      : ${THREADS}"
echo "Prompt       : ${PROMPT}"
echo

if [[ "${DO_PUSH}" -eq 1 ]]; then
  echo ">> Pushing artifacts to device (this can take a while for large weights)..."
  "${ADB[@]}" shell "rm -rf ${DEV_MODEL} && mkdir -p ${DEV_BIN} ${DEV_MODEL}"
  "${ADB[@]}" push "${LIBS_DIR}/." "${DEV_BIN}"
  "${ADB[@]}" push "${MODEL_DIR}/." "${DEV_MODEL}"
  "${ADB[@]}" shell "chmod 755 ${DEV_BIN}/${BINARY_NAME}"
else
  echo ">> --no-push: reusing artifacts already on device."
fi

echo
echo ">> Running on-device inference..."
# Run from the bin dir so LD_LIBRARY_PATH=. picks up the co-located .so files.
"${ADB[@]}" shell "cd ${DEV_BIN} && LD_LIBRARY_PATH=. NNTR_NUM_THREADS=${THREADS} ./${BINARY_NAME} ${DEV_MODEL} '${PROMPT}'"
