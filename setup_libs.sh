#!/bin/bash
# setup_libs.sh
# Downloads and configures external dependencies (OpenCV) and Models for Liperty.
# TARGET_LIBS: app/src/main/cpp/libs
# TARGET_ASSETS: app/src/main/assets

# Configuration
# OpenCV 4.13.0 ships native libs aligned to 16 KiB pages (.so segments
# at 0x4000 instead of 0x1000), so libopencv_java4.so is 16 KB-compatible
# on Android 15. The build.gradle patches below still apply -- the SDK
# layout hasn't changed.
OPENCV_VERSION="4.13.0"
OPENCV_ZIP="opencv-${OPENCV_VERSION}-android-sdk.zip"
OPENCV_URL="https://github.com/opencv/opencv/releases/download/${OPENCV_VERSION}/${OPENCV_ZIP}"
TARGET_LIBS="app/src/main/cpp/libs"
TARGET_OPENCV="${TARGET_LIBS}/opencv"
TARGET_ASSETS="app/src/main/assets"

echo "========================================"
echo "Liperty Library & Model Setup"
echo "Target Libs: ${TARGET_LIBS}"
echo "Target Assets: ${TARGET_ASSETS}"
echo "========================================"

# 1. Create Target Directories
mkdir -p "$TARGET_LIBS"
mkdir -p "$TARGET_ASSETS"

# --- OpenCV Setup ---

# 2. Download OpenCV
if [ -d "$TARGET_OPENCV" ]; then
    echo "[!] OpenCV directory already exists at $TARGET_OPENCV"
else
    echo "[+] Downloading OpenCV Android SDK ${OPENCV_VERSION}..."
    if command -v wget &> /dev/null; then
        wget -O "${TARGET_LIBS}/${OPENCV_ZIP}" "$OPENCV_URL"
    elif command -v curl &> /dev/null; then
        curl -L -o "${TARGET_LIBS}/${OPENCV_ZIP}" "$OPENCV_URL"
    else
        echo "Error: Neither wget nor curl found."
        exit 1
    fi

    # 3. Extract
    echo "[+] Extracting ${OPENCV_ZIP}..."
    OUTPUT_FILE="${TARGET_LIBS}/${OPENCV_ZIP}"
    if file "$OUTPUT_FILE" | grep -q "HTML"; then
        echo "ERROR: Downloaded file appears to be an HTML page, not a zip. Google Drive download likely failed."
        rm -f "$OUTPUT_FILE"
        exit 1
    fi
    unzip -q "$OUTPUT_FILE" -d "$TARGET_LIBS"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to unzip. Download may have failed."
        exit 1
    fi

    # 4. Rename/Structure
    if [ -d "${TARGET_LIBS}/OpenCV-android-sdk" ]; then
        echo "[+] Configuring path..."
        mv "${TARGET_LIBS}/OpenCV-android-sdk" "$TARGET_OPENCV"

        if [ -d "${TARGET_OPENCV}/samples" ]; then
            echo "[+] Removing OpenCV samples..."
            rm -rf "${TARGET_OPENCV}/samples"
        fi

        OPENCV_BUILD_GRADLE="${TARGET_OPENCV}/sdk/build.gradle"
        if [ -f "$OPENCV_BUILD_GRADLE" ]; then
            # Replace OpenCV's shipped build.gradle outright with a minimal
            # AGP 9 / Java 17 -compatible one. The brittle sed patches we used
            # for 4.10.0 broke against 4.13.0's reorganized layout (orphan
            # braces left after externalNativeBuild/prefab block deletions).
            # Since we only need to package the prebuilt .so files from
            # native/libs and the Java sources from java/src as an Android
            # library module, the minimal config below is sufficient and
            # version-stable.
            echo "[+] Writing replacement build.gradle for OpenCV module..."
            cat > "$OPENCV_BUILD_GRADLE" <<'GRADLE_EOF'
// Minimal replacement for OpenCV's shipped build.gradle.
// Written by setup_libs.sh.
// We only need OpenCV as an Android library that exposes:
//   - prebuilt .so files from native/libs (consumed by libliperty_cv.so
//     via find_package(OpenCV) in app/src/main/cpp/CMakeLists.txt)
//   - Java classes from java/src (org.opencv.* API surface)
// We don't need OpenCV's externalNativeBuild, prefab, or maven-publish
// blocks -- the app links directly against the prebuilts.
apply plugin: 'com.android.library'

android {
    namespace 'org.opencv'
    compileSdk 34

    defaultConfig {
        minSdkVersion 21
    }

    compileOptions {
        sourceCompatibility JavaVersion.VERSION_17
        targetCompatibility JavaVersion.VERSION_17
    }

    buildFeatures {
        buildConfig true
    }

    buildTypes {
        debug {
            packagingOptions {
                doNotStrip '**/*.so'
            }
        }
        release {
            packagingOptions {
                doNotStrip '**/*.so'
            }
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.txt'
        }
    }

    sourceSets {
        main {
            jniLibs.srcDirs = ['native/libs']
            java.srcDirs = ['java/src']
            res.srcDirs = ['java/res']
            manifest.srcFile 'java/AndroidManifest.xml'
        }
    }
}
GRADLE_EOF
        else
            echo "[!] Warning: OpenCV build.gradle not found at $OPENCV_BUILD_GRADLE"
        fi

    else
        echo "[!] Error: Extraction failed or folder structure unexpected."
        exit 1
    fi

    # 7. Cleanup
    rm "${TARGET_LIBS}/${OPENCV_ZIP}"
    echo "[+] OpenCV installed."
fi

# --- Model & Data Setup ---

echo "[+] Setting up Models, Tools, and VALLR Data..."

# Google Drive IDs provided by user
VALLR_GD_ID="1TdthJ9ibfruV5BQ_LWP2go2RWJoXlTrh"
TOOLS_GD_ID="1YjGOWbhIqBN626vCqZAuSok5uzCiFHyt"
ASSETS_GD_ID="11ajiCy4skJo5B8K2OrFAipmBGc2Eus_m"

download_from_gdrive() {
    local file_id=$1
    local output_file=$2
    echo "[+] Downloading ID: ${file_id} to ${output_file}..."
    if command -v curl &> /dev/null; then
        CONFIRM=$(curl -sc /tmp/gdrive_cookie.txt "https://drive.google.com/uc?export=download&id=${file_id}" | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')
        curl -Lb /tmp/gdrive_cookie.txt "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${file_id}" -o "$output_file"
    elif command -v wget &> /dev/null; then
        wget --load-cookies /tmp/gdrive_cookie.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/gdrive_cookie.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${file_id} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')" -O "$output_file" && rm -rf /tmp/gdrive_cookie.txt
    fi
}

# Each Google Drive bundle below is a *convenience* artifact for local dev,
# not a requirement for the Android build. Drive returns an HTML interstitial
# for files >~100MB or when sharing permissions changed, and its file IDs
# rot. Don't fail CI on a Drive miss — the things the APK actually needs
# (Auto-AVSR ONNX, mediapipe .task files, TTS models) are pulled from
# HuggingFace or GitHub Releases or a public mediapipe URL further down.

try_extract_or_skip() {
    # try_extract_or_skip <local_zip> <dest_dir> <human_name>
    local zip_path="$1"
    local dest_dir="$2"
    local label="$3"
    if [ ! -f "$zip_path" ]; then
        echo "[!] ${label}: download produced no file. Skipping."
        return 0
    fi
    if file "$zip_path" | grep -q "HTML"; then
        echo "[!] ${label}: Drive returned an HTML page (interstitial / permissions / wrong ID). Skipping."
        rm -f "$zip_path"
        return 0
    fi
    echo "[+] Extracting $(basename "$zip_path")..."
    if ! unzip -q -o "$zip_path" -d "$dest_dir"; then
        echo "[!] ${label}: unzip failed. Skipping."
        rm -f "$zip_path"
        return 0
    fi
    rm -f "$zip_path"
    return 0
}

# 1. Vallr Bundle (research scaffolding only — VALLR/ Python code is not
#    compiled into the APK; superseded by the Auto-AVSR backend at runtime.)
if [ ! -f "VALLR/VALLR.path" ]; then
    mkdir -p "VALLR"
    download_from_gdrive "$VALLR_GD_ID" "Vallr.zip"
    try_extract_or_skip "Vallr.zip" "VALLR/" "VALLR bundle"
else
    echo "[*] VALLR data already exists."
fi

# 2. Tools Bundle (offline notebook helpers — not consumed by the Android build.)
if [ ! -d "tools/external" ]; then
    mkdir -p "tools"
    download_from_gdrive "$TOOLS_GD_ID" "tools.zip"
    try_extract_or_skip "tools.zip" "tools/" "tools bundle"
else
    echo "[*] Tools already exist."
fi

# 3. Assets Bundle (face/hand landmarkers, ssr/tramba/voice_converter stubs,
#    homophones.json, etc). The face/hand .task files have a public-URL
#    fallback at the bottom of this script. The VSR model itself comes
#    from HuggingFace in step 4. So a Drive miss here is non-fatal.
ASSETS_BUNDLE_MARKER="${TARGET_ASSETS}/face_landmarker.task"
if [ ! -f "$ASSETS_BUNDLE_MARKER" ]; then
    download_from_gdrive "$ASSETS_GD_ID" "assets.zip"
    try_extract_or_skip "assets.zip" "$TARGET_ASSETS" "assets bundle"
else
    echo "[*] Assets (Models) already exist."
fi

# 4. VSR ONNX Model: Auto-AVSR (Chaplin's LRS3_V_WER19.1 visual encoder + CTC head)
# Pulled from a private HuggingFace model repo (HereLiesAz/liperty-autoavsr-onnx)
# produced by tools/export_autoavsr_to_onnx.ipynb. Requires HF_TOKEN env var or a
# `~/.cache/huggingface/token` file (run `huggingface-cli login` once).
AUTOAVSR_MODEL="autoavsr_lrs3_visual_ctc.onnx"
AUTOAVSR_VOCAB="unigram5000_units.txt"
AUTOAVSR_HF_REPO="HereLiesAz/liperty-autoavsr-onnx"
AUTOAVSR_DEST_MODEL="${TARGET_ASSETS}/${AUTOAVSR_MODEL}"
AUTOAVSR_DEST_VOCAB="${TARGET_ASSETS}/${AUTOAVSR_VOCAB}"

# --- Hugging Face download hardening --------------------------------------
# Anonymous HF requests are aggressively rate-limited, which is the usual
# cause of flaky / missing model pulls in CI (e.g. the KenLM arm64 prebuilts
# verification gate failing intermittently). Authenticate when a token is
# available — huggingface-cli and huggingface_hub both auto-read HF_TOKEN
# from the environment; we normalize the two common env var names.
HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
export HF_TOKEN
[ -n "$HF_TOKEN" ] && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
if [ -n "$HF_TOKEN" ]; then
    echo "[*] HF_TOKEN detected — Hugging Face downloads will be authenticated."
else
    echo "[*] No HF_TOKEN set — downloading anonymously (subject to rate limits)."
fi

# Run a command with up to 4 attempts and exponential backoff (2s, 4s, 8s).
# Returns the command's exit code from the last attempt. Note: callers must
# NOT pipe the wrapped command (e.g. through 'tail'), or the pipe's exit
# status would mask real download failures and defeat the retry.
hf_retry() {
    local attempt=1 max=4 delay=2
    while true; do
        "$@" && return 0
        if [ "$attempt" -ge "$max" ]; then
            echo "[!] still failing after ${max} attempts: $1"
            return 1
        fi
        echo "[~] attempt ${attempt}/${max} failed; retrying in ${delay}s..."
        sleep "$delay"
        attempt=$((attempt + 1))
        delay=$((delay * 2))
    done
}

# One download attempt of a single file into <local_dir>. Real exit code is
# returned (no output pipe) so hf_retry can see failures. Token is read from
# the environment by both the CLI and the Python client.
_hf_fetch_attempt() {
    local repo="$1" filename="$2" local_dir="$3"
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download "$repo" "$filename" --local-dir "$local_dir" --quiet
    elif command -v python &> /dev/null && python -c "import huggingface_hub" 2>/dev/null; then
        python - "$repo" "$filename" "$local_dir" <<'PY'
import os, sys
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id=sys.argv[1], filename=sys.argv[2], local_dir=sys.argv[3],
                token=os.environ.get("HF_TOKEN") or None)
PY
    else
        echo "ERROR: need huggingface-cli or python+huggingface_hub installed." >&2
        echo "       pip install huggingface_hub  (and set HF_TOKEN for private repos)" >&2
        return 1
    fi
}

# One download attempt of every file matching <pattern> into <local_dir>
# (multi-file / subdirectory pulls, e.g. android-arm64/*). Real exit code.
_hf_snapshot_attempt() {
    local repo="$1" pattern="$2" local_dir="$3"
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download "$repo" --include "$pattern" --local-dir "$local_dir" --quiet
    elif command -v python &> /dev/null && python -c "import huggingface_hub" 2>/dev/null; then
        python - "$repo" "$pattern" "$local_dir" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], repo_type='model',
                  allow_patterns=[sys.argv[2]], local_dir=sys.argv[3],
                  token=os.environ.get("HF_TOKEN") or None)
PY
    else
        echo "ERROR: need huggingface-cli or python+huggingface_hub installed." >&2
        return 1
    fi
}

# Fetch a single file into <local_dir> with auth + retries. Skips if already
# present; returns non-zero if the file is still missing afterward.
hf_fetch_file() {
    local repo="$1" filename="$2" local_dir="$3"
    local dest="${local_dir}/${filename}"
    if [ -f "$dest" ]; then
        echo "[*] ${dest} already exists."
        return 0
    fi
    echo "[+] Downloading ${filename} from huggingface.co/${repo}..."
    hf_retry _hf_fetch_attempt "$repo" "$filename" "$local_dir" || true
    if [ ! -f "$dest" ]; then
        echo "[!] ${dest} not present after download (check connectivity / HF_TOKEN)."
        return 1
    fi
}

download_from_hf() {
    local filename="$1"
    local dest="$2"
    hf_fetch_file "$AUTOAVSR_HF_REPO" "$filename" "$TARGET_ASSETS"
}

download_from_hf "$AUTOAVSR_MODEL" "$AUTOAVSR_DEST_MODEL" || true
download_from_hf "$AUTOAVSR_VOCAB" "$AUTOAVSR_DEST_VOCAB" || true

# SyncVSR backend (KAIST-AILab Vox+LRS2+LRS3, exported by
# tools/export_syncvsr_to_onnx.ipynb + tools/syncvsr_export_stage2.py).
# MainActivity selects between Auto-AVSR and SyncVSR via VSR_BACKEND;
# we pull both so the swap is a code-only flip without re-running setup.
SYNCVSR_HF_REPO="HereLiesAz/liperty-syncvsr-onnx"
download_syncvsr() {
    local filename="$1"
    hf_fetch_file "$SYNCVSR_HF_REPO" "$filename" "$TARGET_ASSETS"
}
download_syncvsr "syncvsr_lrs3_visual_ctc.onnx" || true
download_syncvsr "syncvsr_lrs3_visual_ctc_fp16.onnx" || true
download_syncvsr "syncvsr_unigram_units.txt" || true
# SyncVSR seq2seq: encoder-only hidden states (no CTC head) + attention
# decoder. Exported by tools/syncvsr_export_stage4_encoder.py and
# tools/syncvsr_export_stage3_decoder.py respectively. Wired together
# in AvHubertSeq2SeqInference.createSyncVsr() and routed by
# MainActivity.SYNCVSR_USE_SEQ2SEQ. ~759 MB + ~273 MB.
download_syncvsr "syncvsr_lrs3_encoder.onnx" || true
download_syncvsr "syncvsr_lrs3_decoder.onnx" || true

# Personalized SyncVSR encoder (per-user LoRA-merged). Optional — only
# present after the user runs the calibration flow on-device, exports
# via TrainingDataExporter, and trains offline through
# tools/train_syncvsr_lora.ipynb (which uploads the result to
# <user>/liperty-syncvsr-personal-lora). When present at runtime AND
# MainActivity.SYNCVSR_USE_PERSONAL_LORA is true, replaces the
# generic encoder. Edit SYNCVSR_PERSONAL_HF_REPO to point at your own
# trained-LoRA repo before re-running setup_libs.sh.
SYNCVSR_PERSONAL_HF_REPO="${SYNCVSR_PERSONAL_HF_REPO:-HereLiesAz/liperty-syncvsr-personal-lora}"
download_personal_syncvsr() {
    local filename="$1"
    hf_fetch_file "$SYNCVSR_PERSONAL_HF_REPO" "$filename" "$TARGET_ASSETS" || true
}
download_personal_syncvsr "syncvsr_lrs3_encoder_personal.onnx" || true

# --- AV-HuBERT V3 Backend (encoder + Transformer-decoder seq2seq) ---
# Optional. If missing the app still works in V2 mode (Auto-AVSR CTC).
# See docs/AVHUBERT_V3_BACKEND.md. Repo is PUBLIC so HF_TOKEN is not
# required, but huggingface_hub still needs to be installed.
AVHUBERT_HF_REPO="HereLiesAz/liperty-avhubert-encoder"
AVHUBERT_FILES=(
    "avhubert_base_vox_433h_visual_encoder.onnx"
    "avhubert_base_vox_433h_decoder.onnx"
    "avhubert_base_vox_433h_dict.txt"
)

download_from_hf_repo() {
    # download_from_hf_repo <repo_id> <filename> <dest>
    # <dest> must equal <local_dir>/<filename>; we derive local_dir from it so
    # existing call sites keep working unchanged.
    local repo="$1"
    local filename="$2"
    local dest="$3"
    hf_fetch_file "$repo" "$filename" "$(dirname "$dest")"
}

for f in "${AVHUBERT_FILES[@]}"; do
    download_from_hf_repo "$AVHUBERT_HF_REPO" "$f" "${TARGET_ASSETS}/${f}" || true
done

# --- On-Device Training Artifacts (personalization Step 3) ---
# ORT Training artifacts for per-user LoRA fine-tuning of the AV-HuBERT
# visual encoder. Built by tools/kaggle_build_training_artifacts.py and
# hosted at HereLiesAz/liperty-v3-training-artifacts (~820 MB total).
# Optional: missing artifacts = training is unavailable; inference works fine.
TRAINING_HF_REPO="HereLiesAz/liperty-v3-training-artifacts"
TRAINING_ASSETS="${TARGET_ASSETS}/training"
mkdir -p "$TRAINING_ASSETS"

TRAINING_FILES=(
    "training_model.onnx"
    "eval_model.onnx"
    "optimizer_model.onnx"
)

for f in "${TRAINING_FILES[@]}"; do
    download_from_hf_repo "$TRAINING_HF_REPO" "$f" "${TRAINING_ASSETS}/${f}" || true
done

# Checkpoint is a directory — use snapshot_download with pattern matching
if [ ! -d "${TRAINING_ASSETS}/checkpoint" ] || [ -z "$(ls -A "${TRAINING_ASSETS}/checkpoint" 2>/dev/null)" ]; then
    echo "[+] Pulling training checkpoint from ${TRAINING_HF_REPO}..."
    hf_retry _hf_snapshot_attempt "$TRAINING_HF_REPO" "checkpoint/*" "$TRAINING_ASSETS" || true
    if [ -d "${TRAINING_ASSETS}/checkpoint" ]; then
        echo "[+] Training checkpoint installed."
    else
        echo "[!] Training checkpoint download failed. On-device training will be unavailable."
    fi
else
    echo "[*] Training checkpoint already present."
fi

# --- KenLM Language Model (shallow-fusion / n-best rescoring) ---
# LibriSpeech 3-gram pruned 1e-7, KenLM trie+q8 binary (~27 MB).
# Vocabulary is UPPERCASE — the Kotlin scorer uppercases words before
# querying. Optional; missing LM = pure-CTC decoder with no rescoring.
KENLM_HF_REPO="HereLiesAz/liperty-lm"
KENLM_FILE="librispeech_3gram.bin"
download_from_hf_repo "$KENLM_HF_REPO" "$KENLM_FILE" "${TARGET_ASSETS}/${KENLM_FILE}" || true

# --- KenLM Android NDK prebuilts (arm64-v8a) ---
# Cross-compiled by tools/kaggle_build_kenlm_android.py and pulled
# into app/src/main/cpp/kenlm/android-arm64/ so the NDK build can link
# against libkenlm.a + libkenlm_util.a. Without these, kenlm_jni.cpp
# compiles as a stub that returns 0 (KenLmScorer.isNativeLoaded stays
# false and the rescoring stack is a no-op).
KENLM_NDK_DIR="app/src/main/cpp/kenlm/android-arm64"
mkdir -p "$KENLM_NDK_DIR"
if [ ! -f "${KENLM_NDK_DIR}/libkenlm.a" ]; then
    echo "[+] Pulling KenLM Android prebuilts from ${KENLM_HF_REPO}..."
    hf_retry _hf_snapshot_attempt "$KENLM_HF_REPO" "android-arm64/*" "app/src/main/cpp/kenlm" || true
    if [ -f "${KENLM_NDK_DIR}/libkenlm.a" ]; then
        echo "[+] KenLM NDK prebuilts installed ($(du -sh "$KENLM_NDK_DIR" | cut -f1))"
    else
        echo "[!] KenLM NDK prebuilts NOT installed after retries — the build's"
        echo "    'Verify KenLM prebuilts' gate will fail. Check that"
        echo "    ${KENLM_HF_REPO} has android-arm64/libkenlm.a + libkenlm_util.a,"
        echo "    and that HF_TOKEN is set if the repo is private/rate-limited."
    fi
else
    echo "[*] KenLM NDK prebuilts already present at ${KENLM_NDK_DIR}"
fi

# --- TTS / Voice Cloning ONNX Models ---
# Pre-converted ONNX models for PocketTTSEngine (voice cloning pipeline),
# pulled from HereLiesAz/liperty-pocket-tts (built by
# tools/export_tts_to_onnx.ipynb). The previous version of this block
# pointed at a GitHub Release tag (v0.1.0-models) that doesn't actually
# host the .onnx files, so every fresh setup_libs.sh run got 9-byte
# HTML stubs that ORT couldn't load.
TTS_HF_REPO="HereLiesAz/liperty-pocket-tts"
# OpenVoice v2 two-stage pipeline:
#   base.onnx          — MeloTTS English base TTS (~70 MB)
#   se_extractor.onnx  — Speaker Embedding extractor, 256-d (~20 MB)
#   tone_converter.onnx — timbre transfer, ref voice → cloned (~50 MB)
#   vocab.json         — MeloTTS symbol table + ARPABET mapping (~10 KB)
#   cmudict_compact.txt — CMU Pronouncing Dictionary for g2p (~3 MB)
#   g2p_neural.onnx    — Neural G2P for OOV words (~500 KB, optional)
TTS_MODELS=(
    "pocket_tts_base.onnx"
    "pocket_tts_se_extractor.onnx"
    "pocket_tts_tone_converter.onnx"
    "pocket_tts_vocab.json"
    "cmudict_compact.txt"
    "g2p_neural.onnx"
)

download_tts() {
    local filename="$1"
    local dest="${TARGET_ASSETS}/${filename}"
    if [ -f "$dest" ] && [ "$(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest" 2>/dev/null)" -gt 1000 ]; then
        echo "[*] ${filename} already exists."
        return 0
    fi
    rm -f "$dest"   # Drop any old tiny/HTML stub so it gets re-pulled.
    if hf_fetch_file "$TTS_HF_REPO" "$filename" "$TARGET_ASSETS"; then
        echo "[+] ${filename} installed ($(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest" 2>/dev/null) bytes)."
    else
        echo "[!] ${filename} download failed. Generate via tools/export_tts_to_onnx.ipynb on Kaggle."
    fi
}

for model in "${TTS_MODELS[@]}"; do
    download_tts "$model"
done

# OpenVoice v2 replaced the optional FreeVC path with the bundled
# Tone Color Converter (pocket_tts_tone_converter.onnx). No extra
# voice-conversion download required.

# Fallback/Utility: Face and Hand Landmarkers (if still missing after assets.zip)
FACE_TASK_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
HAND_TASK_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if [ ! -f "${TARGET_ASSETS}/face_landmarker.task" ]; then
    echo "[+] Downloading face_landmarker.task (Fallback)..."
    if command -v wget &> /dev/null; then
        wget -O "${TARGET_ASSETS}/face_landmarker.task" "$FACE_TASK_URL"
    elif command -v curl &> /dev/null; then
        curl -L -o "${TARGET_ASSETS}/face_landmarker.task" "$FACE_TASK_URL"
    fi
fi

if [ ! -f "${TARGET_ASSETS}/hand_landmarker.task" ]; then
    echo "[+] Downloading hand_landmarker.task (Fallback)..."
    if command -v wget &> /dev/null; then
        wget -O "${TARGET_ASSETS}/hand_landmarker.task" "$HAND_TASK_URL"
    elif command -v curl &> /dev/null; then
        curl -L -o "${TARGET_ASSETS}/hand_landmarker.task" "$HAND_TASK_URL"
    fi
fi

echo "========================================"
echo "SUCCESS"
echo "OpenCV installed at: ${TARGET_OPENCV}"
echo "Models installed at: ${TARGET_ASSETS}"
echo "========================================"
