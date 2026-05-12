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

download_from_hf() {
    local filename="$1"
    local dest="$2"
    if [ -f "$dest" ]; then
        echo "[*] ${dest} already exists."
        return 0
    fi
    echo "[+] Downloading ${filename} from huggingface.co/${AUTOAVSR_HF_REPO}..."
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download "$AUTOAVSR_HF_REPO" "$filename" \
            --local-dir "$TARGET_ASSETS" --quiet 2>&1 | tail -2
    elif command -v python &> /dev/null && python -c "import huggingface_hub" 2>/dev/null; then
        python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='$AUTOAVSR_HF_REPO', filename='$filename', local_dir='$TARGET_ASSETS')
" 2>&1 | tail -2
    else
        echo "ERROR: Need either huggingface-cli or python+huggingface_hub installed."
        echo "       pip install huggingface_hub  &&  huggingface-cli login"
        return 1
    fi
    if [ ! -f "$dest" ]; then
        echo "ERROR: ${dest} not present after download. Check your HF_TOKEN."
        return 1
    fi
}

download_from_hf "$AUTOAVSR_MODEL" "$AUTOAVSR_DEST_MODEL" || true
download_from_hf "$AUTOAVSR_VOCAB" "$AUTOAVSR_DEST_VOCAB" || true

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
    local repo="$1"
    local filename="$2"
    local dest="$3"
    if [ -f "$dest" ]; then
        echo "[*] ${dest} already exists."
        return 0
    fi
    echo "[+] Downloading ${filename} from huggingface.co/${repo}..."
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download "$repo" "$filename" \
            --local-dir "$TARGET_ASSETS" --quiet 2>&1 | tail -2
    elif command -v python &> /dev/null && python -c "import huggingface_hub" 2>/dev/null; then
        python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='$repo', filename='$filename', local_dir='$TARGET_ASSETS')
" 2>&1 | tail -2
    else
        echo "[!] Need huggingface-cli or python+huggingface_hub for V3 download. Skipping."
        return 1
    fi
    if [ ! -f "$dest" ]; then
        echo "[!] ${dest} not present after download."
        return 1
    fi
}

for f in "${AVHUBERT_FILES[@]}"; do
    download_from_hf_repo "$AVHUBERT_HF_REPO" "$f" "${TARGET_ASSETS}/${f}" || true
done

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
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download "$KENLM_HF_REPO" \
            --include "android-arm64/*" \
            --local-dir "app/src/main/cpp/kenlm" --quiet 2>&1 | tail -2
    elif command -v python &> /dev/null && python -c "import huggingface_hub" 2>/dev/null; then
        python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$KENLM_HF_REPO', repo_type='model',
                  allow_patterns=['android-arm64/*'],
                  local_dir='app/src/main/cpp/kenlm')
" 2>&1 | tail -2
    else
        echo "[!] Skipping KenLM NDK prebuilts: need huggingface-cli or python+huggingface_hub"
    fi
    if [ -f "${KENLM_NDK_DIR}/libkenlm.a" ]; then
        echo "[+] KenLM NDK prebuilts installed ($(du -sh "$KENLM_NDK_DIR" | cut -f1))"
    fi
else
    echo "[*] KenLM NDK prebuilts already present at ${KENLM_NDK_DIR}"
fi

# --- TTS / Voice Cloning ONNX Models ---
# Pre-converted ONNX models for PocketTTSEngine (voice cloning pipeline).
# If not available from GitHub Releases, generate them locally with:
#   pip install speechbrain TTS torch onnx onnxruntime numpy
#   python tools/convert_tts_models.py

TTS_RELEASE_TAG="v0.1.0-models"
TTS_MODELS=("pocket_tts_speaker.onnx" "pocket_tts_acoustic.onnx" "pocket_tts_vocoder.onnx")

for model in "${TTS_MODELS[@]}"; do
    if [ ! -f "${TARGET_ASSETS}/${model}" ]; then
        TTS_URL="https://github.com/HereLiesAz/Liperty/releases/download/${TTS_RELEASE_TAG}/${model}"
        echo "[+] Downloading ${model} from GitHub Releases..."
        if command -v curl &> /dev/null; then
            curl -L -o "${TARGET_ASSETS}/${model}" "$TTS_URL"
        elif command -v wget &> /dev/null; then
            wget -O "${TARGET_ASSETS}/${model}" "$TTS_URL"
        fi
        if [ -f "${TARGET_ASSETS}/${model}" ] && file "${TARGET_ASSETS}/${model}" | grep -q "HTML"; then
            echo "[!] Download failed for ${model} (got HTML). Generate locally with: python tools/convert_tts_models.py"
            rm -f "${TARGET_ASSETS}/${model}"
        else
            echo "[+] ${model} installed."
        fi
    else
        echo "[*] ${model} already exists."
    fi
done

# Optional: FreeVC voice conversion model (requires manual download)
if [ ! -f "${TARGET_ASSETS}/pocket_tts_vc.onnx" ]; then
    echo "[*] pocket_tts_vc.onnx not found. Voice conversion is optional."
    echo "    To enable: download FreeVC from https://github.com/OlaWod/FreeVC"
    echo "    then run: python tools/convert_tts_models.py --freevc"
fi

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
