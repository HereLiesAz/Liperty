#!/bin/bash
# setup_libs.sh
# Downloads and configures external dependencies (OpenCV) and Models for Liperty.
# TARGET_LIBS: app/src/main/cpp/libs
# TARGET_ASSETS: app/src/main/assets

# Configuration
OPENCV_VERSION="4.10.0"
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
            echo "[+] Patching OpenCV build.gradle..."

            # 5. Patch for AGP 9.0 Compatibility (ProGuard)
            # Replace deprecated 'proguard-android.txt' with 'proguard-android-optimize.txt'
            sed 's/proguard-android.txt/proguard-android-optimize.txt/g' "$OPENCV_BUILD_GRADLE" > "${OPENCV_BUILD_GRADLE}.tmp" && mv "${OPENCV_BUILD_GRADLE}.tmp" "$OPENCV_BUILD_GRADLE"

            # 6. Patch for JVM Target Compatibility (Java 17)
            echo "[+] Upgrading OpenCV source compatibility to Java 17..."
            sed 's/JavaVersion.VERSION_1_8/JavaVersion.VERSION_17/g' "$OPENCV_BUILD_GRADLE" > "${OPENCV_BUILD_GRADLE}.tmp" && mv "${OPENCV_BUILD_GRADLE}.tmp" "$OPENCV_BUILD_GRADLE"

            # 7. Patch for AGP 9 built-in Kotlin (remove kotlin-android plugin)
            echo "[+] Removing kotlin-android plugin for AGP 9 compatibility..."
            sed "/apply plugin: 'kotlin-android'/d" "$OPENCV_BUILD_GRADLE" > "${OPENCV_BUILD_GRADLE}.tmp" && mv "${OPENCV_BUILD_GRADLE}.tmp" "$OPENCV_BUILD_GRADLE"

            # 8. Replace deprecated compileSdkVersion with compileSdk
            sed 's/compileSdkVersion /compileSdk /g' "$OPENCV_BUILD_GRADLE" > "${OPENCV_BUILD_GRADLE}.tmp" && mv "${OPENCV_BUILD_GRADLE}.tmp" "$OPENCV_BUILD_GRADLE"

            # 9. Remove kotlinOptions block (no longer needed without kotlin-android plugin)
            sed '/kotlinOptions {/,/}/d' "$OPENCV_BUILD_GRADLE" > "${OPENCV_BUILD_GRADLE}.tmp" && mv "${OPENCV_BUILD_GRADLE}.tmp" "$OPENCV_BUILD_GRADLE"

            # 10. Remove externalNativeBuild blocks (avoids ninja/Google Drive conflicts;
            #     prebuilt .so files are already in jniLibs, app CMake links directly)
            echo "[+] Removing externalNativeBuild from OpenCV module..."
            sed '/externalNativeBuild {/,/}/d' "$OPENCV_BUILD_GRADLE" > "${OPENCV_BUILD_GRADLE}.tmp" && mv "${OPENCV_BUILD_GRADLE}.tmp" "$OPENCV_BUILD_GRADLE"

            # 11. Remove prefab blocks (app links OpenCV via its own CMakeLists.txt)
            sed '/prefabPublishing/d' "$OPENCV_BUILD_GRADLE" > "${OPENCV_BUILD_GRADLE}.tmp" && mv "${OPENCV_BUILD_GRADLE}.tmp" "$OPENCV_BUILD_GRADLE"
            sed '/prefab {/,/}/d' "$OPENCV_BUILD_GRADLE" > "${OPENCV_BUILD_GRADLE}.tmp" && mv "${OPENCV_BUILD_GRADLE}.tmp" "$OPENCV_BUILD_GRADLE"

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

# 1. Vallr Bundle
if [ ! -f "VALLR/VALLR.path" ]; then
    mkdir -p "VALLR"
    download_from_gdrive "$VALLR_GD_ID" "Vallr.zip"
    if [ -f "Vallr.zip" ]; then
        if file "Vallr.zip" | grep -q "HTML"; then
            echo "ERROR: Downloaded file appears to be an HTML page, not a zip. Google Drive download likely failed."
            rm -f "Vallr.zip"
            exit 1
        fi
        echo "[+] Extracting Vallr.zip..."
        unzip -o "Vallr.zip" -d "VALLR/"
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to unzip. Download may have failed."
            exit 1
        fi
        rm "Vallr.zip"
    fi
else
    echo "[*] VALLR data already exists."
fi

# 2. Tools Bundle
if [ ! -d "tools/external" ]; then
    mkdir -p "tools"
    download_from_gdrive "$TOOLS_GD_ID" "tools.zip"
    if [ -f "tools.zip" ]; then
        if file "tools.zip" | grep -q "HTML"; then
            echo "ERROR: Downloaded file appears to be an HTML page, not a zip. Google Drive download likely failed."
            rm -f "tools.zip"
            exit 1
        fi
        echo "[+] Extracting tools.zip..."
        unzip -o "tools.zip" -d "tools/"
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to unzip. Download may have failed."
            exit 1
        fi
        rm "tools.zip"
    fi
else
    echo "[*] Tools already exist."
fi

# 3. Assets Bundle (Models — face/hand landmarkers, ssr/tramba/voice_converter stubs,
#    homophones.json, etc). The VSR model itself is downloaded separately in step 4.
ASSETS_BUNDLE_MARKER="${TARGET_ASSETS}/face_landmarker.task"
if [ ! -f "$ASSETS_BUNDLE_MARKER" ]; then
    download_from_gdrive "$ASSETS_GD_ID" "assets.zip"
    if [ -f "assets.zip" ]; then
        if file "assets.zip" | grep -q "HTML"; then
            echo "ERROR: Downloaded file appears to be an HTML page, not a zip. Google Drive download likely failed."
            rm -f "assets.zip"
            exit 1
        fi
        echo "[+] Extracting assets.zip..."
        unzip -o "assets.zip" -d "$TARGET_ASSETS"
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to unzip. Download may have failed."
            exit 1
        fi
        rm "assets.zip"
    fi
else
    echo "[*] Assets (Models) already exist."
fi

# 4. VSR ONNX Model (from GitHub Releases — too large for git).
# Downloaded directly into app/src/main/assets/ so it's bundled in the APK and
# loaded by OnnxModelEngine at runtime. Skipped if the asset already exists.
ONNX_MODEL="vallr_model.onnx"
ONNX_URL="https://github.com/HereLiesAz/Liperty/releases/download/v0.1.0-models/${ONNX_MODEL}"
ONNX_DEST="${TARGET_ASSETS}/${ONNX_MODEL}"
if [ ! -f "$ONNX_DEST" ]; then
    echo "[+] Downloading ${ONNX_MODEL} from GitHub Releases into ${ONNX_DEST}..."
    if command -v curl &> /dev/null; then
        curl -L -o "$ONNX_DEST" "$ONNX_URL"
    elif command -v wget &> /dev/null; then
        wget -O "$ONNX_DEST" "$ONNX_URL"
    fi
    if [ -f "$ONNX_DEST" ] && file "$ONNX_DEST" | grep -q "HTML"; then
        echo "ERROR: Downloaded ONNX file appears to be HTML. Download likely failed."
        rm -f "$ONNX_DEST"
    fi
else
    echo "[*] ${ONNX_DEST} already exists."
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
