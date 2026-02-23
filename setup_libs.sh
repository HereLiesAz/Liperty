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
    unzip -q "${TARGET_LIBS}/${OPENCV_ZIP}" -d "$TARGET_LIBS"

    # 4. Rename/Structure
    if [ -d "${TARGET_LIBS}/OpenCV-android-sdk" ]; then
        echo "[+] Configuring path..."
        mv "${TARGET_LIBS}/OpenCV-android-sdk" "$TARGET_OPENCV"

        if [ -d "${TARGET_OPENCV}/samples" ]; then
            echo "[+] Removing OpenCV samples..."
            rm -rf "${TARGET_OPENCV}/samples"
        fi
    else
        echo "[!] Error: Extraction failed or folder structure unexpected."
        exit 1
    fi

    # 5. Cleanup
    rm "${TARGET_LIBS}/${OPENCV_ZIP}"
    echo "[+] OpenCV installed."
fi

# --- Model Setup ---

echo "[+] Setting up Models..."

# VSR Model (Dummy Generation if missing)
if [ ! -f "${TARGET_ASSETS}/vsr_model.tflite" ]; then
    echo "[+] Generating dummy VSR model (vsr_model.tflite)..."
    python3 tools/create_dummy_model.py
else
    echo "[*] VSR model already exists."
fi

# Face Landmarker (Download from MediaPipe if missing)
FACE_TASK_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
if [ ! -f "${TARGET_ASSETS}/face_landmarker.task" ]; then
    echo "[+] Downloading face_landmarker.task..."
    if command -v wget &> /dev/null; then
        wget -O "${TARGET_ASSETS}/face_landmarker.task" "$FACE_TASK_URL"
    elif command -v curl &> /dev/null; then
        curl -L -o "${TARGET_ASSETS}/face_landmarker.task" "$FACE_TASK_URL"
    fi
else
    echo "[*] Face Landmarker task already exists."
fi

echo "========================================"
echo "SUCCESS"
echo "OpenCV installed at: ${TARGET_OPENCV}"
echo "Models installed at: ${TARGET_ASSETS}"
echo "========================================"
