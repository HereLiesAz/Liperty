#!/bin/bash
# setup_libs.sh
# Downloads and configures external dependencies (OpenCV) for GraffitiXR.
# TARGET: core/nativebridge/libs

# Configuration
OPENCV_VERSION="4.13.0"
OPENCV_ZIP="opencv-${OPENCV_VERSION}-android-sdk.zip"
OPENCV_URL="https://github.com/opencv/opencv/releases/download/${OPENCV_VERSION}/${OPENCV_ZIP}"
TARGET_BASE="core/nativebridge/libs"
TARGET_OPENCV="${TARGET_BASE}/opencv"

echo "========================================"
echo "GraffitiXR Library Setup"
echo "Target: ${TARGET_BASE}"
echo "========================================"

# 1. Create Target Directory
if [ ! -d "$TARGET_BASE" ]; then
    echo "[+] Creating directory: $TARGET_BASE"
    mkdir -p "$TARGET_BASE"
fi

# 2. Download OpenCV
if [ -d "$TARGET_OPENCV" ]; then
    echo "[!] OpenCV directory already exists at $TARGET_OPENCV"
    read -p "Re-download and overwrite? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping OpenCV setup."
        exit 0
    fi
    rm -rf "$TARGET_OPENCV"
fi

echo "[+] Downloading OpenCV Android SDK ${OPENCV_VERSION}..."
# Use curl or wget depending on availability
if command -v wget &> /dev/null; then
    wget -O "${TARGET_BASE}/${OPENCV_ZIP}" "$OPENCV_URL"
elif command -v curl &> /dev/null; then
    curl -L -o "${TARGET_BASE}/${OPENCV_ZIP}" "$OPENCV_URL"
else
    echo "Error: Neither wget nor curl found."
    exit 1
fi

# 3. Extract
echo "[+] Extracting ${OPENCV_ZIP}..."
unzip -q "${TARGET_BASE}/${OPENCV_ZIP}" -d "$TARGET_BASE"

# 4. Rename/Structure
# The zip extracts to 'OpenCV-android-sdk'. We rename it to 'opencv'.
if [ -d "${TARGET_BASE}/OpenCV-android-sdk" ]; then
    echo "[+] Configuring path..."
    mv "${TARGET_BASE}/OpenCV-android-sdk" "$TARGET_OPENCV"
else
    echo "[!] Error: Extraction failed or folder structure unexpected."
    exit 1
fi

# 5. Cleanup
echo "[+] Cleaning up..."
rm "${TARGET_BASE}/${OPENCV_ZIP}"

echo "========================================"
echo "SUCCESS"
echo "OpenCV installed at: ${TARGET_OPENCV}"
echo "Ensure settings.gradle.kts points to: project(\":opencv\").projectDir = file(\"core/nativebridge/libs/opencv/sdk\")"
echo "========================================"
