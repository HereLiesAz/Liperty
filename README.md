# Liperty

**Liperty** is a real-time, on-device Visual Speech Recognition (VSR) and Silent Speech Interface (SSI) application designed for the Deaf, Hard-of-Hearing, and Speech-Impaired communities. It utilizes advanced deep learning models to convert lip movements into text or synthesized speech, operating entirely offline to ensure privacy and low latency.

## Features

*   **Real-time Lipreading:** Transcribes interlocutor speech via the camera feed.
*   **Silent Speech Interface (SSI):** Uses an "Artificial Larynx" mode (phone vibration) as a carrier signal that the user modulates silently.
*   **Multimodal Laryngeal Sensing:** Synchronizes contact-mic and accelerometer data for robust voice activity detection.
*   **Voice Cloning:** Create and manage personalized voice profiles using local ONNX-based PocketTTS.
*   **Privacy-First:** All processing (Face Mesh, VSR, DSP, TTS) happens on-device.
*   **Innovative UI/UX:**
    *   **Interactive Overlay:** Transcription rendered as tappable word blocks directly over video.
    *   **Pinch-to-Zoom Scaling:** Dynamic font sizing via multi-touch gestures.
    *   **Gesture Control:** Air-swipes and waves for hands-free correction and pausing.
*   **Telephoto Optimization:** Automatically selects telephoto lenses to reduce perspective distortion.

## Architecture

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for a deep dive into the technical stack, including:
*   **Planned:** Native C++ AAudio/Oboe and Bluetooth LE Audio LC3 codec support for robust Alaryngeal speech translation.
*   CameraX & MediaPipe for the vision pipeline.
*   JNI/OpenCV for hardware-accelerated image normalization.
*   LiteRT (TFLite) for inference and on-device personalization (LoRA).
*   Multimodal DSP pipeline for laryngeal sensing.

## Setup & Installation

1.  Clone the repository.
2.  Run the initialization script:
    ```bash
    ./setup_libs.sh
    ```
    This script downloads the OpenCV Android SDK, project models, and patches dependencies for AGP 9.1+ compatibility.
3.  Open in Android Studio (Ladybug or newer recommended).
4.  Sync Gradle project.
5.  Connect a physical Android device (Emulator support is limited for NPU/CameraX/Vibration).
6.  Build and Run.

## Legal & Privacy

This application processes biometric data (facial landmarks). See [LEGAL.md](docs/LEGAL.md) for compliance information regarding Wiretap laws and Biometric Privacy acts (BIPA/GDPR).

## Research

Based on the expert frameworks detailed in `docs/RESEARCH.md`.
