# System Architecture

## Overview

The Liperty application architecture is designed for real-time, on-device execution of Visual Speech Recognition (VSR) and Silent Speech Interface (SSI) models. It prioritizes low latency (sub-100ms) and absolute privacy (RAM-only ephemeral processing).

## Core Pipelines

### 1. Vision & VSR Pipeline
*   **Input (CameraX):** Captures high-quality frames, defaulting to the Front camera. Rear camera utilizes Telephoto lenses where available to minimize perspective distortion.
*   **Preprocessing (JNI/OpenCV):**
    *   **Stabilization:** `RectKalmanFilter` smooths bounding box coordinates to counter hand jitter.
    *   **Normalization:** Gaussian Blur and Histogram Equalization implemented in C++ via pixel locking for high-frequency efficiency.
*   **Inference (LiteRT):** Executes `.tflite` models (VALLR/DeepLip) with GPU acceleration.
*   **Decoding:** Custom CTC Beam Search with prefix merging for accurate linguistic reconstruction.

### 2. Laryngeal Sensing (SSI) Pipeline
*   **Carrier (Artificial Larynx):** High-intensity multi-motor vibration via `VibratorManager` acts as a carrier sound source.
*   **Sensing (Multimodal):**
    *   **Contact-mic:** Captures throat-conducted audio via the `VOICE_RECOGNITION` source.
    *   **Accelerometer:** Tracks 0-400Hz laryngeal vibrations for high-precision VAD gating.
*   **DSP Stage:**
    *   **Spectral Subtraction:** Removes haptic noise using a calibrated noise profile.
    *   **Equalization:** Emphasizes speech-modulation bands (300Hz–3.5kHz).
    *   **Expansion:** Extrapolates harmonics beyond the 2kHz laryngeal "deaf zone."

### 3. Voice Management
*   **PocketTTS Engine:** Executes ONNX-based voice cloning models locally.
*   **VoiceStore:** Securely persists voice embeddings and identities.

### 4. Planned Alaryngeal and SSR Audio Translation (RESEARCH3)
*   **Native Low-Latency Audio:** Planned Native C++ Oboe / AAudio wrapper for ultra-low latency playback.
*   **Expansion:** Planned TRAMBA high-frequency bandwidth expansion model for reconstructing components from BCMs.
*   **Voice Cloning:** Planned real-time Voice Cloning architectures as detailed in RESEARCH3.md.
*   **Noise Suppression:** Planned Pitch-synchronous generalized spectral subtraction via NDK.

## Hardware Acceleration Strategy

*   **Vision Pipeline:** GPU via MediaPipe and OpenCV NDK.
*   **ML Inference:** GPU/NPU delegates for TFLite.
*   **DSP Pipeline:** SIMD-optimized FFT operations where applicable.

## UI/UX Framework

*   **Jetpack Compose:** Powers the main interactive overlay and navigation.
*   **AzNavRail:** A specialized navigation system for assistive accessibility.
*   **Multi-touch Gestures:** Transformable modifiers for pinch-to-zoom text scaling.
*   **MediaPipe Hand Landmarker:** Enables hands-free "Air Gesture" controls.

## Technical Stack

*   **Language:** Kotlin (Native) / C++ (NDK).
*   **ML:** LiteRT (TensorFlow Lite), MediaPipe, ONNX Runtime.
*   **Sensors:** Camera2/CameraX, Android Sensor Framework, Vibrator API.
*   **Concurrency:** Kotlin Coroutines & StateFlow.
