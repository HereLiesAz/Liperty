# System Architecture

## Overview

The Liperty application architecture is designed for real-time, on-device execution of Visual Speech Recognition (VSR) and Voice Reconstruction (BC/EL) pipelines. It prioritizes low latency (sub-100ms) and absolute privacy (RAM-only ephemeral processing).

## Core Pipelines

### 1. Vision & VSR Pipeline
*   **Input (CameraX):** Captures high-quality frames, defaulting to the Front camera. Rear camera utilizes Telephoto lenses where available to minimize perspective distortion.
*   **Preprocessing (JNI/OpenCV):**
    *   **Stabilization:** `RectKalmanFilter` smooths bounding box coordinates to counter hand jitter.
    *   **Normalization:** Gaussian Blur and Histogram Equalization implemented in C++ via pixel locking for high-frequency efficiency. **Note:** Histogram equalization is currently bypassed — the native implementation was collapsing per-frame contrast, making all frames near-identical. Raw crops are passed to inference instead.
*   **Inference (LiteRT):** Executes `.tflite` models (VALLR/DeepLip) with GPU acceleration.
*   **Decoding:** Custom CTC Beam Search with prefix merging for accurate linguistic reconstruction.

### 2. Voice Reconstruction (BC/EL) Pipeline
*   **BC Mode (Bone Conduction Larynx):** `GlottalCarrierGenerator` produces a glottal pulse carrier (80–200 Hz) routed to BC headphones via `ArtificialLarynx` + `AudioRouter`. The phone's built-in mic captures the acoustically modulated result.
*   **EL Mode (Electrolarynx Translator):** Captures external electrolarynx buzz via the phone mic. No carrier generation needed.
*   **Shared DSP Stage (`VibraPhoneDSP`):**
    *   **Spectral Subtraction:** Removes noise using a calibrated noise profile.
    *   **Equalization:** Emphasizes speech-modulation bands (300Hz–3.5kHz).
    *   **Mel Spectrogram → VoiceConverter → Inverse Mel:** Maps processed audio to the user's cloned voice.
*   **Audio Routing (`AudioRouter`):** Manages full-duplex configuration (simultaneous mic input + headphone output). Forces phone built-in mic in BC mode to avoid capturing the carrier from the headphone mic.

### 3. Voice Management
*   **PocketTTS Engine:** Executes ONNX-based voice cloning models locally.
*   **VoiceStore:** Securely persists voice embeddings and identities.

### 4. Planned Alaryngeal and SSR Audio Translation (RESEARCH3)
*   **Native Low-Latency Audio:** Native C++ Oboe / AAudio wrapper for ultra-low latency playback.
*   **Expansion:** TRAMBA high-frequency bandwidth expansion model for reconstructing components from BCMs.
*   **Voice Cloning:** Real-time Voice Cloning architectures as detailed in RESEARCH3.md.
*   **Noise Suppression:** Pitch-synchronous generalized spectral subtraction via NDK.

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
