# System Architecture

## Overview

The Liperty application architecture is designed for real-time, on-device execution of Visual Speech Recognition (VSR) models. It prioritizes low latency (sub-100ms) and privacy (zero-cloud dependencies).

## High-Level Pipeline

1.  **Camera Input (CameraX):**
    *   **Goal:** Capture high-quality video frames.
    *   **Configuration:**
        *   **Resolution:** 1080p (downscaled for ROI).
        *   **FPS:** 25-30 FPS fixed.
        *   **Lens Selection:** Prefer **Telephoto** (2x/3x) for rear camera to minimize "moustache distortion" (geometric warping of facial features). Prefer Front-facing for SSI.
    *   **Stabilization:** Hardware OIS + Software smoothing (Kalman Filter) on bounding box coordinates.

2.  **Frame Preprocessing (MediaPipe + OpenCV/Bitmap):**
    *   **Face Detection:** MediaPipe Face Mesh (BlazeFace).
    *   **Landmark Extraction:** 468 3D landmarks.
    *   **Lip ROI Extraction:**
        *   Identify upper/lower lip vermilion borders.
        *   Apply affine transformation to neutralize head roll/pitch/yaw.
        *   Crop to consistent size (e.g., 96x96, 112x112, 128x128).
    *   **Normalization:** Convert to Grayscale -> Gaussian Blur -> Contrast Stretching (Histogram Equalization).

3.  **Visual Speech Recognition (VSR) Model (LiteRT / TensorFlow Lite):**
    *   **Input:** Sequence of grayscale lip ROIs (Tensor: `[Batch, Time, Height, Width, Channel]`).
    *   **Model Options:**
        *   **DeepLip (CNN-LSTM-CTC):** Lightweight, suitable for older devices.
        *   **VALLR (Transformer + LLM):** State-of-the-art. Encoder predicts phonemes; Decoder reconstructs words.
    *   **Execution:** GPU Delegate for CNN/Transformer layers.

4.  **Language Model Decoding (On-Device LLM):**
    *   **Input:** Raw phoneme/viseme sequence or CTC probabilities.
    *   **Processing:**
        *   **Beam Search:** Find most likely word sequences.
        *   **Homophene Correction:** Use language context to disambiguate (e.g., "pat" vs "bat" vs "mat").
    *   **Execution:** NPU Delegate (if available) for transformer operations.

5.  **User Interface (Jetpack Compose / View System):**
    *   **OverlayView:** Draws bounding box around mouth to confirm tracking.
    *   **TranscriptionView:** Displays real-time text.
    *   **Gesture Correction:**
        *   Swipe Left/Right on a word to cycle through homophene candidates.
        *   Swipe Up to Speak (TTS).

## Data Flow

```mermaid
graph TD
    A[CameraX Stream] -->|ImageProxy| B[FrameAnalyzer]
    B -->|Bitmap| C[FaceMesh Detector]
    C -->|Landmarks| D[ROI Extractor]
    D -->|Cropped Lip Frames| E[Input Buffer]
    E -->|Tensor Batch| F[TFLite Interpreter (VSR)]
    F -->|Phonemes/Visemes| G[Language Decoder (LLM)]
    G -->|Text| H[UI State]
    H -->|Render| I[Screen]
```

## Hardware Acceleration Strategy

*   **Vision Pipeline (MediaPipe):** GPU.
*   **VSR Encoder (CNN/Transformer):** GPU or DSP.
*   **LLM Decoder:** NPU (Neural Processing Unit) via NNAPI or TFLite Delegates.

## Key Technologies

*   **Language:** Kotlin
*   **Camera:** CameraX
*   **ML:** TensorFlow Lite (LiteRT), MediaPipe
*   **Concurrency:** Kotlin Coroutines, Flow
*   **UI:** Jetpack Compose (or XML Layouts for overlay performance)
