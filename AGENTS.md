# AGENTS.md

This file provides guidance to AI Agents (e.g., WARP, Claude) when working with code in this repository.

## Project Overview

**Liperty** is a real-time, on-device Visual Speech Recognition (VSR) and Silent Speech Interface (SSI) application. It converts lip movements or silently articulated vibrations into text or synthesized speech.

### Key Characteristics
- **Platform**: Android (Native Kotlin/C++)
- **ML Stack**: LiteRT (TFLite), MediaPipe, ONNX Runtime (PocketTTS)
- **Vision**: JNI/OpenCV integration for frame normalization.
- **SSI**: Multimodal fusion of Contact-mic and Accelerometer data.
- **Privacy**: No cloud dependencies; ephemeral RAM processing only.

## Development Environment & Setup

### 1. Dependency Initialization
You **must** run the setup script before attempting to build:
```bash
./setup_libs.sh
```
This script:
- Configures the OpenCV Android SDK (v4.10.0+).
- Patches AGP compatibility for Java 17.
- Downloads models (VALLR, Face Task, Dummy TFLite).

### 2. Build Commands
- **Build APK**: `./gradlew assembleDebug`
- **Tests**: `./gradlew testDebugUnitTest`

### 3. Hardware Requirements
- **Physical Devices**: Required for testing ML inference, VibratorManager, and high-frequency sensors.
- **Permissions**: App enforces strict Legal Consent for biometric data.

## Coding Guidelines

### Machine Learning & Performance
- **Zero-Allocation**: Reuse buffers (e.g., `BitmapPool`) during camera callbacks.
- **Native Logic**: Use JNI for operations requiring >30Hz frame rates (Normalization, DSP).
- **Delegates**: Use GPU/NPU delegates but always implement CPU fallback.

### UI & Interaction
- **Compose**: Primary UI framework.
- **Accessibility**: Optimized for one-handed use and hands-free gesture control.
- **Transcription**: Rendered as word blocks over the camera; supports pinch-to-zoom scaling.

### Implementation Status (Crucial)
Consult `docs/TODO.md` before starting tasks. The project currently uses **Mock/Dummy** components for:
- VSR Model weights (character indexing aligned, but random weights).
- SSI Formant Extrapolation (simple spectral folding).
- Voice Cloning inference (returns white noise mockup).

## Legal & Privacy (BIPA/GDPR)
- **Never** write raw frames, landmarks, or audio to persistent storage.
- Ephemeral processing in RAM is a legal requirement for this project.
