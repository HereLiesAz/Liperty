# RESEARCH3 Implementation Plan

This document outlines the engineering tasks required to implement the features detailed in `RESEARCH3.md`, focusing on advanced laryngeal sensing, Silent Speech Recognition (SSR), and ultra-low latency voice generation.

## Phase 8: Hardware Input & Pre-Processing (Bone Conduction)

*   **Integrate Bone Conduction Microphone (BCM) Support:**
    *   Implement input routing to explicitly prioritize BCMs and head-worn accelerometers when connected.
*   **Audio Bandwidth Expansion (Super-Resolution):**
    *   Deploy a lightweight deep-learning model (e.g., TRAMBA architecture) to reconstruct high-frequency vocal components attenuated by tissue transmission.
    *   Execute the model in real-time (ideally via TFLite/NPU) before passing the signal to the transcription/translation engine.

## Phase 9: Architecture 1 - Silent Speech Recognition (SSR)

*   **SSR Translation Module:**
    *   Integrate a CNN-based machine learning model to decode non-auditory physiological signals (bone vibrations) into text strings.
    *   Establish a continuous inference pipeline translating sensor input to text tokens.
*   **Ultra-Low Latency On-Device TTS:**
    *   Optimize the current Text-to-Speech engine (PocketTTS) to operate in a streaming fashion.
    *   Achieve first-word audio latency targets of ~130 milliseconds.

## Phase 10: Architecture 2 - Electrolarynx Translator

*   **Active Noise Suppression:**
    *   Implement pitch-synchronous generalized spectral subtraction to dynamically estimate and subtract the mechanical EL buzz (self-noise) from captured speech in real-time.
*   **Real-Time Voice Cloning (Intelligibility Enhancement):**
    *   Deploy an advanced voice conversion neural network (e.g., Respeecher-style architecture).
    *   Map the cleaned, monotonic EL speech to a high-fidelity, natural-sounding target voice (the patient's original pre-operative voice).

## Phase 11: Wireless Latency & Native Audio Stack

*   **Native C++ Audio Integration:**
    *   Bypass the standard Android Java/Kotlin audio framework.
    *   Implement AAudio or the Oboe C++ wrapper for audio playback and capture.
    *   Request exclusive Memory-Mapped (MMAP) buffers (`AAUDIO_SHARING_MODE_EXCLUSIVE`) and low-latency performance modes (`AAUDIO_PERFORMANCE_MODE_LOW_LATENCY`) to hit ~20ms internal audio latency.
*   **Bluetooth LE Audio & LC3 Codec:**
    *   Ensure the application and hardware capture logic is fully compatible with Bluetooth 5.2/5.3 LE Audio.
    *   Leverage Isochronous Channels (ISOC) and the Low Complexity Communications Codec (LC3/LC3plus) to reduce wireless transmission latency to the 10-15 millisecond range.
