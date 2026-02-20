# Liperty

**Liperty** is a real-time, on-device Visual Speech Recognition (VSR) application designed for the Deaf, Hard-of-Hearing, and Speech-Impaired communities. It utilizes advanced deep learning models to convert lip movements into text (for lipreading) or synthesized speech (for silent speech interfaces), operating entirely offline to ensure privacy and low latency.

## Features

*   **Real-time Lipreading:** Uses rear camera to transcribe interlocutor speech.
*   **Silent Speech Interface (SSI):** Uses front camera to voice silent mouthing.
*   **Privacy-First:** All processing (Face Mesh, VSR, LLM) happens on-device using LiteRT (TensorFlow Lite).
*   **Telephoto Optimization:** Automatically selects telephoto lenses to reduce perspective distortion.
*   **Gesture Correction:** Innovative swipe-based interface for correcting homophene errors.

## Architecture

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for a deep dive into the technical stack, including:
*   CameraX & MediaPipe for vision pipeline.
*   VALLR / DeepLip model integration.
*   LiteRT Hardware Acceleration (GPU/NPU).

## Setup & Installation

1.  Clone the repository.
2.  Open in Android Studio (Ladybug or newer recommended).
3.  Sync Gradle project.
4.  Connect a physical Android device (Emulator support is limited for NPU/CameraX).
5.  Build and Run.

**Note:** You will need to download the pre-trained `.tflite` models and place them in `app/src/main/assets/`. See `docs/TODO.md` for details on model training/acquisition.

## Legal & Privacy

This application processes biometric data (facial landmarks). See [LEGAL.md](docs/LEGAL.md) for compliance information regarding Wiretap laws and Biometric Privacy acts (BIPA/GDPR).

## Research

Based on the research document in `docs/RESEARCH.md`.
