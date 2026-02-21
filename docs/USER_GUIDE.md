# Liperty User Guide

Liperty is an offline-first Visual Speech Recognition (VSR) application for Android. It uses on-device machine learning to transcribe speech from lip movements.

## Installation

1.  Clone the repository.
2.  Open in Android Studio (Iguana or later).
3.  Ensure your device runs Android 8.0 (Oreo) or higher.
4.  Build and Run the `app` module.

**Note:** The application requires the MediaPipe Face Mesh model (`face_landmarker.task`) in `app/src/main/assets/`.

## Permissions

On first launch, the app will request:
*   **Camera Permission:** Required to capture video for lip reading.

You must also agree to the "Legal Consent" dialog to proceed.

## Usage

### Main Screen

*   **Camera Preview:** Shows the live video feed.
*   **Overlay:**
    *   **Green Box:** Detected Face.
    *   **Blue Box:** Tracked Mouth Region.
*   **Transcription:** Real-time text appears at the bottom.

### Gestures

*   **Swipe Up:** Speak the current transcription using Text-to-Speech (TTS).
*   **Double Tap:** Clear the transcription and reset the buffer.
*   **Swipe Left/Right:** Cycle through alternative words (Homophene Correction) for the selected word.

### Camera Controls

*   **Switch Camera:** Tap the button to toggle between Front and Rear cameras.
    *   *Rear Camera:* Prioritizes Telephoto lens if available for better lip capture.

## Troubleshooting

*   **"Model Missing":** If the VSR model is not found, the app runs in dummy mode. Ensure `vsr_model.tflite` is in assets (currently using placeholder logic).
*   **No Face Detected:** Ensure good lighting and face the camera directly.
