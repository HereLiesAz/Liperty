# Liperty User Guide

Welcome to **Liperty**, the offline Visual Speech Recognition (VSR) app.

## Getting Started

### 1. Permissions & Consent
On first launch, you will be prompted to grant **Camera** permissions.
You must also accept the **Legal Consent** dialog regarding biometric data processing. All processing happens locally on your device; no video is sent to the cloud.

### 2. Camera Modes
*   **Lipreading Mode (Rear Camera):** Point the camera at a speaker. The app automatically selects the Telephoto lens (if available) to improve accuracy.
*   **Silent Speech Interface (Front Camera):** Tap the **Switch Cam** button to use the front camera for silent mouthing.

### 3. Usage
*   **Align:** Ensure the speaker's face is visible. A **Green Box** indicates face detection. A **Blue Box** indicates lip tracking.
*   **Transcribe:** The app continuously buffers video frames. When speech is detected (simulated in prototype), text will appear at the bottom.
*   **Correcting Errors:**
    *   **Swipe Left/Right** on the text to cycle through similar-looking words (homophenes). e.g., "pat" <-> "bat" <-> "mat".
    *   **Double Tap** anywhere to clear the current transcript.
*   **Speak (TTS):**
    *   **Swipe Up** to have the app speak the current sentence aloud using Text-to-Speech.

## Troubleshooting

*   **"No face detected":** Improve lighting or move closer to the camera.
*   **Performance:** Older devices may experience lower FPS. The app displays performance stats (FPS/Inference Time) in debug logs.
*   **Permissions:** If denied, go to Android Settings -> Apps -> Liperty -> Permissions and enable Camera.

## Privacy
Your privacy is paramount.
*   **Offline First:** AI models run on-device (LiteRT).
*   **Ephemeral:** Video frames are processed in RAM and discarded. They are not saved to storage.
