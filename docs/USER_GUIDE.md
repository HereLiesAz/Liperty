# Liperty User Guide

Liperty is an offline-first Visual Speech Recognition (VSR) and Silent Speech Interface (SSI) application for Android.

## Installation

1.  Clone the repository.
2.  Run `./setup_libs.sh` to download necessary models and SDKs.
3.  Open in Android Studio and Build/Run the `app` module.

## Permissions

On first launch, the app will request:
*   **Camera:** To capture video for lip reading.
*   **Record Audio:** For voice activity detection and voice cloning.
*   **Vibrate:** To power the Artificial Larynx functionality.

You must also agree to the **Legal Consent** dialog to proceed with biometric processing.

## Usage Modes

### 1. Lip Reading (VSR)
*   Toggle **Lip-Read ON** in the navigation rail.
*   Aim the camera at an interlocutor.
*   **Interactive Text:** Tappable word blocks appear over the camera view.
*   **Pinch-to-Zoom:** Use two fingers to scale the transcription text size dynamically.

### 2. Voice Box (SSI)
*   Toggle **Larynx ON** in the navigation rail.
*   The phone will vibrate intensely to provide a carrier sound source.
*   Press the back of the phone firmly against your throat, below the Adam's apple.
*   Mouth words silently; your articulators will modulate the vibration into a signal the app can process.

### 3. Voice Management
*   Navigate to the **Voice** screen.
*   **Clone Voice:** Import a `.wav` file or record a sample to create a personalized TTS profile.
*   **Select Profile:** Choose from saved voices for the "Speak" functionality.

### 4. Personalization (Tweak)
*   Navigate to **Tweak** in the rail.
*   Mouth the provided phonetically rich phrases to fine-tune the model to your specific articulatory patterns (LoRA).

## Gestures & Controls

*   **Wave Gesture:** Rapidly wave your hand in front of the camera to **Pause/Resume** transcription.
*   **Air Swipes:** Wave horizontally (left/right) to cycle through **Homophene candidates** for the selected word.
*   **Touch Word:** Tap any word in the transcription to select it for manual correction or gesture-based cycling.
*   **Sensitivity Sliders:** Use the vertical sliders on the home screen to adjust VSR confidence and LRA pressure thresholds.

## Troubleshooting

*   **Portrait Orientation:** Ensure the phone is held vertically. The recognition engine is optimized for portrait alignment.
*   **No Face Detected:** Ensure good lighting and that the mouth is clearly visible.
*   **Weak SSI Signal:** Ensure the phone is pressed firmly against the throat and any thick protective cases are removed.
