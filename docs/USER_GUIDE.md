# Liperty User Guide

Liperty is an offline-first Visual Speech Recognition (VSR) and Voice Reconstruction app for Android.

> ⚠️ Lipreading accuracy is **experimental and not yet validated** — see the accuracy note in the README. Treat transcriptions as a best-effort draft.

## Installation

1. Clone the repository and run `./setup_libs.sh` (downloads the OpenCV SDK + MediaPipe models and patches the build).
2. Open in Android Studio (JDK 17) and Build/Run the `app` module on a physical device.
3. On **first launch the app downloads ~1.4 GB of ML models** over Wi-Fi (one time); after that it runs fully offline.

## Permissions

On first launch the app requests:
* **Camera** — capture video for lip reading.
* **Record Audio** — voice-activity detection and voice cloning.
* **Vibrate** — power the Artificial Larynx (bone-conduction) mode.

You must also accept the **Legal Consent** dialog before any biometric processing.

## Usage modes

### 1. Lip Reading (VSR)
* Toggle **Lip-Read ON** in the navigation rail.
* Aim the camera at the speaker, holding the phone in **portrait** with the mouth well-lit and clearly visible.
* The transcription appears as a text overlay over the camera view.

### 2. Voice Box (Artificial Larynx / SSI)
* Toggle **Larynx ON** in the navigation rail.
* The phone vibrates to provide a carrier sound source.
* Press the back of the phone firmly against your throat, below the Adam's apple, and mouth words silently; your articulators modulate the vibration into a signal the app processes.

### 3. Voice Management
* Navigate to the **Voice** screen.
* **Clone Voice:** import a `.wav` file or record a sample to create a personalized TTS profile.
* **Select Profile:** choose a saved voice for the "Speak" functionality.

## Planned features (not yet shipped)

These are on the roadmap ([`TODO.md`](TODO.md)) and **not available in the current build**:

* **Interactive transcription** — tappable word blocks over the camera, pinch-to-zoom font scaling, and per-word confidence shading.
* **Gesture control** — wave to pause/resume; horizontal air-swipes to cycle homophene candidates.
* **Personalization ("Tweak")** — mouthing phonetically rich phrases to fine-tune the model to your articulation (on-device LoRA).
* **Electrolarynx translator & Bone-Conduction (BCM) modes** — hardware electrolarynx buzz suppression and bone-conduction-mic capture with bandwidth expansion, plus Bluetooth LE Audio (LC3) low-latency output.

## Troubleshooting

* **Portrait orientation:** hold the phone vertically; recognition is tuned for portrait alignment.
* **No face detected:** ensure good lighting and an unobstructed mouth.
* **Weak SSI signal:** press the phone firmly against the throat and remove thick cases.
