# Privacy Policy

**Liperty** — Visual Speech Recognition & Voice Reconstruction App

*Last updated: May 8, 2026*

## 1. Overview

Liperty is a fully offline Android application that provides visual speech recognition (lipreading) and voice reconstruction for Deaf, Hard-of-Hearing, and speech-impaired users. This privacy policy explains what data the app accesses, how it is used, and how it is protected.

**The core principle: all processing happens on your device. No data is transmitted to any server.**

## 2. Data We Access

### Camera
The app uses your device camera to capture video of your face in real time. Video frames are processed in volatile memory (RAM) to detect facial landmarks and extract lip movements for speech recognition. Frames are never saved to disk.

### Microphone
When the Bone Conduction Larynx or Electrolarynx Translator modes are active, the app accesses the device microphone to capture audio for voice reconstruction. Audio is processed in real-time in RAM and is not recorded or stored.

### Bluetooth
The app may connect to Bluetooth or Bluetooth LE Audio devices (such as bone conduction headphones) to route audio signals for voice reconstruction. No personal data is transmitted over Bluetooth — only generated carrier audio signals.

### Sensors
The app may use the device vibrator for haptic feedback. No sensor data is stored or transmitted.

## 3. Biometric Data

### What We Process
- **Facial landmarks**: 468 facial mesh points extracted by MediaPipe for lip-region detection
- **Lip geometry**: Mouth region crops used as input to the speech recognition model
- **Voice characteristics**: Mel spectrograms derived from microphone audio during voice reconstruction

### How We Handle It
- All biometric data is processed **exclusively in volatile memory (RAM)**
- Facial landmarks are extracted, used for lip cropping, and **immediately discarded**
- No facial geometry, lip images, voice recordings, or raw audio is ever written to persistent storage (disk/NAND)
- No biometric data is transmitted to any server, cloud service, or third party
- The app has **no internet permission** and cannot send data off-device

### Compliance
This handling is designed to comply with:
- **Illinois Biometric Information Privacy Act (BIPA)** — no collection, storage, or transmission of biometric identifiers without consent
- **EU General Data Protection Regulation (GDPR)** — data minimization, purpose limitation, processing only with consent
- **California Consumer Privacy Act (CCPA)** — no sale or sharing of personal information

## 4. Voice Profiles

If you use the voice import feature to create a voice profile for personalized text-to-speech:
- Voice profile data (speaker embeddings) is stored **locally on your device only**
- You can delete your voice profile at any time through the app's Voice Management screen
- Voice profiles are never uploaded, shared, or transmitted

## 5. Transcription Data

- Transcribed text from lipreading or voice reconstruction is displayed on-screen and held in memory
- Transcription text is **not automatically saved** to disk
- If you explicitly choose to save a transcript, it is stored locally on your device
- No transcription data is sent to any server

## 6. Data We Do NOT Collect

Liperty does **not** collect, store, or transmit:
- Video or images of your face
- Audio recordings
- Location data
- Device identifiers or advertising IDs
- Usage analytics or telemetry
- Any data to third-party services
- Crash reports (no network capability)

## 7. No Network Access

Liperty has **no internet permission** (`android.permission.INTERNET` is absent from the manifest). The app is physically incapable of transmitting any data off your device. All machine learning inference runs locally using on-device models.

## 8. On-Device Machine Learning

All ML models run locally on your device:
- Visual speech recognition model (TensorFlow Lite)
- Face landmark detection (MediaPipe)
- Voice conversion model (TensorFlow Lite)
- Text-to-speech synthesis (ONNX Runtime)

No cloud ML services are used. Model weights are bundled with the app or downloaded once during setup and stored locally.

## 9. Consent

On first launch, the app displays a mandatory consent dialog explaining that it will process facial biometric data. You must explicitly agree before the camera activates. You may decline, in which case the app will not function.

## 10. Children's Privacy

Liperty does not knowingly process data from children under 13. The app's consent dialog must be acknowledged by a user who can legally provide consent.

## 11. Data Retention

- **Biometric data**: Zero retention — processed in RAM and discarded immediately
- **Transcription text**: In-memory only unless you explicitly save it
- **Voice profiles**: Stored locally until you delete them
- **Preferences**: App settings stored in local SharedPreferences (no biometric data)

## 12. Your Rights

Since all data stays on your device, you have complete control:
- **Access**: All data is visible to you in the app
- **Deletion**: Uninstalling the app removes all locally stored data
- **Portability**: Voice profiles and saved transcripts are stored as local files you can manage
- **Opt-out**: You can disable any mode (lipreading, voice reconstruction) at any time

## 13. Changes to This Policy

We may update this privacy policy to reflect changes in the app's functionality. The "Last updated" date at the top indicates the most recent revision.

## 14. Contact

For questions about this privacy policy or the app's data practices, please open an issue on our GitHub repository or contact the developer.

---

*Liperty is an open-source project committed to privacy-first design for assistive technology.*
