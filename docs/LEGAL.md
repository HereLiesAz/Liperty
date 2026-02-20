# Legal & Privacy Guidelines

The LipRead-Android application processes sensitive biometric data (facial geometry) and potentially records private conversations. Strict adherence to legal and ethical guidelines is mandatory.

## 1. Wiretap Laws & Recording Consent

### Federal & State Compliance
*   **One-Party vs. All-Party Consent:** While US federal law permits one-party consent (the user), many states (CA, FL, IL, PA, WA, etc.) require **all-party consent**.
*   **Requirement:** Do not record or transcribe conversations without the explicit knowledge and consent of all participants.

### Implementation Mandates
*   **Visual Notification:** The app UI must clearly display a "Recording Active" or "Transcribing" banner when the rear camera is active.
*   **Pre-Recording Consent:** Before enabling the transcription feature, the app should prompt the user to confirm they have obtained consent from the interlocutor.

## 2. Biometric Information Privacy (BIPA/GDPR)

### Face Mesh & Landmarks
*   **Definition:** The 468 facial landmarks extracted by MediaPipe constitute **biometric identifiers**.
*   **Storage Prohibition:** Under laws like the Illinois Biometric Information Privacy Act (BIPA), capturing, storing, or transmitting this data without written consent is illegal.

### Data Minimization Strategy
*   **Ephemeral Processing Only:**
    *   Video frames are processed in volatile memory (RAM).
    *   Facial landmarks are extracted, used for ROI cropping, and immediately discarded.
    *   No video, landmarks, or raw audio is written to persistent storage (disk/NAND) or transmitted to a server.
*   **On-Device Execution:** All ML inference happens locally. No data leaves the device.

## 3. HIPAA & Medical Privacy (If Applicable)

*   If this app is used in a medical setting, transcribed text may constitute Protected Health Information (PHI).
*   **Encryption:** Any saved transcripts (if user explicitly saves them) must be encrypted at rest using Android Keystore.

## 4. Disclaimer

**NO LEGAL ADVICE:** This document is for informational purposes only and does not constitute legal advice. Developers and users are responsible for complying with all applicable laws in their jurisdiction.
