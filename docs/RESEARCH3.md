# **Engineering a Next-Generation Alaryngeal Speech Prosthesis: Integrating Bone Conduction Transducers and Mobile Applications**

## **Introduction to Alaryngeal Voice Rehabilitation**

Following a total laryngectomy—a surgical procedure typically necessitated by advanced laryngeal carcinoma or severe trauma—the patient’s trachea is surgically disconnected from the pharynx to establish a permanent stoma for breathing. This life-saving anatomical alteration permanently removes the vocal folds, effectively eliminating the biological glottal sound source and resulting in aphonia. To restore verbal communication, many patients rely on a traditional electrolarynx (EL).

A conventional EL is a handheld, battery-operated electromechanical device pressed against the cervical skin, delivering repetitive mechanical pressure waves into the vocal tract. Despite its reliability, the legacy EL suffers from profound limitations. The acoustic output is notoriously robotic and monotonic. Furthermore, the traditional piston-driven actuator generates severe acoustic leakage, directly radiating a buzzing sound into the surrounding air at levels approaching 87.6 dBC, which severely degrades overall intelligibility.1

The proliferation of high-performance mobile computing, ultra-low-latency wireless audio protocols, and advanced bone conduction (BC) sensors presents an unprecedented opportunity to fundamentally reinvent this paradigm. Rather than using hardware to physically vibrate the user's neck, modern architectures utilize the smartphone as a highly advanced proxy and translator. By capturing vibrations via Bone Conduction Microphones (BCMs) or translating the audio of a traditional EL via machine learning, developers can output clear, natural-sounding speech through the smartphone's primary speakers.

## **Hardware: Bone Conduction Microphones (BCMs) as Input Sensors**

To facilitate a system where the smartphone speaks on behalf of the user, the application must capture the user's intended speech with high fidelity. Standard air-conduction microphones are highly susceptible to environmental noise.

In contrast, Bone Conduction Microphones (BCMs) and head-worn accelerometers pick up vocal cord and tissue vibrations directly through the skin and skull.2 Because they are not sensitive to changes in air pressure, BCMs are naturally robust against ambient acoustic noise, allowing the system to isolate the user's intent even in crowded, loud environments.2 This makes them the ideal input hardware for capturing silent articulation or the mechanical vibrations of an alaryngeal speaker.

However, tissue transmission inherently attenuates high-frequency vocal components, leading to a muffled and degraded signal.2 To correct this before transcription, the mobile app must implement an audio super-resolution or bandwidth expansion model. Experimental deep-learning architectures, such as the TRAMBA model developed at Northwestern University, process speech from a single head-worn accelerometer to reconstruct these missing high-frequency characteristics in real-time, drastically improving the signal-to-noise ratio and intelligibility before the data is passed to the recognition engine.2

## **Architecture 1: Vibration-to-Text-to-Speech (Silent Speech Recognition)**

In this first architectural paradigm, the user mouths words silently or speaks with preserved, unvoiced articulation. The BCM captures the resulting mechanical vibrations, and the smartphone acts as a voice proxy, translating the silent physical inputs into loud, perfectly clear speech.

### **Translating Movement to Text via SSR**

Following signal capture and high-frequency enhancement, the application employs Silent Speech Recognition (SSR). SSR utilizes advanced machine learning models, such as Convolutional Neural Networks (CNNs), to decode non-auditory physiological signals and predict the intended words. Research focusing on post-laryngectomy speech recognition has proven highly promising, with models capable of decoding surface electromyographic (sEMG) signals and bone vibrations into text strings with high accuracy. By integrating a deep learning-based SSR module, the application continuously translates the structural vibrations picked up by the bone conduction sensor into an accurate text transcription.

### **On-Device Text-to-Speech (TTS) Output**

The final stage of the SSR pipeline is generating audible speech through the smartphone's speakers using Text-to-Speech (TTS). To maintain conversational flow and avoid the jarring delays common in cloud-based APIs, the TTS engine must operate entirely on-device.3 Modern on-device streaming TTS solutions can begin synthesizing speech incrementally as the SSR engine outputs text tokens. By processing the text stream in real-time, highly optimized on-device TTS engines can achieve first-word audio latencies as low as 130 milliseconds without relying on an internet connection, effectively giving the user a real-time, natural-sounding voice.3

## **Architecture 2: The Smartphone as an Electrolarynx Translator**

For patients who prefer to continue using their physical, traditional electrolarynx, the mobile application can be reconfigured to act as an advanced audio translator. In this paradigm, the user speaks using their EL as usual, but utilizes a headset (with an air-conduction microphone or BCM) connected to the smartphone to capture the resulting speech. The app then strips away the mechanical buzz and broadcasts a natural human voice.

### **Active Noise Suppression and Leakage Filtering**

The primary obstacle in EL speech is the massive acoustic leakage (self-noise) radiating from the device's housing, which masks the articulated words.1 The mobile application must first clean the captured audio. Pitch-synchronous generalized spectral subtraction can be applied dynamically.4 The app dynamically estimates the noise spectrum of the mechanical EL buzz from a set of past audio frames and subtracts it from the captured speech signal in real-time.4

### **Real-Time Voice Cloning and Intelligibility Enhancement**

Once the mechanical noise is stripped away, the audio remains monotonic and unnatural. To solve this, the application utilizes AI voice cloning technology. Companies like Respeecher have demonstrated that advanced voice cloning models can take degraded, robotic EL speech as an input and map it to a high-fidelity, natural-sounding target voice.6 By running a localized, low-latency voice conversion neural network, the app can output speech that mimics the patient's original, pre-operative voice, restoring their vocal identity and drastically improving their quality of life and social confidence.6

## **Overcoming the Wireless Latency Bottleneck**

Whether utilizing SSR or EL translation, the transducer/microphone is typically separated from the smartphone via a wireless link. Latency is the single most critical failure point of the entire prosthesis system. If the delay between the user articulating a word and the phone broadcasting it is too high, it disrupts the cognitive feedback loop and destroys conversational fluency.

### **The Android Audio Stack and Native APIs**

Standard audio playback through the Android Java/Kotlin framework introduces massive latency. To achieve deterministic, ultra-low latency, developers must bypass the Java layer entirely and utilize native C++ audio APIs like AAudio or the Oboe C++ wrapper.7 By requesting an exclusive Memory-Mapped (MMAP) buffer (AAUDIO\_SHARING\_MODE\_EXCLUSIVE) and declaring the stream for low-latency performance (AAUDIO\_PERFORMANCE\_MODE\_LOW\_LATENCY), the application writes audio data directly into a buffer shared with the hardware DSP.7 A properly optimized Android application utilizing double buffering and avoiding sample rate conversion can achieve internal audio latencies as low as 20 milliseconds.7

### **Bluetooth LE Audio and the LC3 Codec Solution**

Transmitting the audio data wirelessly introduces severe Bluetooth protocol limitations. Historically, the Advanced Audio Distribution Profile (A2DP) prioritized fidelity but introduced 100 to 200 milliseconds of latency, while the Hands-Free Profile (HFP) achieved lower latency by severely compressing the audio quality.

The advent of Bluetooth Low Energy (LE) Audio and the Low Complexity Communications Codec (LC3) resolves this paradox. LE Audio leverages Isochronous Channels (ISOC) to stream high-fidelity data without buffering bloat. The LC3 codec and its superset, LC3plus, support ultra-low delay profiles with latencies commonly dropping into the 10–15 millisecond range, well below the threshold of human perception.8 To leverage this breakthrough, the hardware capturing the user's input must feature a modern Bluetooth 5.2/5.3 chipset compatible with LE Audio and LC3 decoding.

## **Conclusion**

By pivoting away from physical tissue excitation and instead utilizing bone conduction sensors and standard microphones as inputs, developers can transform the smartphone into a powerful voice prosthesis. Whether translating silent vibrations into text-to-speech, or using AI voice cloning to filter and naturalize the mechanical buzz of a traditional electrolarynx, this software-driven architecture bypasses the physical limitations of legacy medical hardware. Supported by the ultra-low latency of Bluetooth LE Audio and native MMAP processing, this approach has the potential to return a fluid, expressive, and highly natural voice to the laryngectomized population.

#### **Works cited**

1. 3D Simulation of an Audible Ultrasonic Electrolarynx Using Difference Waves \- PMC, accessed March 14, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC4234661/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4234661/)  
2. Improving Acoustic and Bone Conduction Speech Enhancement | News \- Northwestern's McCormick School of Engineering, accessed March 14, 2026, [https://www.mccormick.northwestern.edu/news/articles/2024/12/improving-acoustic-and-bone-conduction-speech-enhancement/](https://www.mccormick.northwestern.edu/news/articles/2024/12/improving-acoustic-and-bone-conduction-speech-enhancement/)  
3. Android Real-Time TTS: Streaming Text-to-Speech Tutorial \[2025\] \- Picovoice, accessed March 14, 2026, [https://picovoice.ai/blog/android-streaming-text-to-speech/](https://picovoice.ai/blog/android-streaming-text-to-speech/)  
4. Real-time enhancement of electrolaryngeal speech by spectral subtraction \- IEEE Xplore, accessed March 14, 2026, [https://ieeexplore.ieee.org/document/6176807/](https://ieeexplore.ieee.org/document/6176807/)  
5. Enhancement of electrolaryngeal speech by spectral subtraction, spectral compensation, and introduction of jitter and shimmer \- SciSpace, accessed March 14, 2026, [https://scispace.com/pdf/enhancement-of-electrolaryngeal-speech-by-spectral-1g3vor0pms.pdf](https://scispace.com/pdf/enhancement-of-electrolaryngeal-speech-by-spectral-1g3vor0pms.pdf)  
6. Respeecher's Voice Synthesis: Restoring Natural Speech for Laryngectomy Patients, accessed March 14, 2026, [https://www.respeecher.com/case-studies/respeecher-helps-patients-speech-disabilities-recover-voice](https://www.respeecher.com/case-studies/respeecher-helps-patients-speech-disabilities-recover-voice)  
7. Low latency audio | Android game development, accessed March 14, 2026, [https://developer.android.com/games/sdk/oboe/low-latency-audio](https://developer.android.com/games/sdk/oboe/low-latency-audio)  
8. Gaming is Next Driver for Bluetooth Low Energy and LE Audio \- Ceva's IP, accessed March 14, 2026, [https://www.ceva-ip.com/blog/gaming-is-next-driver-for-bluetooth-low-energy-and-le-audio/](https://www.ceva-ip.com/blog/gaming-is-next-driver-for-bluetooth-low-energy-and-le-audio/)
