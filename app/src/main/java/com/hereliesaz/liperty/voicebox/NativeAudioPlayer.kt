package com.hereliesaz.liperty.voicebox

object NativeAudioPlayer {
    init {
        System.loadLibrary("liperty_cv")
    }

    /** Starts the AAudio low-latency stream. Returns true if successful. */
    external fun startPlaybackNative(): Boolean

    /** Pushes PCM float audio data into the native playback buffer. */
    external fun writeAudioDataNative(audioData: FloatArray)

    /** Stops and cleans up the native AAudio stream. */
    external fun stopPlaybackNative()
}
