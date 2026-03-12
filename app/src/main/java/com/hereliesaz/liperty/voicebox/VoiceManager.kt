package com.hereliesaz.liperty.voicebox

import android.content.Context
import android.speech.tts.TextToSpeech
import android.util.Log
import java.util.Locale

/**
 * Manages all Text-to-Speech operations for Liperty.
 * Orchestrates between the system TTS and the custom PocketTTS (Voice Cloning) engine.
 */
class VoiceManager(context: Context, private val onInit: (Boolean) -> Unit) : TextToSpeech.OnInitListener {

    private var systemTts: TextToSpeech? = null
    private var isSystemTtsReady = false
    
    // Future: PocketTTS engine
    // private var pocketTts: PocketTTSEngine? = null

    init {
        systemTts = TextToSpeech(context, this)
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = systemTts?.setLanguage(Locale.US)
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e("VoiceManager", "System TTS Language not supported")
                onInit(false)
            } else {
                isSystemTtsReady = true
                onInit(true)
            }
        } else {
            Log.e("VoiceManager", "System TTS Initialization failed")
            onInit(false)
        }
    }

    fun speak(text: String, queueMode: Int = TextToSpeech.QUEUE_FLUSH) {
        // For now, only system TTS is implemented.
        // Later: Add logic to check user preference and use PocketTTS if selected.
        if (isSystemTtsReady) {
            systemTts?.speak(text, queueMode, null, null)
        } else {
            Log.w("VoiceManager", "TTS not ready. Cannot speak: $text")
        }
    }

    fun stop() {
        systemTts?.stop()
        // pocketTts?.stop()
    }

    fun shutdown() {
        systemTts?.shutdown()
        // pocketTts?.shutdown()
    }
}
