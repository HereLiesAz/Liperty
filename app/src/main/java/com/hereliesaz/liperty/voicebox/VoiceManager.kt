package com.hereliesaz.liperty.voicebox

import android.content.Context
import android.speech.tts.TextToSpeech
import android.util.Log
import com.hereliesaz.liperty.voicebox.cloning.VoiceStore
import java.io.File
import java.util.Locale

/**
 * Manages all Text-to-Speech operations for Liperty.
 * Orchestrates between the system TTS and the custom PocketTTS (Voice Cloning) engine.
 */
class VoiceManager(private val context: Context, private val onInit: (Boolean) -> Unit) : TextToSpeech.OnInitListener {

    private var systemTts: TextToSpeech? = null
    private var isSystemTtsReady = false
    
    private val pocketTts: PocketTTSEngine = PocketTTSEngine(context)
    private val voiceStore: VoiceStore = VoiceStore(context)
    
    private var activeVoice: VoiceState? = null

    init {
        systemTts = TextToSpeech(context, this)
        pocketTts.initialize()
        
        // Load preference or default voice
        loadActiveVoice()
    }

    private fun loadActiveVoice() {
        val sharedPrefs = context.getSharedPreferences("LipertyPrefs", Context.MODE_PRIVATE)
        val voiceName = sharedPrefs.getString("active_voice", null)
        if (voiceName != null) {
            val allVoices = voiceStore.loadAllVoices()
            activeVoice = allVoices.find { it.name == voiceName }
        }
    }

    /**
     * Clones a voice from a reference audio file and saves it.
     */
    fun cloneAndSaveVoice(name: String, audioFile: File, onResult: (Boolean) -> Unit) {
        val voiceState = pocketTts.cloneVoice(audioFile)
        if (voiceState != null) {
            val namedState = voiceState.copy(name = name)
            voiceStore.saveVoice(namedState)
            onResult(true)
        } else {
            onResult(false)
        }
    }

    /**
     * Sets the active voice for future speak() calls.
     */
    fun setActiveVoice(voice: VoiceState?) {
        activeVoice = voice
        val sharedPrefs = context.getSharedPreferences("LipertyPrefs", Context.MODE_PRIVATE)
        sharedPrefs.edit().putString("active_voice", voice?.name).apply()
    }

    fun getClonedVoices(): List<VoiceState> = voiceStore.loadAllVoices()

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
