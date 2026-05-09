package com.hereliesaz.liperty.voicebox

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
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
    private var streamingAudioTrack: AudioTrack? = null

    init {
        systemTts = TextToSpeech(context, this)
        pocketTts.initialize()
        
        // Load preference or default voice
        loadActiveVoice()
    }

    fun loadActiveVoice() {
        val sharedPrefs = context.getSharedPreferences("LipertyPrefs", Context.MODE_PRIVATE)
        val voiceName = sharedPrefs.getString("active_voice", null)
        if (voiceName != null) {
            val allVoices = voiceStore.loadAllVoices()
            activeVoice = allVoices.find { it.name == voiceName }
        } else {
            activeVoice = null
        }
    }

    /**
     * Clones a voice from one or more reference audio files and saves it.
     */
    fun cloneAndSaveVoice(name: String, audioFiles: List<File>, onResult: (Boolean) -> Unit) {
        try {
            val voiceState = pocketTts.cloneVoice(name, audioFiles)
            voiceStore.saveVoice(voiceState)
            onResult(true)
        } catch (e: Exception) {
            Log.e("VoiceManager", "Voice cloning failed", e)
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
        val voice = activeVoice
        if (voice != null) {
            // Use PocketTTS for cloned voice
            val audio = pocketTts.generateAudio(text, voice)
            if (audio != null) {
                playAudioStatic(audio)
            } else {
                Log.e("VoiceManager", "Failed to generate audio with PocketTTS")
            }
        } else {
            // Fallback to system TTS
            if (isSystemTtsReady) {
                systemTts?.speak(text, queueMode, null, null)
            } else {
                Log.w("VoiceManager", "TTS not ready. Cannot speak: $text")
            }
        }
    }

    /**
     * Streams TTS output for ultra-low latency.
     * Tokens are synthesized and played as soon as they are ready.
     */
    fun speakStreaming(text: String) {
        val voice = activeVoice ?: return
        
        // Ensure streaming track is ready
        initStreamingTrack()
        
        // Sequence processing (should be called from a background thread)
        pocketTts.generateAudioStreaming(text, voice).forEach { chunk ->
            streamingAudioTrack?.write(chunk, 0, chunk.size, AudioTrack.WRITE_BLOCKING)
            streamingAudioTrack?.play() // Ensure it's playing
        }
    }

    private fun initStreamingTrack() {
        if (streamingAudioTrack == null) {
            try {
                val minBufSize = AudioTrack.getMinBufferSize(
                    16000,
                    AudioFormat.CHANNEL_OUT_MONO,
                    AudioFormat.ENCODING_PCM_FLOAT
                )
                val track = AudioTrack.Builder()
                    .setAudioAttributes(AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build())
                    .setAudioFormat(AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                        .setSampleRate(16000)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .build())
                    .setBufferSizeInBytes(Math.max(minBufSize, 8192))
                    .setTransferMode(AudioTrack.MODE_STREAM)
                    .build()
                if (track.state == AudioTrack.STATE_INITIALIZED) {
                    streamingAudioTrack = track
                } else {
                    Log.e("VoiceManager", "Streaming AudioTrack failed to initialize")
                    track.release()
                }
            } catch (e: Exception) {
                Log.e("VoiceManager", "Failed to create streaming AudioTrack", e)
            }
        }
    }

    fun playAudio(samples: FloatArray) = playAudioStatic(samples)

    private fun playAudioStatic(samples: FloatArray) {
        val audioTrack: AudioTrack
        try {
            audioTrack = AudioTrack.Builder()
                .setAudioAttributes(AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build())
                .setAudioFormat(AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setSampleRate(16000)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build())
                .setBufferSizeInBytes(samples.size * 4)
                .setTransferMode(AudioTrack.MODE_STATIC)
                .build()
        } catch (e: Exception) {
            Log.e("VoiceManager", "Failed to create AudioTrack", e)
            return
        }

        if (audioTrack.state != AudioTrack.STATE_INITIALIZED) {
            Log.e("VoiceManager", "AudioTrack not initialized")
            audioTrack.release()
            return
        }

        try {
            audioTrack.setPlaybackPositionUpdateListener(object : AudioTrack.OnPlaybackPositionUpdateListener {
                override fun onMarkerReached(track: AudioTrack) {
                    track.stop()
                    track.release()
                }
                override fun onPeriodicNotification(track: AudioTrack) = Unit
            })
            audioTrack.setNotificationMarkerPosition(samples.size)
            audioTrack.write(samples, 0, samples.size, AudioTrack.WRITE_BLOCKING)
            audioTrack.play()
        } catch (e: Exception) {
            Log.e("VoiceManager", "Playback failed", e)
            audioTrack.release()
        }
    }

    fun stop() {
        systemTts?.stop()
        // pocketTts?.stop()
    }

    fun shutdown() {
        systemTts?.shutdown()
        streamingAudioTrack?.apply {
            stop()
            release()
        }
        streamingAudioTrack = null
    }
}
