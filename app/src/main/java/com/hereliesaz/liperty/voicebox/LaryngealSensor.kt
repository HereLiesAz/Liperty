package com.hereliesaz.liperty.voicebox

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import com.hereliesaz.liperty.dsp.VibraPhoneDSP
import com.hereliesaz.liperty.ml.VoiceConverter
import kotlinx.coroutines.*
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.sqrt

/**
 * Real-time voice reconstruction sensor with two operating modes:
 *
 * 1. **BC Mode** — Bone conduction headphones worn around the neck emit a
 *    glottal carrier signal. The user's mouth modulates it; the phone mic
 *    captures the result. DSP + VoiceConverter reconstruct natural speech.
 *
 * 2. **EL Mode** — External electrolarynx held against the throat. The
 *    phone mic captures the buzz; DSP + VoiceConverter process it.
 *
 * Both modes share the same DSP pipeline (spectral subtraction → frequency
 * equalization → mel spectrogram → voice conversion → inverse mel).
 * Voice source expansion is skipped in both because the external device
 * already provides the excitation source.
 */
class LaryngealSensor(private val context: Context) {

    private val dsp = VibraPhoneDSP()
    private val audioRouter = AudioRouter(context)
    private val larynx = ArtificialLarynx(audioRouter)
    private val trambaProcessor = TrambaProcessor(context)
    private val voiceConverter = VoiceConverter(context)

    companion object {
        private const val TAG = "LaryngealSensor"
        private const val BASE_RMS_THRESHOLD = 0.010f
    }

    private val isRunning = AtomicBoolean(false)
    private var processingJob: Job? = null
    @Volatile private var usesBCMode = false
    @Volatile private var sensitivity = 0.5f

    fun setSensitivity(value: Float) {
        sensitivity = value
    }

    /** Adjusts the BC carrier fundamental frequency (80-200 Hz). */
    fun setCarrierF0(hz: Float) {
        larynx.setF0(hz)
    }

    /** Current carrier F0. */
    val carrierF0: Float get() = larynx.f0

    // ── BC Mode ──────────────────────────────────────────────────────────

    /**
     * Starts the Bone Conduction Larynx pipeline.
     *
     * Signal flow:
     * 1. Phone generates glottal pulse carrier → BC headphones (output)
     * 2. Carrier vibrates through throat → mouth modulates it
     * 3. Phone built-in mic captures modulated sound (input)
     * 4. DSP pipeline processes the captured audio
     * 5. Voice conversion maps to the user's cloned voice
     *
     * @param onProcessedAudio Called on IO thread with reconstructed speech audio.
     * @param onVoicingState Called when voice activity is detected/lost.
     * @param onTranscriptionData Called with processed audio for SSR text transcription.
     *        Enables simultaneous voice output + text display.
     */
    fun startBCMode(
        onProcessedAudio: (FloatArray) -> Unit,
        onVoicingState: (Boolean) -> Unit,
        onTranscriptionData: ((FloatArray) -> Unit)? = null
    ) {
        if (isRunning.getAndSet(true)) return
        usesBCMode = true

        processingJob = CoroutineScope(Dispatchers.IO).launch {
            voiceConverter.initialize()
            trambaProcessor.initialize()
            audioRouter.configureFullDuplex()

            // Start carrier signal → BC headphones
            larynx.start()

            val sampleRate = VibraPhoneDSP.SAMPLE_RATE
            val minBufSize = AudioRecord.getMinBufferSize(
                sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
            )

            val recorder = try {
                AudioRecord(
                    MediaRecorder.AudioSource.VOICE_RECOGNITION,
                    sampleRate,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    minBufSize * 4
                )
            } catch (e: SecurityException) {
                Log.e(TAG, "BC mode: RECORD_AUDIO permission not granted", e)
                isRunning.set(false)
                larynx.stop()
                audioRouter.releaseFullDuplex()
                return@launch
            }

            if (recorder.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "BC mode: AudioRecord failed to initialize")
                isRunning.set(false)
                larynx.stop()
                audioRouter.releaseFullDuplex()
                return@launch
            }

            // Force phone built-in mic — NOT the headphone mic
            val phoneMic = audioRouter.getPhoneMic()
            if (phoneMic != null) {
                recorder.setPreferredDevice(phoneMic)
                Log.i(TAG, "BC mode: input forced to built-in mic")
            }

            recorder.startRecording()

            // Noise profile estimation
            val noiseProfile = FloatArray(VibraPhoneDSP.FRAME_SIZE / 2)
            val audioBuffer = ShortArray(VibraPhoneDSP.FRAME_SIZE)

            Log.i(TAG, "BC mode: estimating noise profile...")
            var framesCaptured = 0
            while (framesCaptured < 10 && isActive && isRunning.get()) {
                val read = recorder.read(audioBuffer, 0, audioBuffer.size)
                if (read > 0) framesCaptured++
            }
            for (i in noiseProfile.indices) noiseProfile[i] = 0.002f

            try {
                while (isActive && isRunning.get()) {
                    val read = recorder.read(audioBuffer, 0, audioBuffer.size)
                    if (read > 0) {
                        val floatBuffer = FloatArray(read) { i -> audioBuffer[i] / 32768.0f }

                        // RMS-based VAD
                        val threshold = BASE_RMS_THRESHOLD * (1f - sensitivity).coerceAtLeast(0.1f)
                        val rms = sqrt(floatBuffer.fold(0f) { acc, v -> acc + v * v } / read)
                        val isVoicing = rms > threshold
                        onVoicingState(isVoicing)

                        if (isVoicing) {
                            var processed = dsp.spectralSubtraction(floatBuffer, noiseProfile)
                            processed = dsp.frequencyDomainEqualization(processed)
                            // voiceSourceExpansion SKIPPED — carrier provides excitation

                            val melSpec = dsp.computeMelSpectrogram(processed)
                            val decodedMel = voiceConverter.convert(melSpec, 1, melSpec.size)
                            val humanized = dsp.inverseMelSpectrogram(decodedMel)

                            onProcessedAudio(humanized)
                            onTranscriptionData?.invoke(floatBuffer)
                        } else {
                            onProcessedAudio(FloatArray(read) { 0f })
                        }
                    }
                }
            } finally {
                recorder.stop()
                recorder.release()
                larynx.stop()
                audioRouter.releaseFullDuplex()
            }
        }
    }

    // ── EL Mode ──────────────────────────────────────────────────────────

    /**
     * EL Translator capture path.
     *
     * Uses the built-in mic to capture an external electrolarynx buzz.
     * No carrier generation, no TRAMBA, no accelerometer.
     *
     * Pipeline: spectral subtraction → frequency equalization → Mel spectrogram →
     *           VoiceConverter → inverse Mel → [onProcessedAudio]
     */
    fun startELMode(onProcessedAudio: (FloatArray) -> Unit) {
        if (isRunning.getAndSet(true)) return
        usesBCMode = false

        processingJob = CoroutineScope(Dispatchers.IO).launch {
            voiceConverter.initialize()

            val sampleRate = VibraPhoneDSP.SAMPLE_RATE
            val minBufSize = AudioRecord.getMinBufferSize(
                sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
            )

            val recorder = try {
                AudioRecord(
                    MediaRecorder.AudioSource.MIC,
                    sampleRate,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    minBufSize * 4
                )
            } catch (e: SecurityException) {
                Log.e(TAG, "EL mode: RECORD_AUDIO permission not granted", e)
                isRunning.set(false)
                return@launch
            }

            if (recorder.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "EL mode: AudioRecord failed to initialize")
                isRunning.set(false)
                return@launch
            }

            val preferredDevice = audioRouter.getOptimalInputDevice()
            if (preferredDevice != null) recorder.setPreferredDevice(preferredDevice)

            recorder.startRecording()

            val noiseProfile = FloatArray(VibraPhoneDSP.FRAME_SIZE / 2)
            val audioBuffer = ShortArray(VibraPhoneDSP.FRAME_SIZE)

            Log.i(TAG, "EL mode: estimating noise profile...")
            var framesCaptured = 0
            while (framesCaptured < 10 && isActive && isRunning.get()) {
                val read = recorder.read(audioBuffer, 0, audioBuffer.size)
                if (read > 0) framesCaptured++
            }
            for (i in noiseProfile.indices) noiseProfile[i] = 0.002f

            try {
                while (isActive && isRunning.get()) {
                    val read = recorder.read(audioBuffer, 0, audioBuffer.size)
                    if (read > 0) {
                        val floatBuffer = FloatArray(read) { i -> audioBuffer[i] / 32768.0f }

                        val threshold = BASE_RMS_THRESHOLD * (1f - sensitivity).coerceAtLeast(0.1f)
                        val rms = sqrt(floatBuffer.fold(0f) { acc, v -> acc + v * v } / read)

                        if (rms > threshold) {
                            var processed = dsp.spectralSubtraction(floatBuffer, noiseProfile)
                            processed = dsp.frequencyDomainEqualization(processed)

                            val melSpec = dsp.computeMelSpectrogram(processed)
                            val decodedMel = voiceConverter.convert(melSpec, 1, melSpec.size)
                            val humanized = dsp.inverseMelSpectrogram(decodedMel)
                            onProcessedAudio(humanized)
                        } else {
                            onProcessedAudio(FloatArray(read) { 0f })
                        }
                    }
                }
            } finally {
                recorder.stop()
                recorder.release()
            }
        }
    }

    // ── Lifecycle ────────────────────────────────────────────────────────

    fun stop() {
        if (!isRunning.getAndSet(false)) return
        if (usesBCMode) {
            larynx.stop()
        }
        processingJob?.cancel()
        voiceConverter.close()
    }
}
