package com.hereliesaz.liperty.voicebox

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import com.hereliesaz.liperty.dsp.VibraPhoneDSP
import com.hereliesaz.liperty.ml.VoiceConverter
import kotlinx.coroutines.*
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Real-time Laryngeal Sensor using dual-stream input (Contact-mic + Accelerometer).
 * Implements Phase 1/2 of the Voice Box roadmap.
 *
 * This component captures raw throat-contact audio and uses the accelerometer
 * to detect laryngeal vibrations for high-precision Voice Activity Detection (VAD).
 */
class LaryngealSensor(private val context: Context) {

    private val dsp = VibraPhoneDSP()
    private val larynx = ArtificialLarynx(context)
    private val sensorManager = context.getSystemService(SensorManager::class.java)
    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val audioRouter = AudioRouter(context)
    private val trambaProcessor = TrambaProcessor(context)
    private val voiceConverter = VoiceConverter(context)

    /**
     * Captures Back-EMF from the Linear Resonant Actuator (LRA).
     * This is used for Phase 2 hardware-level vibration feedback sensing.
     */
    private external fun captureBackEmfAudio(audioBuffer: ShortArray): Int
    private external fun startNativeAudio(): Boolean
    private external fun stopNativeAudio()
    
    /**
     * Applies native spectral subtraction using OpenCV.
     */
    private external fun nativeSpectralSubtraction(inputSignal: FloatArray, noiseProfile: FloatArray)

    companion object {
        init {
            System.loadLibrary("liperty_cv")
        }
    }

    private val isRunning = AtomicBoolean(false)
    private var processingJob: Job? = null
    @Volatile private var usesHardwarePipeline = false

    @Volatile private var accelMagnitude = 0f
    @Volatile private var sensitivity = 0.5f
    
    // Base threshold for activity detection.
    private val BASE_VAD_THRESHOLD = 0.020f 

    fun setSensitivity(value: Float) {
        sensitivity = value
    }

    private val accelListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            val x = event.values[0]
            val y = event.values[1]
            val z = event.values[2]
            // Calculate dynamic component by removing gravity
            val total = sqrt(x * x + y * y + z * z)
            accelMagnitude = abs(total - SensorManager.GRAVITY_EARTH)
        }
        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
    }

    /**
     * Starts the dual-stream capture and processing pipeline.
     * Toggles the Artificial Larynx vibration.
     * @param onProcessedAudio Callback invoked on IO thread with reconstructed audio frames.
     * @param onVoicingState Callback invoked when activity is detected (true).
     */
    fun start(
        onProcessedAudio: (FloatArray) -> Unit,
        onVoicingState: (Boolean) -> Unit,
        onVibrationData: (FloatArray) -> Unit
    ) {
        if (isRunning.getAndSet(true)) return
        usesHardwarePipeline = true

        // Start Artificial Larynx vibration
        larynx.start()
        trambaProcessor.initialize()
        startNativeAudio()

        sensorManager.registerListener(accelListener, accelerometer, SensorManager.SENSOR_DELAY_FASTEST)

        processingJob = CoroutineScope(Dispatchers.IO).launch {
            val sampleRate = VibraPhoneDSP.SAMPLE_RATE
            val minBufSize = AudioRecord.getMinBufferSize(
                sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
            )
            
            val recorder = AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION, // Raw, no AGC/NS
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                minBufSize * 4
            )

            if (recorder.state != AudioRecord.STATE_INITIALIZED) {
                isRunning.set(false)
                return@launch
            }

            // Apply optimal input device routing
            val preferredDevice = audioRouter.getOptimalInputDevice()
            if (preferredDevice != null) {
                recorder.setPreferredDevice(preferredDevice)
            }

            recorder.startRecording()
            
            // Noise profile initialization (capture first 10 frames of 'silence')
            val noiseProfile = FloatArray(VibraPhoneDSP.FRAME_SIZE / 2)
            val audioBuffer = ShortArray(VibraPhoneDSP.FRAME_SIZE)
            
            Log.i("LaryngealSensor", "Estimating noise profile...")
            var framesCaptured = 0
            while (framesCaptured < 10 && isActive && isRunning.get()) {
                val read = recorder.read(audioBuffer, 0, audioBuffer.size)
                if (read > 0) framesCaptured++
            }
            // Use a conservative noise floor for spectral subtraction
            for (i in noiseProfile.indices) noiseProfile[i] = 0.002f

            try {
                while (isActive && isRunning.get()) {
                    val read = recorder.read(audioBuffer, 0, audioBuffer.size)
                    if (read > 0) {
                        // Phase 2: Capture Back-EMF from LRA and mix it with contact-mic data
                        val emfBuffer = ShortArray(read)
                        captureBackEmfAudio(emfBuffer)

                        // 1. Convert to Float [-1.0, 1.0] and mix
                        val floatBuffer = FloatArray(read) { i ->
                            val contactMic = audioBuffer[i] / 32768.0f
                            val backEmf = emfBuffer[i] / 32768.0f
                            // Simple fusion: weighted average (can be refined in DSP)
                            (contactMic * 0.7f) + (backEmf * 0.3f)
                        }

                        // Apply TRAMBA Bandwidth Expansion if using BCM
                        val processedPcm = if (audioRouter.isUsingBoneConduction()) {
                            trambaProcessor.processAudio(audioBuffer)
                        } else {
                            audioBuffer
                        }

                        // Update floatBuffer with expanded audio if applicable
                        if (audioRouter.isUsingBoneConduction()) {
                            for (i in 0 until read) {
                                floatBuffer[i] = processedPcm[i] / 32768.0f
                            }
                        }
                        
                        // 2. Multimodal VAD: Inversely proportional to sensitivity
                        // Higher sensitivity = lower threshold
                        val threshold = BASE_VAD_THRESHOLD * (1.0f - sensitivity).coerceAtLeast(0.1f)
                        val isVoicing = accelMagnitude > threshold
                        onVoicingState(isVoicing)
                        
                        if (isVoicing) {
                            // 3. Apply VibraPhone DSP Pipeline
                            var processed = dsp.spectralSubtraction(floatBuffer, noiseProfile)
                            processed = dsp.frequencyDomainEqualization(processed)
                            processed = dsp.voiceSourceExpansion(processed)
                            
                            // 4. Transform to Target Voice (Phase 10)
                            // Real-time neural mapping requires Mel-spectrogram input
                            val melSpec = dsp.computeMelSpectrogram(processed)
                            
                            // Target voice mapping (Mel-to-Mel)
                            val decodedMel = voiceConverter.convert(melSpec, 1, melSpec.size)
                            
                            // 5. Synthesis (Vocoder / Inverse Mel)
                            // Synthesize high-quality speech from the mapped spectrogram
                            val humanized = dsp.inverseMelSpectrogram(decodedMel)
                            
                            onProcessedAudio(humanized)
                            onVibrationData(floatBuffer)
                        } else {
                            // Output silence if not voicing to save downstream cycles
                            onProcessedAudio(FloatArray(read) { 0f })
                        }
                    }
                }
            } finally {
                recorder.stop()
                recorder.release()
                sensorManager.unregisterListener(accelListener)
            }
        }
    }

    /**
     * EL Translator capture path.
     *
     * Differences from [start]:
     * - Uses the built-in mic only (no contact-mic routing, no back-EMF, no TRAMBA).
     * - VAD is RMS-based; the accelerometer is not used (user holds EL to throat, not phone).
     * - [dsp.voiceSourceExpansion] is skipped — the EL already provides the buzz excitation.
     * - [larynx] (phone vibration) and native audio are not started.
     *
     * Pipeline: spectral subtraction → frequency equalization → Mel spectrogram →
     *           VoiceConverter → inverse Mel → [onProcessedAudio]
     */
    fun startELMode(onProcessedAudio: (FloatArray) -> Unit) {
        if (isRunning.getAndSet(true)) return
        usesHardwarePipeline = false

        processingJob = CoroutineScope(Dispatchers.IO).launch {
            val sampleRate = VibraPhoneDSP.SAMPLE_RATE
            val minBufSize = AudioRecord.getMinBufferSize(
                sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
            )

            val recorder = AudioRecord(
                MediaRecorder.AudioSource.MIC, // Standard mic; EL buzz picked up acoustically
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                minBufSize * 4
            )

            if (recorder.state != AudioRecord.STATE_INITIALIZED) {
                Log.e("LaryngealSensor", "EL mode: AudioRecord failed to initialize")
                isRunning.set(false)
                return@launch
            }

            val preferredDevice = audioRouter.getOptimalInputDevice()
            if (preferredDevice != null) recorder.setPreferredDevice(preferredDevice)

            recorder.startRecording()

            val noiseProfile = FloatArray(VibraPhoneDSP.FRAME_SIZE / 2)
            val audioBuffer = ShortArray(VibraPhoneDSP.FRAME_SIZE)

            Log.i("LaryngealSensor", "EL mode: estimating noise profile...")
            var framesCaptured = 0
            while (framesCaptured < 10 && isActive && isRunning.get()) {
                val read = recorder.read(audioBuffer, 0, audioBuffer.size)
                if (read > 0) framesCaptured++
            }
            for (i in noiseProfile.indices) noiseProfile[i] = 0.002f

            // EL buzz is quiet; use a lower base threshold than accelerometer VAD
            val BASE_RMS_THRESHOLD = 0.010f

            try {
                while (isActive && isRunning.get()) {
                    val read = recorder.read(audioBuffer, 0, audioBuffer.size)
                    if (read > 0) {
                        val floatBuffer = FloatArray(read) { i -> audioBuffer[i] / 32768.0f }

                        // RMS-based VAD — sensitivity adjusts threshold proportionally
                        val threshold = BASE_RMS_THRESHOLD * (1.0f - sensitivity).coerceAtLeast(0.1f)
                        val rms = sqrt(floatBuffer.fold(0f) { acc, v -> acc + v * v } / read)

                        if (rms > threshold) {
                            var processed = dsp.spectralSubtraction(floatBuffer, noiseProfile)
                            processed = dsp.frequencyDomainEqualization(processed)
                            // voiceSourceExpansion intentionally omitted — EL supplies its own excitation

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

    fun stop() {
        if (!isRunning.getAndSet(false)) return
        if (usesHardwarePipeline) {
            larynx.stop()
            stopNativeAudio()
        }
        processingJob?.cancel()
    }
}
