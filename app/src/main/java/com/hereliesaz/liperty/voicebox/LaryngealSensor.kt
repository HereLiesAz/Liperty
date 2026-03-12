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

    private val isRunning = AtomicBoolean(false)
    private var processingJob: Job? = null
    
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
        onVoicingState: (Boolean) -> Unit
    ) {
        if (isRunning.getAndSet(true)) return

        // Start Artificial Larynx vibration
        larynx.start()

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
                        // 1. Convert to Float [-1.0, 1.0]
                        val floatBuffer = FloatArray(read) { audioBuffer[it] / 32768.0f }
                        
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
                            
                            onProcessedAudio(processed)
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

    fun stop() {
        if (!isRunning.getAndSet(false)) return
        larynx.stop()
        processingJob?.cancel()
    }
}
