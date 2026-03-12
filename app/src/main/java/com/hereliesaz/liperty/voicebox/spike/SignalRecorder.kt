package com.hereliesaz.liperty.voicebox.spike

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.io.FileWriter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Phase 0 Spike: Dual-stream recorder.
 *
 * Captures accelerometer + contact-mic simultaneously while the phone is
 * pressed against the user's larynx. Output files are pulled and analyzed
 * with tools/spike_analysis/analyze.py to determine signal viability for
 * the VibraPhone-inspired DSP pipeline.
 *
 * SENSOR RATIONALE (RESEARCH.md / VibraPhone MobiSys 2016):
 *   - LRA back-EMF is not accessible on stock Android. The Vibrator API is
 *     write-only; back-EMF reading requires hardware modification (soldering).
 *   - Primary signal: microphone pressed to throat. Bone/tissue conduction
 *     produces the same spectral characteristics as back-EMF: attenuated highs,
 *     distorted formants, deaf above ~2kHz. Same VibraPhone DSP pipeline applies.
 *   - Secondary signal: accelerometer captures low-frequency laryngeal vibration
 *     (F0 range 85–400 Hz) for Voice Activity Detection.
 *   - Back-EMF remains a Phase 2+ aspirational path (rooted device / custom driver).
 *
 * Output files (written to app external files dir):
 *   spike_audio_<timestamp>.wav   — 16kHz mono 16-bit PCM, RIFF/WAV header
 *   spike_accel_<timestamp>.csv   — columns: timestamp_ns, x, y, z
 *
 * Pull from device:
 *   adb pull /sdcard/Android/data/com.hereliesaz.liperty/files/ ./spike_data/
 *
 * Then run: python3 tools/spike_analysis/analyze.py ./spike_data/
 */
class SignalRecorder(private val context: Context) {

    companion object {
        const val AUDIO_SAMPLE_RATE = 16_000        // Hz — matches downstream TFLite expectations
        const val AUDIO_CHANNEL     = AudioFormat.CHANNEL_IN_MONO
        const val AUDIO_ENCODING    = AudioFormat.ENCODING_PCM_16BIT
        // VOICE_RECOGNITION disables AGC, noise suppression, echo cancellation —
        // essential for preserving raw laryngeal contact signal.
        const val AUDIO_SOURCE      = MediaRecorder.AudioSource.VOICE_RECOGNITION
        const val BYTES_PER_SAMPLE  = 2             // 16-bit PCM = 2 bytes
    }

    private val sensorManager = context.getSystemService(SensorManager::class.java)
    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

    private val isRecording = AtomicBoolean(false)
    private var audioJob: Job? = null
    private var audioRecord: AudioRecord? = null

    private val accelSamples = mutableListOf<String>()
    private val accelLock    = Any()

    data class RecordingResult(
        val audioFile: File,
        val accelFile: File,
        val durationMs:   Long,
        val audioSamples: Int,
        val accelSamples: Int
    )

    // -------------------------------------------------------------------------
    // Accelerometer listener
    // -------------------------------------------------------------------------

    private val accelListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            if (!isRecording.get()) return
            // SensorEvent.timestamp is in nanoseconds, same epoch as System.nanoTime().
            // This lets us align accel and audio streams in the analysis script.
            val line = "${event.timestamp},${event.values[0]},${event.values[1]},${event.values[2]}"
            synchronized(accelLock) { accelSamples.add(line) }
        }
        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
    }

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /**
     * Begin a dual-stream recording session.
     *
     * @param durationMs  Recording length in milliseconds. Cap at 30s for spike use.
     * @param onComplete  Callback on the IO thread with result metadata and file refs.
     * @param onError     Callback if initialization fails.
     */
    fun record(
        durationMs: Long = 10_000L,
        onComplete: (RecordingResult) -> Unit,
        onError:    (String) -> Unit
    ) {
        if (isRecording.getAndSet(true)) {
            onError("Already recording")
            return
        }

        val timestamp = System.currentTimeMillis()
        val outputDir = context.getExternalFilesDir(null) ?: run {
            isRecording.set(false)
            onError("External storage unavailable")
            return
        }

        val audioFile = File(outputDir, "spike_audio_$timestamp.wav")
        val accelFile = File(outputDir, "spike_accel_$timestamp.csv")

        // Setup AudioRecord
        val minBuf = AudioRecord.getMinBufferSize(AUDIO_SAMPLE_RATE, AUDIO_CHANNEL, AUDIO_ENCODING)
        if (minBuf == AudioRecord.ERROR_BAD_VALUE || minBuf == AudioRecord.ERROR) {
            isRecording.set(false)
            onError("AudioRecord not supported on this device")
            return
        }
        // 4× minimum buffer reduces overrun risk on loaded devices
        val bufSize  = minBuf * 4
        val recorder = AudioRecord(
            AUDIO_SOURCE, AUDIO_SAMPLE_RATE, AUDIO_CHANNEL, AUDIO_ENCODING, bufSize
        )
        if (recorder.state != AudioRecord.STATE_INITIALIZED) {
            isRecording.set(false)
            onError("AudioRecord init failed — check RECORD_AUDIO permission")
            return
        }
        audioRecord = recorder

        // Register accelerometer
        // SENSOR_DELAY_FASTEST (~200 Hz on most devices): needed to resolve F0 up to
        // 255 Hz (Nyquist requires >510 Hz). SENSOR_DELAY_GAME (~50 Hz) would alias.
        synchronized(accelLock) { accelSamples.clear() }
        sensorManager.registerListener(
            accelListener, accelerometer, SensorManager.SENSOR_DELAY_FASTEST
        )

        // Recording coroutine on IO dispatcher — avoids blocking main thread
        audioJob = CoroutineScope(Dispatchers.IO).launch {
            val startNs  = System.nanoTime()
            val pcmChunk = ByteArray(bufSize)
            val pcmData  = mutableListOf<Byte>()

            recorder.startRecording()

            while (isActive && isRecording.get()) {
                val read = recorder.read(pcmChunk, 0, bufSize)
                if (read > 0) {
                    for (i in 0 until read) pcmData.add(pcmChunk[i])
                }
                val elapsedMs = (System.nanoTime() - startNs) / 1_000_000L
                if (elapsedMs >= durationMs) break
            }

            val durationActual = (System.nanoTime() - startNs) / 1_000_000L
            recorder.stop()
            recorder.release()
            audioRecord = null
            isRecording.set(false)
            sensorManager.unregisterListener(accelListener)

            // Flush audio as WAV
            val pcmBytes = pcmData.toByteArray()
            writeWav(audioFile, pcmBytes, AUDIO_SAMPLE_RATE)

            // Flush accelerometer CSV
            val accelSnap: List<String>
            synchronized(accelLock) { accelSnap = accelSamples.toList() }
            BufferedWriter(FileWriter(accelFile)).use { writer ->
                writer.write("timestamp_ns,x,y,z\n")
                accelSnap.forEach { writer.write("$it\n") }
            }

            onComplete(
                RecordingResult(
                    audioFile    = audioFile,
                    accelFile    = accelFile,
                    durationMs   = durationActual,
                    audioSamples = pcmBytes.size / BYTES_PER_SAMPLE,
                    accelSamples = accelSnap.size
                )
            )
        }
    }

    /** Stop an in-progress recording before its natural end. */
    fun stop() {
        isRecording.set(false)
        audioRecord?.stop()
        audioJob?.cancel()
    }

    // -------------------------------------------------------------------------
    // WAV header writer
    // -------------------------------------------------------------------------

    /**
     * Wraps raw 16-bit mono PCM bytes in a standard RIFF/WAV container.
     * Reference: http://soundfile.sapp.org/doc/WaveFormat/
     */
    private fun writeWav(file: File, pcm: ByteArray, sampleRate: Int) {
        val channels      = 1
        val bitsPerSample = 16
        val byteRate      = sampleRate * channels * bitsPerSample / 8
        val blockAlign    = channels * bitsPerSample / 8

        FileOutputStream(file).use { fos ->
            val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
            // RIFF chunk
            header.put("RIFF".toByteArray(Charsets.US_ASCII))
            header.putInt(36 + pcm.size)       // ChunkSize
            header.put("WAVE".toByteArray(Charsets.US_ASCII))
            // fmt sub-chunk
            header.put("fmt ".toByteArray(Charsets.US_ASCII))
            header.putInt(16)                  // Subchunk1Size for PCM
            header.putShort(1)                 // AudioFormat: 1 = PCM (uncompressed)
            header.putShort(channels.toShort())
            header.putInt(sampleRate)
            header.putInt(byteRate)
            header.putShort(blockAlign.toShort())
            header.putShort(bitsPerSample.toShort())
            // data sub-chunk
            header.put("data".toByteArray(Charsets.US_ASCII))
            header.putInt(pcm.size)            // Subchunk2Size
            fos.write(header.array())
            fos.write(pcm)
        }
    }
}