package com.hereliesaz.liperty.voicebox.recording

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import kotlinx.coroutines.*
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Standard Audio Recorder for Voice Cloning (PocketTTS).
 * Captures 16kHz mono 16-bit PCM for extraction of voice characteristics.
 */
class VoiceRecorder(private val context: Context) {

    companion object {
        private const val TAG = "VoiceRecorder"
        private const val SAMPLE_RATE = 16_000
        private const val CHANNEL = AudioFormat.CHANNEL_IN_MONO
        private const val ENCODING = AudioFormat.ENCODING_PCM_16BIT
        private const val AUDIO_SOURCE = MediaRecorder.AudioSource.MIC
        private const val BYTES_PER_SAMPLE = 2
    }

    private val isRecording = AtomicBoolean(false)
    private var recordingJob: Job? = null
    private var audioRecord: AudioRecord? = null

    /**
     * Start recording voice for cloning.
     * @param durationMs Maximum recording time.
     * @param onComplete Callback with the recorded file and sample count.
     */
    fun startRecording(
        durationMs: Long = 10_000L,
        onComplete: (File, Int) -> Unit,
        onError: (String) -> Unit
    ) {
        if (isRecording.getAndSet(true)) {
            onError("Already recording")
            return
        }

        val outputDir = context.getExternalFilesDir(null) ?: run {
            isRecording.set(false)
            onError("External storage unavailable")
            return
        }
        val audioFile = File(outputDir, "voice_clone_sample_${System.currentTimeMillis()}.wav")

        val minBufSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, ENCODING)
        if (minBufSize == AudioRecord.ERROR || minBufSize == AudioRecord.ERROR_BAD_VALUE) {
            isRecording.set(false)
            onError("AudioRecord not supported on this device")
            return
        }

        val recorder = AudioRecord(AUDIO_SOURCE, SAMPLE_RATE, CHANNEL, ENCODING, minBufSize * 4)
        if (recorder.state != AudioRecord.STATE_INITIALIZED) {
            isRecording.set(false)
            onError("AudioRecord init failed")
            return
        }
        audioRecord = recorder

        recordingJob = CoroutineScope(Dispatchers.IO).launch {
            try {
                val startNs = System.nanoTime()
                val pcmData = mutableListOf<Byte>()
                val buffer = ByteArray(minBufSize * 2)

                recorder.startRecording()
                Log.i(TAG, "Voice recording started.")

                while (isActive && isRecording.get()) {
                    val read = recorder.read(buffer, 0, buffer.size)
                    if (read > 0) {
                        for (i in 0 until read) pcmData.add(buffer[i])
                    }
                    val elapsedMs = (System.nanoTime() - startNs) / 1_000_000L
                    if (elapsedMs >= durationMs) break
                }

                recorder.stop()
                recorder.release()
                audioRecord = null
                isRecording.set(false)

                val pcmBytes = pcmData.toByteArray()
                writeWavHeader(audioFile, pcmBytes, SAMPLE_RATE)
                
                onComplete(audioFile, pcmBytes.size / BYTES_PER_SAMPLE)
                Log.i(TAG, "Voice recording complete: ${audioFile.absolutePath}")
            } catch (e: Exception) {
                isRecording.set(false)
                onError("Recording error: ${e.message}")
            }
        }
    }

    fun stopRecording() {
        isRecording.set(false)
        audioRecord?.stop()
        recordingJob?.cancel()
    }

    private fun writeWavHeader(file: File, pcm: ByteArray, sampleRate: Int) {
        val channels = 1
        val bitsPerSample = 16
        val byteRate = sampleRate * channels * bitsPerSample / 8
        val blockAlign = channels * bitsPerSample / 8

        FileOutputStream(file).use { fos ->
            val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
            header.put("RIFF".toByteArray())
            header.putInt(36 + pcm.size)
            header.put("WAVE".toByteArray())
            header.put("fmt ".toByteArray())
            header.putInt(16)
            header.putShort(1)
            header.putShort(channels.toShort())
            header.putInt(sampleRate)
            header.putInt(byteRate)
            header.putShort(blockAlign.toShort())
            header.putShort(bitsPerSample.toShort())
            header.put("data".toByteArray())
            header.putInt(pcm.size)
            fos.write(header.array())
            fos.write(pcm)
        }
    }
}
