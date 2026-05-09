package com.hereliesaz.liperty.voicebox

import android.media.AudioAttributes
import android.media.AudioDeviceInfo
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log
import com.hereliesaz.liperty.dsp.VibraPhoneDSP
import kotlinx.coroutines.*
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.PI
import kotlin.math.sin

/**
 * Generates a continuous Rosenberg-model glottal pulse train and streams it
 * to an [AudioTrack]. Designed to drive bone conduction headphones worn
 * around the neck as an artificial larynx carrier signal.
 *
 * The waveform is phase-continuous across buffer boundaries so there are
 * no clicks or pops when the buffer refills.
 */
class GlottalCarrierGenerator {

    companion object {
        private const val TAG = "GlottalCarrier"
        private const val SAMPLE_RATE = VibraPhoneDSP.SAMPLE_RATE  // 16 kHz
        private const val BUFFER_FRAMES = 1024
        private const val AMPLITUDE = 0.7f
    }

    @Volatile
    var f0: Float = 120f
        private set

    private val isRunning = AtomicBoolean(false)
    private var audioTrack: AudioTrack? = null
    private var generatorJob: Job? = null

    /** Phase accumulator — persists across buffers for click-free output. */
    private var phase: Float = 0f

    /**
     * Starts streaming the glottal carrier to the given output device.
     * If [outputDevice] is null, the system default output is used.
     */
    fun start(outputDevice: AudioDeviceInfo? = null) {
        if (isRunning.getAndSet(true)) return

        val attrs = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION)
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
            .build()

        val format = AudioFormat.Builder()
            .setSampleRate(SAMPLE_RATE)
            .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
            .build()

        val minBuf = AudioTrack.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT
        )
        val bufSize = maxOf(minBuf, BUFFER_FRAMES * 4)

        val track = AudioTrack.Builder()
            .setAudioAttributes(attrs)
            .setAudioFormat(format)
            .setBufferSizeInBytes(bufSize)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()

        if (outputDevice != null) {
            track.setPreferredDevice(outputDevice)
        }

        audioTrack = track
        phase = 0f

        track.play()
        Log.i(TAG, "Carrier started: F0=${f0} Hz, device=${outputDevice?.productName ?: "default"}")

        generatorJob = CoroutineScope(Dispatchers.IO).launch {
            val buffer = FloatArray(BUFFER_FRAMES)
            try {
                while (isActive && isRunning.get()) {
                    generateBuffer(buffer)
                    track.write(buffer, 0, buffer.size, AudioTrack.WRITE_BLOCKING)
                }
            } finally {
                track.stop()
                track.release()
                Log.i(TAG, "Carrier stopped")
            }
        }
    }

    fun stop() {
        if (!isRunning.getAndSet(false)) return
        generatorJob?.cancel()
        generatorJob = null
        audioTrack = null
    }

    /** Updates the fundamental frequency in real time. Clamped to 80-200 Hz. */
    fun setF0(hz: Float) {
        f0 = hz.coerceIn(80f, 200f)
    }

    /**
     * Fills [buffer] with Rosenberg-model glottal pulse samples.
     *
     * The Rosenberg model produces a half-sine during the "open phase" (40% of
     * the glottal period) and silence during the "closed phase" (60%). This
     * yields a rich harmonic spectrum similar to natural vocal fold vibration.
     *
     * Phase is maintained across calls for click-free output.
     */
    internal fun generateBuffer(buffer: FloatArray) {
        val currentF0 = f0
        val period = (SAMPLE_RATE / currentF0).coerceAtLeast(1f)
        val openPhase = period * 0.4f

        for (i in buffer.indices) {
            val phaseInPeriod = phase % period
            buffer[i] = if (phaseInPeriod < openPhase) {
                (sin(PI * phaseInPeriod / openPhase).toFloat()) * AMPLITUDE
            } else {
                0f
            }
            phase += 1f
            // Prevent float drift by wrapping at period boundary
            if (phase >= period) phase -= period
        }
    }
}
