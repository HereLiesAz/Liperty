package com.hereliesaz.liperty.voicebox

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Artificial Larynx: Generates continuous vibration to act as a sound source.
 *
 * Mute users press the back of the phone against their throat. The phone's
 * vibration motor (LRA) creates a carrier signal that the user's mouth
 * and tongue modulate into speech sounds.
 */
class ArtificialLarynx(context: Context) {

    private val vibrator: Vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        val vibratorManager = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
        vibratorManager.defaultVibrator
    } else {
        @Suppress("DEPRECATION")
        context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
    }

    private val isActive = AtomicBoolean(false)

    /**
     * Starts continuous vibration at a specific frequency/intensity.
     * Note: Most standard Android LRAs are resonant around 160-230Hz.
     */
    fun start() {
        if (isActive.getAndSet(true)) return

        // Create a repeating waveform for continuous carrier signal.
        // timings: [off_duration, on_duration]
        // amplitudes: [off_intensity, on_intensity]
        // repeat: 0 (index to start repeating from)
        // Using a single long 'on' period for a steady buzz.
        val effect = VibrationEffect.createWaveform(
            longArrayOf(0, 5000), 
            intArrayOf(0, 255), 
            0
        )

        vibrator.vibrate(effect)
    }

    fun stop() {
        if (!isActive.getAndSet(false)) return
        vibrator.cancel()
    }
}
