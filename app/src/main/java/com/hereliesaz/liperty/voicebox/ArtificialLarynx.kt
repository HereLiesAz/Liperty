package com.hereliesaz.liperty.voicebox

import android.media.AudioDeviceInfo

/**
 * Artificial Larynx: Generates a continuous glottal pulse carrier signal
 * and streams it to bone conduction headphones worn around the neck.
 *
 * The carrier travels through the throat tissue; the user's mouth and tongue
 * modulate it into speech-like sounds that the phone microphone captures.
 */
class ArtificialLarynx(private val audioRouter: AudioRouter) {

    private val carrierGenerator = GlottalCarrierGenerator()

    /**
     * Starts the carrier signal, routing output to the best available
     * bone conduction / Bluetooth device.
     */
    fun start() {
        carrierGenerator.start(audioRouter.getOptimalOutputDevice())
    }

    fun stop() {
        carrierGenerator.stop()
    }

    /** Adjusts the fundamental frequency of the carrier (80-200 Hz). */
    fun setF0(hz: Float) {
        carrierGenerator.setF0(hz)
    }

    /** Current fundamental frequency. */
    val f0: Float get() = carrierGenerator.f0
}
