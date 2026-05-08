package com.hereliesaz.liperty.voicebox

import android.content.Context
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.os.Build
import android.util.Log

/**
 * Manages audio input and output routing for the Voice Box pipeline.
 *
 * Input: prioritizes BCMs and head-worn sensors for capture.
 * Output: routes carrier signals to bone conduction headphones.
 * Full-duplex: configures simultaneous input + output for BC larynx mode.
 */
class AudioRouter(private val context: Context) {

    private val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    private var previousMode: Int? = null

    companion object {
        private const val TAG = "AudioRouter"
    }

    // ── Input routing ────────────────────────────────────────────────────

    /**
     * Determines the optimal audio input device.
     * Prioritizes Bluetooth LE Audio (LC3) and head-worn sensors.
     */
    fun getOptimalInputDevice(): AudioDeviceInfo? {
        val devices = audioManager.getDevices(AudioManager.GET_DEVICES_INPUTS)

        val bleDevice = devices.find { it.type == AudioDeviceInfo.TYPE_BLE_HEADSET }
        if (bleDevice != null) {
            Log.i(TAG, "Input: BLE Headset (LC3)")
            return bleDevice
        }

        val bluetoothDevice = devices.find { it.type == AudioDeviceInfo.TYPE_BLUETOOTH_SCO }
        if (bluetoothDevice != null) {
            Log.i(TAG, "Input: Bluetooth SCO")
            return bluetoothDevice
        }

        val wiredDevice = devices.find { it.type == AudioDeviceInfo.TYPE_WIRED_HEADSET }
        if (wiredDevice != null) {
            Log.i(TAG, "Input: Wired Headset")
            return wiredDevice
        }

        Log.i(TAG, "Input: Built-in Mic (fallback)")
        return devices.find { it.type == AudioDeviceInfo.TYPE_BUILTIN_MIC }
    }

    /**
     * Returns the phone's built-in microphone explicitly.
     *
     * In BC larynx mode we need the phone mic (captures acoustically modulated
     * sound from the user's mouth), NOT the headphone mic. When BC headphones
     * are connected, the system may default to the headphone mic — this method
     * overrides that.
     */
    fun getPhoneMic(): AudioDeviceInfo? {
        val devices = audioManager.getDevices(AudioManager.GET_DEVICES_INPUTS)
        return devices.find { it.type == AudioDeviceInfo.TYPE_BUILTIN_MIC }
    }

    /**
     * Checks if the current routing is using a high-fidelity BCM path.
     */
    fun isUsingBoneConduction(): Boolean {
        val currentDevice = getOptimalInputDevice()
        return currentDevice?.let {
            it.type == AudioDeviceInfo.TYPE_BLE_HEADSET ||
            it.type == AudioDeviceInfo.TYPE_BLUETOOTH_SCO ||
            it.type == AudioDeviceInfo.TYPE_WIRED_HEADSET
        } ?: false
    }

    // ── Output routing ───────────────────────────────────────────────────

    /**
     * Returns the best output device for the BC carrier signal.
     * Priority: BLE headset > Bluetooth SCO > Wired headset > Built-in speaker.
     */
    fun getOptimalOutputDevice(): AudioDeviceInfo? {
        val devices = audioManager.getDevices(AudioManager.GET_DEVICES_OUTPUTS)

        val bleDevice = devices.find { it.type == AudioDeviceInfo.TYPE_BLE_HEADSET }
        if (bleDevice != null) {
            Log.i(TAG, "Output: BLE Headset")
            return bleDevice
        }

        val bleSpk = devices.find { it.type == AudioDeviceInfo.TYPE_BLE_SPEAKER }
        if (bleSpk != null) {
            Log.i(TAG, "Output: BLE Speaker")
            return bleSpk
        }

        val scoDevice = devices.find { it.type == AudioDeviceInfo.TYPE_BLUETOOTH_SCO }
        if (scoDevice != null) {
            Log.i(TAG, "Output: Bluetooth SCO")
            return scoDevice
        }

        val wiredDevice = devices.find { it.type == AudioDeviceInfo.TYPE_WIRED_HEADSET }
        if (wiredDevice != null) {
            Log.i(TAG, "Output: Wired Headset")
            return wiredDevice
        }

        Log.i(TAG, "Output: Built-in Speaker (fallback)")
        return devices.find { it.type == AudioDeviceInfo.TYPE_BUILTIN_SPEAKER }
    }

    // ── Full-duplex configuration ────────────────────────────────────────

    /**
     * Configures the audio system for full-duplex operation (simultaneous
     * output to BC headphones and input from phone mic).
     *
     * Sets [AudioManager.MODE_IN_COMMUNICATION] which enables low-latency
     * bidirectional audio. On API 31+ also sets the communication device
     * to the BC headphone output.
     */
    fun configureFullDuplex() {
        if (previousMode == null) {
            previousMode = audioManager.mode
        }
        audioManager.mode = AudioManager.MODE_IN_COMMUNICATION

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val outputDevice = getOptimalOutputDevice()
            if (outputDevice != null) {
                val success = audioManager.setCommunicationDevice(outputDevice)
                Log.i(TAG, "Full-duplex: setCommunicationDevice(${outputDevice.productName}) = $success")
            }
        }

        Log.i(TAG, "Full-duplex configured (MODE_IN_COMMUNICATION)")
    }

    /**
     * Restores the audio mode to whatever it was before [configureFullDuplex].
     */
    fun releaseFullDuplex() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            audioManager.clearCommunicationDevice()
        }
        previousMode?.let { audioManager.mode = it }
        previousMode = null
        Log.i(TAG, "Full-duplex released")
    }
}
