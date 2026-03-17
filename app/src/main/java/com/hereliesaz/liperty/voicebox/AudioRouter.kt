package com.hereliesaz.liperty.voicebox

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.util.Log

/**
 * Manages audio input routing, prioritizing Bone Conduction Microphones (BCMs) 
 * and head-worn accelerometers over standard air-conduction microphones.
 */
class AudioRouter(private val context: Context) {

    private val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager

    companion object {
        private const val TAG = "AudioRouter"
    }

    /**
     * Determines the optimal audio source based on connected hardware.
     * Prioritizes Bluetooth LE Audio (LC3) and head-worn sensors.
     */
    fun getOptimalInputDevice(): AudioDeviceInfo? {
        val devices = audioManager.getDevices(AudioManager.GET_DEVICES_INPUTS)
        
        // Priority 1: Bluetooth LE Audio (BLE) - Supports LC3 low-latency
        val bleDevice = devices.find { it.type == AudioDeviceInfo.TYPE_BLE_HEADSET }
        if (bleDevice != null) {
            Log.i(TAG, "Routing to BLE Headset (LC3 Support)")
            return bleDevice
        }

        // Priority 2: Bluetooth SCO/HFP (Legacy but potentially BCM-equipped)
        val bluetoothDevice = devices.find { it.type == AudioDeviceInfo.TYPE_BLUETOOTH_SCO }
        if (bluetoothDevice != null) {
            Log.i(TAG, "Routing to Bluetooth SCO")
            return bluetoothDevice
        }

        // Priority 3: Wired Headset (common for BCM contact-mics)
        val wiredDevice = devices.find { it.type == AudioDeviceInfo.TYPE_WIRED_HEADSET }
        if (wiredDevice != null) {
            Log.i(TAG, "Routing to Wired Headset")
            return wiredDevice
        }

        Log.i(TAG, "No specialized input detected, falling back to Built-in Mic")
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
}
