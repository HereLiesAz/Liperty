package com.hereliesaz.liperty.voicebox

import android.annotation.SuppressLint
import android.bluetooth.BluetoothManager
import android.content.Context
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.os.Build
import androidx.annotation.RequiresApi

object BluetoothLEAudioManager {

    private var previousMode: Int? = null

    /**
     * Attempts to start an Isochronous Stream (ISOC) for Bluetooth LE Audio using the LC3 Codec.
     * This is crucial for transmitting high-fidelity data with ultra-low latency (10-15ms).
     *
     * Note: Full LE Audio support requires Android 13 (API 33) or higher.
     */
    @SuppressLint("MissingPermission")
    @RequiresApi(Build.VERSION_CODES.TIRAMISU)
    fun startIsochronousStream(context: Context): Boolean {
        val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val bluetoothManager = context.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
        val bluetoothAdapter = bluetoothManager.adapter

        if (bluetoothAdapter == null || !bluetoothAdapter.isEnabled) {
            return false
        }

        // 1. Check for LE Audio capability
        if (bluetoothAdapter.isLeAudioSupported != android.bluetooth.BluetoothStatusCodes.FEATURE_SUPPORTED) {
            return false
        }

        // 2. Configure AudioManager for Communication (low latency)
        if (previousMode == null) {
            previousMode = audioManager.mode
        }
        audioManager.mode = AudioManager.MODE_IN_COMMUNICATION

        // 3. Route audio to BLE Headset if connected
        val devices = audioManager.getDevices(AudioManager.GET_DEVICES_OUTPUTS)
        for (device in devices) {
            if (device.type == AudioDeviceInfo.TYPE_BLE_HEADSET ||
                device.type == AudioDeviceInfo.TYPE_BLE_SPEAKER ||
                device.type == AudioDeviceInfo.TYPE_BLE_BROADCAST) {

                val result = audioManager.setCommunicationDevice(device)
                if (result) {
                    // Successfully routed to LE Audio device. The OS will automatically
                    // negotiate the LC3 codec and ISOC channels.
                    return true
                }
            }
        }

        return false
    }

    /**
     * Helper to stop the communication routing.
     */
    @RequiresApi(Build.VERSION_CODES.S)
    fun stopIsochronousStream(context: Context) {
        val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        audioManager.clearCommunicationDevice()
        previousMode?.let { audioManager.mode = it }
        previousMode = null
    }
}
