package com.hereliesaz.liperty.voicebox

import android.annotation.SuppressLint
import android.bluetooth.BluetoothManager
import android.content.Context
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.bluetooth.BluetoothLeAudio
import android.bluetooth.BluetoothProfile
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi

object BluetoothLEAudioManager {

    private const val TAG = "BLEAudioManager"
    private var leAudioProxy: BluetoothLeAudio? = null
    private var previousMode: Int? = null
    private var nativeLoaded = false

    init {
        try {
            System.loadLibrary("liperty_cv")
            nativeLoaded = true
        } catch (e: Throwable) {
            Log.e(TAG, "Failed to load native library 'liperty_cv'", e)
        }
    }

    private external fun initLC3(sampleRate: Int, frameDurationUs: Int)
    private external fun nativeEncodeLC3(pcm: ShortArray, outEncoded: ByteArray): Int
    private external fun nativeDecodeLC3(encoded: ByteArray, outPcm: ShortArray): Int

    fun encodeLC3(pcm: ShortArray, outEncoded: ByteArray): Int {
        if (!nativeLoaded) return -1
        return nativeEncodeLC3(pcm, outEncoded)
    }

    fun decodeLC3(encoded: ByteArray, outPcm: ShortArray): Int {
        if (!nativeLoaded) return -1
        return nativeDecodeLC3(encoded, outPcm)
    }

    /**
     * Initializes the LE Audio profile proxy.
     *
     * Called from [com.hereliesaz.liperty.LipertyApplication.onCreate], which
     * runs before any runtime permission has been granted on a fresh install.
     * `getProfileProxy(LE_AUDIO)` requires BLUETOOTH_CONNECT on API 31+ and
     * will throw SecurityException without it — taking the whole process down
     * before MainActivity ever runs. We swallow every Throwable here so the
     * absence of LE Audio routing degrades gracefully into "no LE Audio
     * routing" instead of a force-close.
     */
    @SuppressLint("MissingPermission")
    fun initialize(context: Context) {
        try {
            val bluetoothManager =
                context.getSystemService(Context.BLUETOOTH_SERVICE) as? BluetoothManager ?: return
            val adapter = bluetoothManager.adapter ?: return

            adapter.getProfileProxy(context, object : BluetoothProfile.ServiceListener {
                override fun onServiceConnected(profile: Int, proxy: BluetoothProfile) {
                    if (profile == BluetoothProfile.LE_AUDIO) {
                        leAudioProxy = proxy as BluetoothLeAudio
                        Log.i(TAG, "LE Audio Profile connected")
                    }
                }

                override fun onServiceDisconnected(profile: Int) {
                    if (profile == BluetoothProfile.LE_AUDIO) {
                        leAudioProxy = null
                        Log.i(TAG, "LE Audio Profile disconnected")
                    }
                }
            }, BluetoothProfile.LE_AUDIO)
        } catch (t: Throwable) {
            // SecurityException (no BLUETOOTH_CONNECT), IllegalArgumentException
            // (LE_AUDIO unsupported pre-API 33), NoClassDefFoundError on very
            // old devices — none of them should crash the app.
            Log.w(TAG, "LE Audio proxy init failed; continuing without LE Audio routing", t)
        }
    }

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
            Log.w(TAG, "LE Audio not supported by hardware/adapter")
            return false
        }

        // 2. Configure AudioManager for Communication (low latency)
        if (previousMode == null) {
            previousMode = audioManager.mode
        }
        audioManager.mode = AudioManager.MODE_IN_COMMUNICATION

        // 3. Prefer devices from the LeAudio proxy if available
        leAudioProxy?.connectedDevices?.forEach { device ->
            Log.i(TAG, "Found connected LE Audio device: ${device.name}")
            // In API 33+, the OS handles the routing when we select the BLE_HEADSET TYPE
        }

        // 4. Route audio to BLE Headset if connected
        val devices = audioManager.getDevices(AudioManager.GET_DEVICES_OUTPUTS)
        for (device in devices) {
            if (device.type == AudioDeviceInfo.TYPE_BLE_HEADSET ||
                device.type == AudioDeviceInfo.TYPE_BLE_SPEAKER) {

                Log.i(TAG, "Setting communication device to: ${device.productName}")
                val result = audioManager.setCommunicationDevice(device)
                if (result) {
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
