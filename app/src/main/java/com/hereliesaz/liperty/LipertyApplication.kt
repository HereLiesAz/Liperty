package com.hereliesaz.liperty

import android.app.Application
import android.util.Log
import com.hereliesaz.liperty.voicebox.BluetoothLEAudioManager

class LipertyApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        // Anything that throws here force-closes the app before MainActivity
        // ever runs. BluetoothLEAudioManager.initialize already self-protects,
        // but wrap defensively too — a crash in the Application class is the
        // worst possible UX.
        try {
            BluetoothLEAudioManager.initialize(this)
        } catch (t: Throwable) {
            Log.e("LipertyApplication", "BLE Audio init threw — ignoring", t)
        }
    }
}
