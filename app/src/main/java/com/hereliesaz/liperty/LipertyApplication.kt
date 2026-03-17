package com.hereliesaz.liperty

import android.app.Application
import com.hereliesaz.liperty.voicebox.BluetoothLEAudioManager

class LipertyApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        BluetoothLEAudioManager.initialize(this)
    }
}
