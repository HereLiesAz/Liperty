package com.hereliesaz.liperty

import android.app.Application
import android.content.Context
import android.util.Log
import androidx.appcompat.app.AppCompatDelegate
import com.hereliesaz.liperty.voicebox.BluetoothLEAudioManager

class LipertyApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        // Honor the "dark_theme" pref process-wide so the XML-based
        // SettingsActivity (Theme.AppCompat.DayNight.NoActionBar) renders
        // with the same dark/light surface as the Compose MainActivity.
        // Without this, SettingsActivity would always pick the system
        // default and the in-app toggle would have no effect there.
        val sharedPrefs = getSharedPreferences("LipertyPrefs", Context.MODE_PRIVATE)
        val darkTheme = sharedPrefs.getBoolean("dark_theme", true)
        AppCompatDelegate.setDefaultNightMode(
            if (darkTheme) AppCompatDelegate.MODE_NIGHT_YES
            else AppCompatDelegate.MODE_NIGHT_NO
        )

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
