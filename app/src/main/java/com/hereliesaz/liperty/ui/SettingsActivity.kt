package com.hereliesaz.liperty.ui

import android.content.Context
import android.os.Bundle
import android.widget.Switch
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate
import com.hereliesaz.liperty.R

class SettingsActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)

        val switchTelephoto = findViewById<Switch>(R.id.switch_telephoto)
        val switchDarkTheme = findViewById<Switch>(R.id.switch_dark_theme)

        val sharedPrefs = getSharedPreferences("LipertyPrefs", Context.MODE_PRIVATE)

        // Load saved preferences
        val savedTelephoto = sharedPrefs.getBoolean("telephoto_preference", true)
        val savedDarkTheme = sharedPrefs.getBoolean("dark_theme", true)

        switchTelephoto.isChecked = savedTelephoto
        switchDarkTheme.isChecked = savedDarkTheme

        switchTelephoto.setOnCheckedChangeListener { _, isChecked ->
            sharedPrefs.edit().putBoolean("telephoto_preference", isChecked).apply()
        }

        switchDarkTheme.setOnCheckedChangeListener { _, isChecked ->
            sharedPrefs.edit().putBoolean("dark_theme", isChecked).apply()
            // Apply immediately so the surface flips without an app restart.
            AppCompatDelegate.setDefaultNightMode(
                if (isChecked) AppCompatDelegate.MODE_NIGHT_YES
                else AppCompatDelegate.MODE_NIGHT_NO
            )
        }
    }
}
