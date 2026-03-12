package com.hereliesaz.liperty.ui

import android.content.Context
import android.os.Bundle
import android.widget.SeekBar
import android.widget.Switch
import androidx.appcompat.app.AppCompatActivity
import com.hereliesaz.liperty.R

class SettingsActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)

        val switchTelephoto = findViewById<Switch>(R.id.switch_telephoto)

        val sharedPrefs = getSharedPreferences("LipertyPrefs", Context.MODE_PRIVATE)

        // Load saved preferences
        val savedTelephoto = sharedPrefs.getBoolean("telephoto_preference", true)

        switchTelephoto.isChecked = savedTelephoto

        switchTelephoto.setOnCheckedChangeListener { _, isChecked ->
            sharedPrefs.edit().putBoolean("telephoto_preference", isChecked).apply()
        }
    }
}
