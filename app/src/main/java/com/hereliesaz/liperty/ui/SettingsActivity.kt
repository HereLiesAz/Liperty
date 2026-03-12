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

        val seekBarFontSize = findViewById<SeekBar>(R.id.seekbar_font_size)
        val switchTelephoto = findViewById<Switch>(R.id.switch_telephoto)

        val sharedPrefs = getSharedPreferences("LipertyPrefs", Context.MODE_PRIVATE)

        // Load saved preferences
        val savedFontSize = sharedPrefs.getInt("font_size", 20)
        val savedTelephoto = sharedPrefs.getBoolean("telephoto_preference", true)

        seekBarFontSize.progress = savedFontSize
        switchTelephoto.isChecked = savedTelephoto

        // Save preferences on change
        seekBarFontSize.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                // Min value handling if API < 26 (though min is set in XML)
                val size = if (progress < 12) 12 else progress
                sharedPrefs.edit().putInt("font_size", size).apply()
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        switchTelephoto.setOnCheckedChangeListener { _, isChecked ->
            sharedPrefs.edit().putBoolean("telephoto_preference", isChecked).apply()
        }
    }
}
