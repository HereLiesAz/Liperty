package com.hereliesaz.liperty.voicebox.spike

import android.app.Application
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlin.math.abs
import kotlin.math.sqrt

data class SpikeUiState(
    val isRecording: Boolean = false,
    val statusText: String = "Press Record to begin",
    val accelMagnitude: Float = 0f,
    val resultLines: List<String> = emptyList(),
    val errorText: String = ""
)

class SpikeViewModel(app: Application) : AndroidViewModel(app) {

    private val _uiState = MutableStateFlow(SpikeUiState())
    val uiState: StateFlow<SpikeUiState> = _uiState.asStateFlow()

    private val recorder = SignalRecorder(app)

    private val sensorManager = app.getSystemService(SensorManager::class.java)
    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

    // Raw magnitude written by sensor thread, polled by UI
    @Volatile private var rawMagnitude = 0f

    private val accelListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            val x = event.values[0]; val y = event.values[1]; val z = event.values[2]
            rawMagnitude = abs(sqrt(x * x + y * y + z * z) - SensorManager.GRAVITY_EARTH)
        }
        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
    }

    init {
        // Register sensor immediately so the placement indicator is live as soon
        // as the screen opens — not just during recording.
        sensorManager.registerListener(
            accelListener, accelerometer, SensorManager.SENSOR_DELAY_FASTEST
        )
        // Poll loop: update UI state at 20 Hz
        viewModelScope.launch {
            while (true) {
                _uiState.value = _uiState.value.copy(accelMagnitude = rawMagnitude)
                delay(50L)
            }
        }
    }

    fun startRecording() {
        if (_uiState.value.isRecording) return
        _uiState.value = _uiState.value.copy(
            isRecording  = true,
            statusText   = "Recording 10 seconds…",
            resultLines  = emptyList(),
            errorText    = ""
        )
        recorder.record(
            durationMs = 10_000L,
            onComplete = { result ->
                _uiState.value = _uiState.value.copy(
                    isRecording = false,
                    statusText  = "Done.",
                    resultLines = listOf(
                        "✅ Recording complete",
                        "",
                        "Audio:  ${result.audioFile.name}",
                        "        ${result.audioSamples} samples @ 16 kHz",
                        "",
                        "Accel:  ${result.accelFile.name}",
                        "        ${result.accelSamples} samples",
                        "",
                        "Duration: ${result.durationMs} ms",
                        "",
                        "Pull with:",
                        "adb pull /sdcard/Android/data/",
                        "com.HereLiesAz.Liperty/files/"
                    )
                )
            },
            onError = { msg ->
                _uiState.value = _uiState.value.copy(
                    isRecording = false,
                    statusText  = "Ready",
                    errorText   = "Error: $msg"
                )
            }
        )
    }

    fun stopRecording() {
        recorder.stop()
        _uiState.value = _uiState.value.copy(
            isRecording = false,
            statusText  = "Stopped early."
        )
    }

    override fun onCleared() {
        super.onCleared()
        sensorManager.unregisterListener(accelListener)
        recorder.stop()
    }
}