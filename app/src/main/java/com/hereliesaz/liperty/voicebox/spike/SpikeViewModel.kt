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

import com.hereliesaz.liperty.voicebox.LaryngealSensor

data class SpikeUiState(
    val isRecording: Boolean = false,
    val isLiveListening: Boolean = false,
    val statusText: String = "Press Record to begin",
    val accelMagnitude: Float = 0f,
    val audioLevel: Float = 0f, // 0..1 for VU meter
    val isVoicing: Boolean = false,
    val resultLines: List<String> = emptyList(),
    val errorText: String = ""
)

class SpikeViewModel(app: Application) : AndroidViewModel(app) {

    private val _uiState = MutableStateFlow(SpikeUiState())
    val uiState: StateFlow<SpikeUiState> = _uiState.asStateFlow()

    private val recorder = SignalRecorder(app)
    private val sensor   = LaryngealSensor(app)

    private val sensorManager = app.getSystemService(SensorManager::class.java)
    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

    // Raw values written by background threads, polled by UI
    @Volatile private var rawMagnitude = 0f
    @Volatile private var rawAudioLevel = 0f
    @Volatile private var rawVoicing = false

    private val accelListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            val x = event.values[0]; val y = event.values[1]; val z = event.values[2]
            rawMagnitude = abs(sqrt(x * x + y * y + z * z) - SensorManager.GRAVITY_EARTH)
        }
        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
    }

    init {
        sensorManager.registerListener(
            accelListener, accelerometer, SensorManager.SENSOR_DELAY_FASTEST
        )
        // Poll loop: update UI state at 20 Hz
        viewModelScope.launch {
            while (true) {
                _uiState.value = _uiState.value.copy(
                    accelMagnitude = rawMagnitude,
                    audioLevel     = rawAudioLevel,
                    isVoicing      = rawVoicing
                )
                delay(50L)
            }
        }
    }

    fun toggleLiveListen() {
        val currentlyListening = _uiState.value.isLiveListening
        if (currentlyListening) {
            sensor.stop()
            _uiState.value = _uiState.value.copy(isLiveListening = false, statusText = "Live Listen Stopped")
            rawAudioLevel = 0f
            rawVoicing = false
        } else {
            _uiState.value = _uiState.value.copy(isLiveListening = true, statusText = "Listening real-time...")
            sensor.start(
                onProcessedAudio = { samples ->
                    // Calculate RMS for VU meter
                    var sum = 0f
                    for (s in samples) sum += s * s
                    val rms = sqrt(sum / samples.size)
                    // Scale for UI (0.0 to 1.0)
                    rawAudioLevel = (rms * 10f).coerceIn(0f, 1f)
                },
                onVoicingState = { voicing ->
                    rawVoicing = voicing
                },
                onVibrationData = { /* Not used in Spike view */ }
            )
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
                        "com.hereliesaz.liperty/files/"
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