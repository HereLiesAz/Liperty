package com.hereliesaz.liperty.ml

import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import androidx.core.content.edit

data class CalibrationUiState(
    val currentPhrase: String = "",
    val progress: Float = 0f,
    val isRecording: Boolean = false,
    val isTraining: Boolean = false,
    val calibrationDone: Boolean = false,
    val statusMessage: String = "Align your lips and press start"
)



class CalibrationViewModel(private val app: Application) : AndroidViewModel(app) {

    private val manager = CalibrationManager(app)
    
    private val _uiState = MutableStateFlow(CalibrationUiState())
    val uiState: StateFlow<CalibrationUiState> = _uiState.asStateFlow()

    init {
        manager.initialize()
        _uiState.value = _uiState.value.copy(currentPhrase = manager.getCurrentPhrase())
    }

    fun onFrameCaptured(bitmap: Bitmap) {
        if (_uiState.value.isRecording) {
            manager.addFrame(bitmap)
        }
    }

    fun startRecording() {
        _uiState.value = _uiState.value.copy(
            isRecording = true,
            statusMessage = "Mouth the phrase clearly..."
        )
    }

    fun stopRecordingAndTrain() {
        _uiState.value = _uiState.value.copy(isRecording = false, isTraining = true, statusMessage = "Updating model...")
        
        viewModelScope.launch {
            manager.processCurrentPhrase(_uiState.value.currentPhrase)
            
            val isDone = manager.isCalibrationComplete()
            if (isDone) {
                val sharedPrefs = app.getSharedPreferences("LipertyPrefs", Context.MODE_PRIVATE)
                sharedPrefs.edit { putBoolean("calibration_complete", true) }
            }

            _uiState.value = _uiState.value.copy(
                isTraining = false,
                currentPhrase = manager.getCurrentPhrase(),
                progress = _uiState.value.progress + 0.25f,
                calibrationDone = isDone,
                statusMessage = if (isDone) "Calibration Complete!" else "Phrase saved. Next one?"
            )
        }
    }

    override fun onCleared() {
        super.onCleared()
        manager.close()
    }
}
