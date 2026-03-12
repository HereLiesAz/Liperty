package com.hereliesaz.liperty.voicebox

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.hereliesaz.liperty.voicebox.cloning.VoiceStore
import com.hereliesaz.liperty.voicebox.recording.VoiceRecorder
import androidx.core.content.edit
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

data class VoiceUiState(
    val voices: List<VoiceState> = emptyList(),
    val activeVoiceName: String? = null,
    val isRecording: Boolean = false,
    val isCloning: Boolean = false,
    val statusMessage: String = "Ready"
)

class VoiceViewModel(application: Application) : AndroidViewModel(application) {

    private val voiceStore = VoiceStore(application)
    private val voiceRecorder = VoiceRecorder(application)
    private val pocketTts = PocketTTSEngine(application)

    private val _uiState = MutableStateFlow(VoiceUiState())
    val uiState: StateFlow<VoiceUiState> = _uiState.asStateFlow()

    init {
        loadVoices()
        pocketTts.initialize()
    }

    fun loadVoices() {
        val allVoices = voiceStore.loadAllVoices()
        val sharedPrefs = getApplication<Application>().getSharedPreferences("LipertyPrefs", android.content.Context.MODE_PRIVATE)
        val activeName = sharedPrefs.getString("active_voice", null)
        _uiState.value = _uiState.value.copy(voices = allVoices, activeVoiceName = activeName)
    }

    fun startRecording() {
        _uiState.value = _uiState.value.copy(isRecording = true, statusMessage = "Recording sample...")
        voiceRecorder.startRecording(
            durationMs = 5000L,
            onComplete = { file, samples ->
                _uiState.value = _uiState.value.copy(isRecording = false, statusMessage = "Recording complete. Cloning...")
                cloneVoice("Voice_${System.currentTimeMillis()}", file)
            },
            onError = { msg ->
                _uiState.value = _uiState.value.copy(isRecording = false, statusMessage = "Error: $msg")
            }
        )
    }

    fun stopRecording() {
        voiceRecorder.stopRecording()
    }

    fun cloneFromUri(uri: Uri) {
        _uiState.value = _uiState.value.copy(isCloning = true, statusMessage = "Importing audio...")
        viewModelScope.launch {
            val tempFile = File(getApplication<Application>().cacheDir, "imported_voice_${System.currentTimeMillis()}.wav")
            val success = withContext(Dispatchers.IO) {
                try {
                    getApplication<Application>().contentResolver.openInputStream(uri)?.use { input ->
                        FileOutputStream(tempFile).use { output ->
                            input.copyTo(output)
                        }
                    }
                    true
                } catch (e: Exception) {
                    e.printStackTrace()
                    false
                }
            }

            if (success) {
                cloneVoice("Imported_${System.currentTimeMillis()}", tempFile)
            } else {
                _uiState.value = _uiState.value.copy(isCloning = false, statusMessage = "Import failed")
            }
        }
    }

    private fun cloneVoice(name: String, audioFile: File) {
        _uiState.value = _uiState.value.copy(isCloning = true)
        viewModelScope.launch {
            val voiceState = pocketTts.cloneVoice(audioFile)
            val namedState = voiceState.copy(name = name)
            voiceStore.saveVoice(namedState)
            loadVoices()
            _uiState.value = _uiState.value.copy(isCloning = false, statusMessage = "Voice cloned: $name")
        }
    }

    fun selectVoice(name: String) {
        val sharedPrefs = getApplication<Application>().getSharedPreferences("LipertyPrefs", android.content.Context.MODE_PRIVATE)
        sharedPrefs.edit { putString("active_voice", name) }
        _uiState.value = _uiState.value.copy(activeVoiceName = name)
    }

    fun deleteVoice(name: String) {
        voiceStore.deleteVoice(name)
        if (_uiState.value.activeVoiceName == name) {
            selectVoice("") // Clear active if deleted
        }
        loadVoices()
    }

    override fun onCleared() {
        super.onCleared()
        pocketTts.close()
    }
}
