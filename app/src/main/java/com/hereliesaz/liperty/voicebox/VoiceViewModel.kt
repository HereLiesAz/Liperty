package com.hereliesaz.liperty.voicebox

import android.app.Application
import android.net.Uri
import android.util.Log
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
    val isNamingVoice: Boolean = false,
    val statusMessage: String = "Ready"
)

class VoiceViewModel(application: Application) : AndroidViewModel(application) {

    private val voiceStore = VoiceStore(application)
    private val voiceRecorder = VoiceRecorder(application)
    private val pocketTts = PocketTTSEngine(application)

    private val _uiState = MutableStateFlow(VoiceUiState())
    val uiState: StateFlow<VoiceUiState> = _uiState.asStateFlow()

    private var pendingCloningUris: List<Uri> = emptyList()
    private var pendingRecordedFile: File? = null

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
                _uiState.value = _uiState.value.copy(isRecording = false, statusMessage = "Recording complete. Choose a name.")
                pendingRecordedFile = file
                _uiState.value = _uiState.value.copy(isNamingVoice = true)
            },
            onError = { msg ->
                _uiState.value = _uiState.value.copy(isRecording = false, statusMessage = "Error: $msg")
            }
        )
    }

    fun stopRecording() {
        voiceRecorder.stopRecording()
    }

    fun startCloningProcess(uris: List<Uri>) {
        if (uris.isEmpty()) return
        pendingCloningUris = uris
        _uiState.value = _uiState.value.copy(isNamingVoice = true)
    }

    fun cancelCloning() {
        pendingCloningUris = emptyList()
        pendingRecordedFile = null
        _uiState.value = _uiState.value.copy(isNamingVoice = false)
    }

    fun confirmVoiceName(name: String) {
        _uiState.value = _uiState.value.copy(isNamingVoice = false, isCloning = true, statusMessage = "Creating profile: $name...")
        
        viewModelScope.launch {
            if (pendingRecordedFile != null) {
                val voiceState = withContext(Dispatchers.IO) {
                    pocketTts.cloneVoice(name, listOf(pendingRecordedFile!!))
                }
                voiceStore.saveVoice(voiceState)
                pendingRecordedFile = null
            } else if (pendingCloningUris.isNotEmpty()) {
                val tempFiles = withContext(Dispatchers.IO) {
                    pendingCloningUris.mapNotNull { uri ->
                        val tempFile = File(getApplication<Application>().cacheDir, "imported_${System.currentTimeMillis()}_${uri.lastPathSegment}.wav")
                        try {
                            getApplication<Application>().contentResolver.openInputStream(uri)?.use { input ->
                                FileOutputStream(tempFile).use { output ->
                                    input.copyTo(output)
                                }
                            }
                            tempFile
                        } catch (e: Exception) {
                            Log.e("VoiceViewModel", "Failed to copy URI to temp file: $uri", e)
                            null
                        }
                    }
                }

                if (tempFiles.isNotEmpty()) {
                    val voiceState = withContext(Dispatchers.IO) {
                        pocketTts.cloneVoice(name, tempFiles)
                    }
                    voiceStore.saveVoice(voiceState)
                    withContext(Dispatchers.IO) {
                        tempFiles.forEach { it.delete() }
                    }
                }
                pendingCloningUris = emptyList()
            }
            
            loadVoices()
            _uiState.value = _uiState.value.copy(isCloning = false, statusMessage = "Voice profile created: $name")
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
