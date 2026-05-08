package com.hereliesaz.liperty.voicebox

import android.app.Application
import android.net.Uri
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.hereliesaz.liperty.voicebox.cloning.AudioPreprocessor
import com.hereliesaz.liperty.voicebox.cloning.SpeakerClusterer
import com.hereliesaz.liperty.voicebox.cloning.VoiceProfile
import com.hereliesaz.liperty.voicebox.cloning.VoiceProfileBuilder
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
    val profiles: List<VoiceProfile> = emptyList(),
    val activeVoiceName: String? = null,
    val isRecording: Boolean = false,
    val isCloning: Boolean = false,
    val isNamingVoice: Boolean = false,
    val statusMessage: String = "Ready"
)

enum class ImportStep {
    IDLE, EXTRACTING, SEGMENTING, CLUSTERING, SPEAKER_SELECTION, READY, SAVING, COMPLETE, ERROR
}

data class ImportState(
    val step: ImportStep = ImportStep.IDLE,
    val progressPercent: Float = 0f,
    val progressMessage: String = "",
    val segments: List<AudioPreprocessor.SpeechSegment> = emptyList(),
    val clusters: List<SpeakerClusterer.SpeakerCluster> = emptyList(),
    val selectedClusterId: Int? = null,
    val qualityScore: Float = 0f,
    val totalSpeechDurationMs: Long = 0,
    val sampleCount: Int = 0,
    /** Per-segment embeddings for the selected speaker's segments. */
    val selectedEmbeddings: List<FloatArray> = emptyList(),
    val error: String? = null
)

class VoiceViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "VoiceViewModel"
    }

    private val voiceStore = VoiceStore(application)
    private val voiceRecorder = VoiceRecorder(application)
    private val pocketTts = PocketTTSEngine(application)
    private val audioPreprocessor = AudioPreprocessor(application)
    private val speakerClusterer = SpeakerClusterer()
    private val profileBuilder = VoiceProfileBuilder()

    private val _uiState = MutableStateFlow(VoiceUiState())
    val uiState: StateFlow<VoiceUiState> = _uiState.asStateFlow()

    private val _importState = MutableStateFlow(ImportState())
    val importState: StateFlow<ImportState> = _importState.asStateFlow()

    private var pendingCloningUris: List<Uri> = emptyList()
    private var pendingRecordedFile: File? = null

    init {
        loadVoices()
        pocketTts.initialize()
    }

    fun loadVoices() {
        val allVoices = voiceStore.loadAllVoices()
        val allProfiles = voiceStore.loadAllProfiles()
        val sharedPrefs = getApplication<Application>().getSharedPreferences("LipertyPrefs", android.content.Context.MODE_PRIVATE)
        val activeName = sharedPrefs.getString("active_voice", null)
        _uiState.value = _uiState.value.copy(
            voices = allVoices,
            profiles = allProfiles,
            activeVoiceName = activeName
        )
    }

    // ── Import Wizard Pipeline ──────────────────────────────────────────

    /**
     * Starts the full import processing pipeline for selected URIs.
     * Steps: extract audio → VAD → segment → extract embeddings → cluster speakers.
     */
    fun startImportProcessing(uris: List<Uri>) {
        viewModelScope.launch {
            try {
                _importState.value = ImportState(
                    step = ImportStep.EXTRACTING,
                    progressMessage = "Extracting audio..."
                )

                // 1. Process each URI through AudioPreprocessor
                val allSegments = mutableListOf<AudioPreprocessor.SpeechSegment>()
                for ((index, uri) in uris.withIndex()) {
                    _importState.value = _importState.value.copy(
                        progressPercent = index.toFloat() / uris.size,
                        progressMessage = "Processing file ${index + 1} of ${uris.size}..."
                    )
                    val result = withContext(Dispatchers.IO) {
                        audioPreprocessor.processUri(uri, uri.lastPathSegment ?: "file_$index")
                    }
                    allSegments.addAll(result.segments)
                }

                if (allSegments.isEmpty()) {
                    _importState.value = ImportState(
                        step = ImportStep.ERROR,
                        error = "No speech detected in the selected files. Try different recordings."
                    )
                    return@launch
                }

                // 2. Extract embeddings for each segment
                _importState.value = _importState.value.copy(
                    step = ImportStep.CLUSTERING,
                    progressMessage = "Analyzing speakers (${allSegments.size} segments)..."
                )

                val embeddings = withContext(Dispatchers.IO) {
                    allSegments.map { segment ->
                        pocketTts.extractEmbeddingFromPcm(segment.pcmData)
                    }
                }

                // 3. Cluster by speaker
                val clusterResult = speakerClusterer.cluster(allSegments, embeddings)

                if (clusterResult.singleSpeaker) {
                    // Skip speaker selection — go straight to profile config
                    val cluster = clusterResult.clusters.first()
                    val quality = profileBuilder.computeQualityScore(
                        cluster.segments.zip(cluster.embeddings).map { (seg, emb) ->
                            com.hereliesaz.liperty.voicebox.cloning.VoiceSample(
                                sourceFileName = seg.sourceFileName,
                                durationMs = seg.durationMs,
                                embedding = emb,
                                snrEstimate = seg.snrEstimate
                            )
                        }
                    )

                    _importState.value = ImportState(
                        step = ImportStep.READY,
                        segments = cluster.segments,
                        clusters = clusterResult.clusters,
                        selectedClusterId = cluster.clusterId,
                        selectedEmbeddings = cluster.embeddings,
                        sampleCount = cluster.segments.size,
                        totalSpeechDurationMs = cluster.totalSpeechDurationMs,
                        qualityScore = quality
                    )
                } else {
                    // Need speaker identification
                    _importState.value = ImportState(
                        step = ImportStep.SPEAKER_SELECTION,
                        segments = allSegments,
                        clusters = clusterResult.clusters,
                        sampleCount = allSegments.size,
                        totalSpeechDurationMs = allSegments.sumOf { it.durationMs }
                    )
                }

            } catch (e: Exception) {
                Log.e(TAG, "Import processing failed", e)
                _importState.value = ImportState(
                    step = ImportStep.ERROR,
                    error = "Processing failed: ${e.message}"
                )
            }
        }
    }

    /**
     * User selects which speaker cluster is theirs (step 3 of wizard).
     */
    fun selectSpeaker(clusterId: Int) {
        _importState.value = _importState.value.copy(selectedClusterId = clusterId)
    }

    /**
     * Confirms the speaker selection and advances to profile configuration.
     */
    fun confirmSpeakerSelection() {
        val state = _importState.value
        val cluster = state.clusters.find { it.clusterId == state.selectedClusterId } ?: return

        val quality = profileBuilder.computeQualityScore(
            cluster.segments.zip(cluster.embeddings).map { (seg, emb) ->
                com.hereliesaz.liperty.voicebox.cloning.VoiceSample(
                    sourceFileName = seg.sourceFileName,
                    durationMs = seg.durationMs,
                    embedding = emb,
                    snrEstimate = seg.snrEstimate
                )
            }
        )

        _importState.value = state.copy(
            step = ImportStep.READY,
            segments = cluster.segments,
            selectedEmbeddings = cluster.embeddings,
            sampleCount = cluster.segments.size,
            totalSpeechDurationMs = cluster.totalSpeechDurationMs,
            qualityScore = quality
        )
    }

    /**
     * Plays a speech segment for speaker identification preview.
     */
    fun playSegmentPreview(segment: AudioPreprocessor.SpeechSegment) {
        viewModelScope.launch {
            try {
                val voiceManager = VoiceManager(getApplication()) {}
                voiceManager.playAudio(segment.pcmData)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to play segment preview", e)
            }
        }
    }

    /**
     * Previews the cloned voice by synthesizing a test phrase.
     */
    fun previewVoice() {
        viewModelScope.launch {
            try {
                val state = _importState.value
                val cluster = state.clusters.find { it.clusterId == state.selectedClusterId } ?: return@launch
                val testVoice = VoiceState("preview", cluster.centroidEmbedding)

                val audio = withContext(Dispatchers.IO) {
                    pocketTts.generateAudio("Hello, this is my voice.", testVoice)
                }
                if (audio != null) {
                    val voiceManager = VoiceManager(getApplication()) {}
                    voiceManager.playAudio(audio)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Voice preview failed", e)
            }
        }
    }

    /**
     * Saves the imported profile (new or add to existing).
     *
     * @param name Profile name (used for new profiles, or the existing name when adding).
     * @param existingProfileName If non-null, adds samples to this existing profile.
     */
    fun saveImportedProfile(name: String, existingProfileName: String?) {
        viewModelScope.launch {
            _importState.value = _importState.value.copy(step = ImportStep.SAVING)

            try {
                val state = _importState.value
                val segments = state.segments
                val embeddings = state.selectedEmbeddings

                if (segments.isEmpty() || embeddings.isEmpty()) {
                    _importState.value = _importState.value.copy(
                        step = ImportStep.ERROR,
                        error = "No segments to save"
                    )
                    return@launch
                }

                withContext(Dispatchers.IO) {
                    if (existingProfileName != null) {
                        val existing = voiceStore.loadProfile(existingProfileName)
                        if (existing != null) {
                            val updated = profileBuilder.addSamples(existing, segments, embeddings)
                            voiceStore.saveProfile(updated)
                        } else {
                            // Existing profile not found — create new instead
                            val profile = profileBuilder.createProfile(name, segments, embeddings)
                            voiceStore.saveProfile(profile)
                        }
                    } else {
                        val profile = profileBuilder.createProfile(name, segments, embeddings)
                        voiceStore.saveProfile(profile)
                    }
                }

                loadVoices()
                _importState.value = ImportState(step = ImportStep.COMPLETE)

            } catch (e: Exception) {
                Log.e(TAG, "Failed to save profile", e)
                _importState.value = _importState.value.copy(
                    step = ImportStep.ERROR,
                    error = "Save failed: ${e.message}"
                )
            }
        }
    }

    /** Resets the import wizard to the initial state. */
    fun resetImport() {
        _importState.value = ImportState()
    }

    // ── Existing Recording Flow (kept for backward compat) ──────────────

    fun startRecording() {
        _uiState.value = _uiState.value.copy(isRecording = true, statusMessage = "Recording sample...")
        voiceRecorder.startRecording(
            durationMs = 5000L,
            onComplete = { file, _ ->
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
                            Log.e(TAG, "Failed to copy URI to temp file: $uri", e)
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
            selectVoice("")
        }
        loadVoices()
    }

    override fun onCleared() {
        super.onCleared()
        pocketTts.close()
    }
}
