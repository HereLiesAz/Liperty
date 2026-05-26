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
import com.hereliesaz.liperty.voicebox.cloning.VoiceQualityTier
import com.hereliesaz.liperty.voicebox.cloning.VoiceStore
import com.hereliesaz.liperty.voicebox.recording.VoiceRecorder
import com.hereliesaz.liperty.personalization.PairedImportProcessor
import com.hereliesaz.liperty.personalization.PairedTrainingStore
import com.hereliesaz.liperty.personalization.PersonalizationConsentManager
import com.hereliesaz.liperty.personalization.VideoFrameExtractor
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

/**
 * UI state for the coached multi-clip recording session
 * ([VoiceViewModel.startCoachedSession]). Distinct from the
 * import wizard's [ImportState] — this is the in-app sequential
 * recording flow, not file import.
 */
data class CoachState(
    /** Number of clips captured so far this session. */
    val clipCount: Int = 0,
    /** Cumulative speech duration (rounded down to whole seconds). */
    val totalSpeechSeconds: Int = 0,
    /** Discrete quality band, recomputed after every clip. */
    val qualityTier: VoiceQualityTier = VoiceQualityTier.NONE,
    /**
     * Pointer into the Harvard-sentence prompt list. Advances on
     * successful capture (`recomputeCoachQualityState(advancePrompt = true)`)
     * but stays put on discard, so retrying lands the user back on
     * the same prompt they just attempted. Decoupling this from
     * [clipCount] is what prevents the prompt from "jumping" when
     * the user discards a bad clip.
     */
    val promptIndex: Int = 0,
    /** True while the [VoiceRecorder] is actively capturing. */
    val isRecording: Boolean = false,
    /** True between "Save" tap and HF/disk write completing. */
    val isSaving: Boolean = false,
    /** Set when the saved profile is written; UI can navigate away. */
    val isSaved: Boolean = false,
    /** Profile name as saved (after [VoiceViewModel.finishCoachedSession]). */
    val savedProfileName: String? = null,
    /** Last status line shown under the recorder. */
    val statusMessage: String = "Ready to record.",
    /** Non-null when the most recent operation errored. */
    val error: String? = null,
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

        /**
         * Default per-clip duration for the coached session. Five
         * seconds is the OpenVoice v2 sweet spot — enough to
         * capture timbre, short enough that users with progressive
         * conditions (ALS) don't fatigue across a 10–15 clip session.
         */
        const val DEFAULT_CLIP_DURATION_MS: Long = 5_000L
    }

    private val voiceStore = VoiceStore(application)
    private val voiceRecorder = VoiceRecorder(application)
    private val pocketTts = PocketTTSEngine(application)
    private val audioPreprocessor = AudioPreprocessor(application)
    private val speakerClusterer = SpeakerClusterer()
    private val profileBuilder = VoiceProfileBuilder()

    // Personalization harvesting from imported videos (opt-in, see
    // startImportProcessing). Roots at the same filesDir/viseme_calibration
    // store the calibration flow and Settings panel use.
    private val personalizationConsent = PersonalizationConsentManager(application)
    private val pairedImportProcessor = PairedImportProcessor(
        context = application,
        // Dedicated AudioPreprocessor instance: the harvest runs concurrently
        // with the voice-clone pipeline (which uses the shared one above), so we
        // don't share potentially non-thread-safe state across the two.
        audioPreprocessor = AudioPreprocessor(application),
        videoFrameExtractor = VideoFrameExtractor(application),
        store = PairedTrainingStore(File(application.filesDir, "viseme_calibration")),
        consent = personalizationConsent,
    )

    private val _uiState = MutableStateFlow(VoiceUiState())
    val uiState: StateFlow<VoiceUiState> = _uiState.asStateFlow()

    private val _importState = MutableStateFlow(ImportState())
    val importState: StateFlow<ImportState> = _importState.asStateFlow()

    private val _coachState = MutableStateFlow(CoachState())
    val coachState: StateFlow<CoachState> = _coachState.asStateFlow()

    private var pendingCloningUris: List<Uri> = emptyList()
    private var pendingRecordedFile: File? = null
    /** Captured WAV files for the active coached session (cacheDir). */
    private val coachedClips: MutableList<File> = mutableListOf()

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
        // Opportunistically harvest paired (audio, lip-video) personalization
        // data from the same imported videos — ONLY when the user has given the
        // separate on-device personalization consent. Runs independently on IO
        // so it never blocks or fails the voice-cloning pipeline; the consent
        // check is also enforced again inside processImport.
        if (personalizationConsent.hasConsent()) {
            viewModelScope.launch(Dispatchers.IO) {
                var saved = 0
                for (uri in uris) {
                    saved += try {
                        pairedImportProcessor.processImport(uri)
                    } catch (e: Exception) {
                        Log.w("VoiceViewModel", "personalization harvest failed for $uri: ${e.message}")
                        0
                    }
                }
                if (saved > 0) {
                    Log.i("VoiceViewModel", "personalization: harvested $saved paired record(s) from import")
                }
            }
        }

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

                if (!pocketTts.isReady) {
                    _importState.value = _importState.value.copy(
                        progressMessage = "Voice preview unavailable — TTS models not loaded"
                    )
                    return@launch
                }

                _importState.value = _importState.value.copy(progressMessage = "Generating preview...")
                val audio = withContext(Dispatchers.IO) {
                    pocketTts.generateAudio("Hello, this is my voice.", testVoice)
                }
                if (audio != null) {
                    _importState.value = _importState.value.copy(progressMessage = "")
                    val voiceManager = VoiceManager(getApplication()) {}
                    voiceManager.playAudio(audio)
                } else {
                    _importState.value = _importState.value.copy(
                        progressMessage = "Preview not available — using system voice instead"
                    )
                }
            } catch (e: Exception) {
                Log.e(TAG, "Voice preview failed", e)
                _importState.value = _importState.value.copy(
                    progressMessage = "Preview failed: ${e.message}"
                )
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

    // ── Coached Multi-Clip Recording ────────────────────────────────────

    /**
     * Begins (or resumes) a coached recording session. Clears any
     * clips left over from a prior session that wasn't saved.
     *
     * The coached session is the voice-banking flow targeted at
     * users with progressive conditions (ALS, pre-laryngectomy):
     * they record a sequence of short clips in one sitting and
     * watch the quality tier tick up as the embedding centroid
     * stabilizes.
     */
    fun startCoachedSession() {
        clearCoachedClipsFromDisk()
        coachedClips.clear()
        _coachState.value = CoachState()
    }

    /**
     * Records one clip of [durationMs] for the coached session.
     * On completion the WAV is appended to the in-memory list and
     * the quality tier is recomputed. Disk writes go to cacheDir
     * (no biometric data in user-visible storage).
     */
    fun recordCoachedClip(durationMs: Long = DEFAULT_CLIP_DURATION_MS) {
        if (_coachState.value.isRecording) return
        _coachState.value = _coachState.value.copy(
            isRecording = true,
            statusMessage = "Recording…"
        )
        voiceRecorder.startRecording(
            durationMs = durationMs,
            onComplete = { file, _ ->
                // VoiceRecorder writes 16 kHz mono. The WAV header
                // carries the rate, so PocketTTSEngine.readWavFile()
                // resamples to 24 kHz at extraction time and
                // wavDurationMs() reports the true duration.
                coachedClips.add(file)
                // Advance prompt only on successful capture; discard
                // leaves promptIndex alone so retrying lands on the
                // same sentence the user just attempted.
                recomputeCoachQualityState(
                    message = "Captured clip ${coachedClips.size}.",
                    advancePrompt = true,
                )
            },
            onError = { msg ->
                _coachState.value = _coachState.value.copy(
                    isRecording = false,
                    statusMessage = "Error: $msg"
                )
            }
        )
    }

    /**
     * User-initiated stop, e.g. they tapped the button mid-clip
     * to commit what they have so far.
     */
    fun stopCoachedRecording() {
        voiceRecorder.stopRecording()
    }

    /**
     * Drops the most recently captured clip. Useful when the user
     * coughs, the dog barks, or someone enters the room mid-clip.
     */
    fun discardLastCoachedClip() {
        if (coachedClips.isEmpty()) return
        val last = coachedClips.removeAt(coachedClips.size - 1)
        last.delete()
        recomputeCoachQualityState(message = "Discarded last clip.")
    }

    /**
     * Saves the captured clips as a new voice profile (or appends
     * them to an existing one if [appendToProfile] is non-null).
     */
    fun finishCoachedSession(name: String, appendToProfile: String? = null) {
        if (coachedClips.isEmpty()) {
            _coachState.value = _coachState.value.copy(
                statusMessage = "No clips captured."
            )
            return
        }
        val clipsSnapshot = coachedClips.toList()
        _coachState.value = _coachState.value.copy(
            isSaving = true,
            statusMessage = "Saving voice profile…"
        )
        viewModelScope.launch {
            try {
                val voiceState = withContext(Dispatchers.IO) {
                    pocketTts.cloneVoice(appendToProfile ?: name, clipsSnapshot)
                }
                withContext(Dispatchers.IO) {
                    voiceStore.saveVoice(voiceState)
                }
                clipsSnapshot.forEach { it.delete() }
                coachedClips.clear()
                _coachState.value = CoachState(
                    isSaved = true,
                    savedProfileName = voiceState.name,
                    statusMessage = "Saved as ${voiceState.name}."
                )
                loadVoices()
            } catch (e: Exception) {
                Log.e(TAG, "Coached session save failed", e)
                _coachState.value = _coachState.value.copy(
                    isSaving = false,
                    error = "Save failed: ${e.message}",
                    statusMessage = "Save failed: ${e.message}"
                )
            }
        }
    }

    /**
     * Abandons the active session and removes captured clips from
     * disk. Safe to call at any point (idempotent).
     */
    fun cancelCoachedSession() {
        voiceRecorder.stopRecording()
        clearCoachedClipsFromDisk()
        coachedClips.clear()
        _coachState.value = CoachState()
    }

    private fun clearCoachedClipsFromDisk() {
        coachedClips.forEach { it.delete() }
    }

    /**
     * Recomputes the derived fields of [CoachState] (count, total
     * seconds, quality tier) from the on-disk clip list and merges
     * them with [message] / [advancePrompt].
     *
     * - Always clears any prior `error` because this helper is only
     *   reached on success paths (clip recorded, clip discarded).
     *   Leaving a stale error in state would freeze the status line
     *   in error styling after the next successful recording.
     * - [advancePrompt] = true after a successful capture; the
     *   prompt index moves forward so the user sees a new sentence.
     *   On discard it's false, so the user lands back on the same
     *   prompt as their just-discarded clip and can retry it.
     */
    private fun recomputeCoachQualityState(message: String, advancePrompt: Boolean = false) {
        val clips = coachedClips.size
        val totalSec = (coachedClips.sumOf { wavDurationMs(it) } / 1000L).toInt()
        val tier = VoiceQualityTier.compute(clips, totalSec)
        val prev = _coachState.value
        _coachState.value = prev.copy(
            isRecording = false,
            clipCount = clips,
            totalSpeechSeconds = totalSec,
            qualityTier = tier,
            promptIndex = if (advancePrompt) prev.promptIndex + 1 else prev.promptIndex,
            statusMessage = message,
            error = null,
        )
    }

    /**
     * Header-only duration for a recorded WAV clip via [WavHeader].
     * Streams at most a few hundred bytes rather than loading the
     * whole file — relevant if the user backgrounds the app mid-
     * session and the OS swaps out memory.
     *
     * Falls back to assuming 16 kHz mono raw PCM (matching what
     * [VoiceRecorder] writes) when the header isn't parseable.
     */
    private fun wavDurationMs(file: File): Long {
        val header = WavHeader.read(file)
        if (header != null) return header.durationMs
        val len = try { file.length() } catch (_: Exception) { return 0L }
        if (len <= 44) return 0L
        return ((len - 44) / 2L) * 1000L / 16_000L
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
