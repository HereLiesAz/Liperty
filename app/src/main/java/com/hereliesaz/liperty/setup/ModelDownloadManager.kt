package com.hereliesaz.liperty.setup

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/**
 * Downloads ML model files from HuggingFace on first launch.
 *
 * After initial setup the app is fully offline — this is the only
 * component that uses the network. Models are stored in
 * [Context.getFilesDir] where [OnnxModelEngine], [PocketTTSEngine],
 * and other loaders already look before falling back to bundled assets.
 */
class ModelDownloadManager(private val context: Context) {

    companion object {
        private const val TAG = "ModelDownloadManager"
        private const val PREFS_NAME = "ModelDownloadPrefs"
        private const val KEY_SETUP_COMPLETE = "setup_complete"
        private const val KEY_SETUP_VERSION = "setup_version"
        /** When true (default), network downloads only run on Wi-Fi/Ethernet —
         *  not cellular. Saves battery (cellular radio is power-hungry) and
         *  avoids large transfers over metered data. */
        private const val KEY_WIFI_ONLY = "download_wifi_only"

        /** Bump when the required model manifest changes (forces re-check). */
        private const val CURRENT_SETUP_VERSION = 3

        /** HuggingFace CDN pattern — follows redirect to CF/S3 edge. */
        private const val HF_BASE = "https://huggingface.co"
        private fun hfUrl(repo: String, file: String) =
            "$HF_BASE/$repo/resolve/main/$file"

        /**
         * Floor for "is this file likely a real artifact vs an HTML error page".
         * HTML error pages from CDNs are typically a few hundred bytes to a
         * couple KB. The optimizer ONNX is a legitimate 538-byte artifact, so
         * we can't crank this any higher. The real "did the download finish"
         * check happens in [downloadFile] against the server's Content-Length.
         */
        private const val MIN_FILE_BYTES = 100L

        /** Buffer size for streaming downloads. */
        private const val BUFFER_SIZE = 64 * 1024 // 64 KB

        /** Max manual (cross-protocol) redirect hops before giving up, so a
         *  misconfigured redirect loop can't recurse into a StackOverflowError. */
        private const val MAX_REDIRECTS = 5
    }

    // ── Model manifest ────────────────────────────────────────────────

    data class ModelSpec(
        val fileName: String,
        val repo: String,
        val required: Boolean,
        val description: String,
        val expectedSizeBytes: Long = -1L,
        /** Override HF URL for non-HuggingFace sources (e.g. MediaPipe CDN). */
        val customUrl: String? = null,
        /**
         * Remote path within the HF repo. Defaults to [fileName]. Use when the
         * local destination path differs from the file's path on the Hub —
         * e.g. training artifacts that live at the repo root but are stored
         * locally under a `training/` subdirectory.
         */
        val remotePath: String? = null
    )

    val models: List<ModelSpec> = listOf(
        // MediaPipe face/hand detection — required for the VSR pipeline
        ModelSpec(
            fileName = "face_landmarker.task",
            repo = "",
            required = true,
            description = "Face detection model",
            expectedSizeBytes = 3_800_000L,
            customUrl = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        ),
        ModelSpec(
            fileName = "hand_landmarker.task",
            repo = "",
            required = false,
            description = "Hand gesture model",
            expectedSizeBytes = 7_900_000L,
            customUrl = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        ),
        // Production VSR model — CTC fallback path (graceful degradation when the
        // larger seq2seq encoder/decoder can't load, e.g. OOM on low-RAM devices).
        ModelSpec(
            fileName = "syncvsr_lrs3_visual_ctc_fp16.onnx",
            repo = "HereLiesAz/liperty-syncvsr-onnx",
            required = true,
            description = "Visual Speech Recognition model",
            expectedSizeBytes = 370_000_000L
        ),
        // SyncVSR seq2seq backend (active production path: VSR_BACKEND ==
        // BACKEND_SYNC_VSR, SYNCVSR_USE_SEQ2SEQ == true). The vocab is tiny but
        // required by BOTH the seq2seq and CTC paths; the encoder/decoder are
        // large and never bundled in the APK, so they MUST be downloaded here.
        ModelSpec(
            fileName = "syncvsr_unigram_units.txt",
            repo = "HereLiesAz/liperty-syncvsr-onnx",
            required = true,
            description = "Speech recognition vocabulary",
            expectedSizeBytes = 65_000L
        ),
        ModelSpec(
            fileName = "syncvsr_lrs3_encoder.onnx",
            repo = "HereLiesAz/liperty-syncvsr-onnx",
            required = true,
            description = "Speech recognition encoder (high accuracy)",
            expectedSizeBytes = 759_000_000L
        ),
        ModelSpec(
            fileName = "syncvsr_lrs3_decoder.onnx",
            repo = "HereLiesAz/liperty-syncvsr-onnx",
            required = true,
            description = "Speech recognition decoder (high accuracy)",
            expectedSizeBytes = 273_000_000L
        ),
        ModelSpec(
            fileName = "librispeech_3gram.bin",
            repo = "HereLiesAz/liperty-lm",
            required = false,
            description = "Language model (improves accuracy)",
            expectedSizeBytes = 27_000_000L
        ),
        // OpenVoice v2 — zero-shot voice cloning, three ONNX components.
        ModelSpec(
            fileName = "pocket_tts_se_extractor.onnx",
            repo = "HereLiesAz/liperty-pocket-tts",
            required = false,
            description = "Voice cloning (speaker embedding extractor)",
            expectedSizeBytes = 3_300_000L
        ),
        ModelSpec(
            fileName = "pocket_tts_base.onnx",
            repo = "HereLiesAz/liperty-pocket-tts",
            required = false,
            description = "Voice cloning (MeloTTS base voice)",
            expectedSizeBytes = 170_000_000L
        ),
        ModelSpec(
            fileName = "pocket_tts_tone_converter.onnx",
            repo = "HereLiesAz/liperty-pocket-tts",
            required = false,
            description = "Voice cloning (tone color converter)",
            expectedSizeBytes = 128_000_000L
        ),
        ModelSpec(
            fileName = "pocket_tts_vocab.json",
            repo = "HereLiesAz/liperty-pocket-tts",
            required = false,
            description = "Voice cloning (MeloTTS vocab)",
            expectedSizeBytes = 6_200L
        ),
        // On-device training artifacts (personalization Step 3)
        // ~820 MB total; optional — app works fine without them.
        ModelSpec(
            fileName = "training/training_model.onnx",
            remotePath = "training_model.onnx",
            repo = "HereLiesAz/liperty-v3-training-artifacts",
            required = false,
            description = "Personalization training model",
            expectedSizeBytes = 410_600_000L
        ),
        ModelSpec(
            fileName = "training/eval_model.onnx",
            remotePath = "eval_model.onnx",
            repo = "HereLiesAz/liperty-v3-training-artifacts",
            required = false,
            description = "Personalization evaluation model",
            expectedSizeBytes = 410_800_000L
        ),
        ModelSpec(
            fileName = "training/optimizer_model.onnx",
            remotePath = "optimizer_model.onnx",
            repo = "HereLiesAz/liperty-v3-training-artifacts",
            required = false,
            description = "Personalization optimizer",
            expectedSizeBytes = 538L
        ),
    )

    // ── Download state ────────────────────────────────────────────────

    data class DownloadState(
        val modelStates: Map<String, ModelDownloadState> = emptyMap(),
        val isComplete: Boolean = false,
        val error: String? = null,
        /** Current Wi-Fi-only preference, mirrored here so the UI is state-driven. */
        val wifiOnly: Boolean = true,
        /** True when a download is held back because Wi-Fi-only is on and the
         *  device is on cellular. The UI shows a "waiting for Wi-Fi" prompt. */
        val blockedOnNetwork: Boolean = false
    )

    data class ModelDownloadState(
        val fileName: String,
        val description: String,
        val required: Boolean,
        val status: Status = Status.PENDING,
        val progressBytes: Long = 0,
        val totalBytes: Long = -1,
        val error: String? = null
    ) {
        val progressFraction: Float
            get() = if (totalBytes > 0) progressBytes.toFloat() / totalBytes else 0f
    }

    enum class Status { PENDING, DOWNLOADING, COMPLETE, SKIPPED, ERROR, WAITING_WIFI }

    private val _state = MutableStateFlow(DownloadState(wifiOnly = isWifiOnly()))
    val state: StateFlow<DownloadState> = _state.asStateFlow()

    private fun prefs() = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    /** Whether downloads are restricted to Wi-Fi/Ethernet. Defaults to true. */
    fun isWifiOnly(): Boolean = prefs().getBoolean(KEY_WIFI_ONLY, true)

    /**
     * Persists the Wi-Fi-only preference. When the user switches to allow
     * cellular while a download is currently blocked, resume it immediately.
     */
    suspend fun setWifiOnly(enabled: Boolean) {
        prefs().edit().putBoolean(KEY_WIFI_ONLY, enabled).apply()
        val wasBlocked = _state.value.blockedOnNetwork
        _state.value = _state.value.copy(wifiOnly = enabled)
        if (!enabled && wasBlocked) {
            // Cellular now permitted and we were waiting — pick up where we left off.
            downloadAll(skipOptional = false)
        }
    }

    /**
     * True if the active network satisfies the current download policy. With
     * Wi-Fi-only on, only WIFI/ETHERNET transports qualify (a metered Wi-Fi
     * hotspot still counts as Wi-Fi by transport, which matches the user's
     * "Wi-Fi only" intent of avoiding the cellular radio). With it off, any
     * validated network is fine.
     */
    private fun networkAllowsDownload(): Boolean {
        if (!isWifiOnly()) return true
        val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager
            ?: return false
        val caps = cm.getNetworkCapabilities(cm.activeNetwork) ?: return false
        return caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) ||
            caps.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET)
    }

    /** True if all required models are present on disk. */
    fun isSetupComplete(): Boolean {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val storedVersion = prefs.getInt(KEY_SETUP_VERSION, 0)

        // Version mismatch — manifest changed (new required models added)
        if (storedVersion < CURRENT_SETUP_VERSION) {
            prefs.edit().putBoolean(KEY_SETUP_COMPLETE, false).apply()
        } else if (prefs.getBoolean(KEY_SETUP_COMPLETE, false)) {
            return true
        }

        // Check if all required models exist (filesDir or bundled assets)
        val allRequiredPresent = models.filter { it.required }.all { isModelPresent(it) }

        if (allRequiredPresent) {
            prefs.edit()
                .putBoolean(KEY_SETUP_COMPLETE, true)
                .putInt(KEY_SETUP_VERSION, CURRENT_SETUP_VERSION)
                .apply()
        }
        return allRequiredPresent
    }

    /**
     * Downloads all models that aren't already present.
     * Call from a coroutine scope (e.g., viewModelScope).
     */
    suspend fun downloadAll(skipOptional: Boolean = false) {
        // Initialize state (preserve the Wi-Fi-only preference; clear stale block)
        val initialStates = models.associate { spec ->
            val alreadyPresent = isModelPresent(spec)
            val shouldSkip = skipOptional && !spec.required
            spec.fileName to ModelDownloadState(
                fileName = spec.fileName,
                description = spec.description,
                required = spec.required,
                status = when {
                    alreadyPresent -> Status.COMPLETE
                    shouldSkip -> Status.SKIPPED
                    else -> Status.PENDING
                }
            )
        }
        _state.value = DownloadState(modelStates = initialStates, wifiOnly = isWifiOnly())

        // Download each model that isn't already present
        for (spec in models) {
            val current = _state.value.modelStates[spec.fileName] ?: continue
            if (current.status == Status.COMPLETE || current.status == Status.SKIPPED) continue

            try {
                downloadModel(spec)
            } catch (e: NetworkBlockedException) {
                // Wi-Fi-only is on and we're on cellular. Stop here (the model is
                // already marked WAITING_WIFI) and let the UI prompt the user.
                _state.value = _state.value.copy(
                    blockedOnNetwork = true,
                    error = "Waiting for Wi-Fi. Connect to Wi-Fi, or allow cellular to continue."
                )
                return
            } catch (e: Exception) {
                Log.e(TAG, "Failed to download ${spec.fileName}", e)
                updateModelState(spec.fileName) {
                    it.copy(status = Status.ERROR, error = e.message)
                }
                if (spec.required) {
                    _state.value = _state.value.copy(
                        error = "Required model failed: ${spec.description}"
                    )
                    return
                }
            }
        }

        // Check completion
        val allRequiredDone = models.filter { it.required }.all { spec ->
            _state.value.modelStates[spec.fileName]?.status == Status.COMPLETE
        }
        if (allRequiredDone) {
            context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                .edit()
                .putBoolean(KEY_SETUP_COMPLETE, true)
                .putInt(KEY_SETUP_VERSION, CURRENT_SETUP_VERSION)
                .apply()
        }
        _state.value = _state.value.copy(isComplete = allRequiredDone)
    }

    /** Retry a specific failed model download. */
    suspend fun retryModel(fileName: String) {
        val spec = models.find { it.fileName == fileName } ?: return
        updateModelState(fileName) { it.copy(status = Status.PENDING, error = null) }
        try {
            downloadModel(spec)
            // Re-check overall completion
            val allRequiredDone = models.filter { it.required }.all { s ->
                _state.value.modelStates[s.fileName]?.status == Status.COMPLETE
            }
            if (allRequiredDone) {
                context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                    .edit()
                    .putBoolean(KEY_SETUP_COMPLETE, true)
                    .putInt(KEY_SETUP_VERSION, CURRENT_SETUP_VERSION)
                    .apply()
                _state.value = _state.value.copy(isComplete = true, error = null)
            }
        } catch (e: NetworkBlockedException) {
            // Already marked WAITING_WIFI by downloadModel; surface the prompt.
            _state.value = _state.value.copy(
                blockedOnNetwork = true,
                error = "Waiting for Wi-Fi. Connect to Wi-Fi, or allow cellular to continue."
            )
        } catch (e: Exception) {
            Log.e(TAG, "Retry failed for $fileName", e)
            updateModelState(fileName) {
                it.copy(status = Status.ERROR, error = e.message)
            }
        }
    }

    /** Thrown by [downloadModel] when a network transfer is needed but the
     *  Wi-Fi-only policy disallows the current (cellular) connection. */
    private class NetworkBlockedException : Exception()

    // ── Internal ──────────────────────────────────────────────────────

    private fun isModelPresent(spec: ModelSpec): Boolean {
        val file = File(context.filesDir, spec.fileName)
        if (file.exists() && file.length() >= MIN_FILE_BYTES) return true
        return try {
            context.assets.open(spec.fileName).use { it.available() >= MIN_FILE_BYTES }
        } catch (_: Exception) { false }
    }

    private suspend fun downloadModel(spec: ModelSpec) = withContext(Dispatchers.IO) {
        val dest = File(context.filesDir, spec.fileName)
        dest.parentFile?.mkdirs()

        // Already present (from assets copy or previous download)
        if (dest.exists() && dest.length() >= MIN_FILE_BYTES) {
            updateModelState(spec.fileName) { it.copy(status = Status.COMPLETE) }
            return@withContext
        }

        // Try assets first (dev build with setup_libs.sh)
        try {
            context.assets.open(spec.fileName).use { input ->
                if (input.available() >= MIN_FILE_BYTES) {
                    dest.outputStream().use { output -> input.copyTo(output) }
                    if (dest.length() >= MIN_FILE_BYTES) {
                        Log.i(TAG, "Copied ${spec.fileName} from assets (${dest.length()} bytes)")
                        updateModelState(spec.fileName) { it.copy(status = Status.COMPLETE) }
                        return@withContext
                    }
                }
            }
        } catch (_: Exception) {
            // Asset not bundled — expected for production builds
        }

        // A network transfer is required from here (not already on disk or in
        // assets). Honor the Wi-Fi-only policy before touching the radio.
        if (!networkAllowsDownload()) {
            updateModelState(spec.fileName) { it.copy(status = Status.WAITING_WIFI) }
            throw NetworkBlockedException()
        }

        // Download from source (custom URL or HuggingFace)
        val url = spec.customUrl ?: hfUrl(spec.repo, spec.remotePath ?: spec.fileName)

        updateModelState(spec.fileName) {
            it.copy(status = Status.DOWNLOADING, progressBytes = 0)
        }

        val tempFile = File(dest.parentFile, "${dest.name}.tmp")
        try {
            downloadFile(url, tempFile, spec.fileName)
            if (tempFile.length() < MIN_FILE_BYTES) {
                tempFile.delete()
                throw RuntimeException(
                    "Downloaded file too small (${tempFile.length()} bytes) — likely a stub or error page"
                )
            }
            tempFile.renameTo(dest)
            Log.i(TAG, "Downloaded ${spec.fileName} (${dest.length()} bytes)")
            updateModelState(spec.fileName) { it.copy(status = Status.COMPLETE) }
        } catch (e: Exception) {
            tempFile.delete()
            throw e
        }
    }

    private suspend fun downloadFile(
        urlStr: String, dest: File, modelKey: String, redirectsLeft: Int = MAX_REDIRECTS
    ): Unit = withContext(Dispatchers.IO) {
        var conn: HttpURLConnection? = null
        try {
            val url = URL(urlStr)
            conn = url.openConnection() as HttpURLConnection
            conn.connectTimeout = 30_000
            conn.readTimeout = 60_000
            conn.instanceFollowRedirects = true

            val responseCode = conn.responseCode
            if (responseCode != HttpURLConnection.HTTP_OK) {
                // HuggingFace returns 302 → CDN. HttpURLConnection follows
                // redirects automatically for same-protocol. For cross-protocol
                // (http→https) we handle manually — bounded by [redirectsLeft]
                // so a misconfigured redirect loop can't recurse to a
                // StackOverflowError.
                if (responseCode in 300..399) {
                    val redirect = conn.getHeaderField("Location")
                    conn.disconnect()
                    if (redirect != null) {
                        if (redirectsLeft <= 0) {
                            throw RuntimeException("Too many redirects fetching $modelKey (loop?)")
                        }
                        downloadFile(redirect, dest, modelKey, redirectsLeft - 1)
                        return@withContext
                    }
                }
                throw RuntimeException("HTTP $responseCode for $urlStr")
            }

            val totalBytes = conn.contentLengthLong
            updateModelState(modelKey) { it.copy(totalBytes = totalBytes) }

            var downloaded = 0L
            conn.inputStream.buffered(BUFFER_SIZE).use { input ->
                dest.outputStream().buffered(BUFFER_SIZE).use { output ->
                    val buffer = ByteArray(BUFFER_SIZE)
                    var lastReportedPercent = -1
                    while (true) {
                        val bytesRead = input.read(buffer)
                        if (bytesRead == -1) break
                        output.write(buffer, 0, bytesRead)
                        downloaded += bytesRead

                        // Throttle state updates to ~1% increments
                        val percent = if (totalBytes > 0) {
                            (downloaded * 100 / totalBytes).toInt()
                        } else -1
                        if (percent != lastReportedPercent) {
                            lastReportedPercent = percent
                            updateModelState(modelKey) {
                                it.copy(progressBytes = downloaded)
                            }
                        }
                    }
                }
            }

            // Truncation guard: if the server advertised a length, we must have
            // received all of it. Catches connection drops that close cleanly
            // mid-stream without throwing IOException.
            if (totalBytes > 0 && downloaded < totalBytes) {
                throw RuntimeException(
                    "Truncated download for $urlStr: $downloaded of $totalBytes bytes"
                )
            }
        } finally {
            conn?.disconnect()
        }
    }

    private fun updateModelState(
        fileName: String,
        transform: (ModelDownloadState) -> ModelDownloadState
    ) {
        val current = _state.value
        val currentModel = current.modelStates[fileName] ?: return
        _state.value = current.copy(
            modelStates = current.modelStates + (fileName to transform(currentModel))
        )
    }
}
