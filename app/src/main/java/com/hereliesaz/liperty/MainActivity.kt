package com.hereliesaz.liperty

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
 import android.graphics.PointF
import android.graphics.RectF
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.ViewGroup
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.view.PreviewView
import androidx.compose.runtime.mutableStateOf
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.hereliesaz.liperty.camera.CameraManager
import com.hereliesaz.liperty.ml.CalibrationViewModel
import com.hereliesaz.liperty.ml.FaceLandmarkerHelper
import com.hereliesaz.liperty.ml.HandGestureHelper
import com.hereliesaz.liperty.voicebox.LaryngealSensor

import com.hereliesaz.liperty.ml.FrameBuffer
import com.hereliesaz.liperty.ml.AvHubertSeq2SeqInference
import com.hereliesaz.liperty.ml.KenLmScorer
import com.hereliesaz.liperty.ml.OnnxModelEngine
import com.hereliesaz.liperty.ml.VisemeIndex
import com.hereliesaz.liperty.ml.VisemeRescorer
import com.hereliesaz.liperty.ml.SubwordCtcBeamDecoder
import com.hereliesaz.liperty.ml.TFLiteEngine
import com.hereliesaz.liperty.ml.VSRInference
import com.hereliesaz.liperty.ml.VocabularyLoader
import com.hereliesaz.liperty.ml.SSRInference
import com.hereliesaz.liperty.ui.LipertyApp
import com.hereliesaz.liperty.ui.OverlayView
import com.hereliesaz.liperty.ui.SettingsActivity
import com.hereliesaz.liperty.ui.TranscriptionManager
import com.hereliesaz.liperty.utils.BitmapPool
import com.hereliesaz.liperty.utils.ImageUtils
import com.hereliesaz.liperty.utils.PerformanceMonitor
import com.hereliesaz.liperty.utils.RectKalmanFilter
import com.hereliesaz.liperty.voicebox.VoiceManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {

    private companion object AutoAvsrConfig {
        // Asset filenames bundled by setup_libs.sh.
        const val AUTOAVSR_MODEL = "autoavsr_lrs3_visual_ctc.onnx"
        const val AUTOAVSR_VOCAB = "unigram5000_units.txt"

        // Mouth-ROI crop pulled from Chaplin's pipelines/data/transforms.py:
        //     CenterCrop(88), Normalize(mean=0.421, std=0.165)  (grayscale)
        const val AUTOAVSR_CROP_SIZE = 88
        const val AUTOAVSR_PIXEL_MEAN = 0.421f
        const val AUTOAVSR_PIXEL_STD = 0.165f

        // Sliding-window inference: each call retains this many recent frames
        // for the next window so the model gets overlapping context across
        // utterance boundaries. With buffer capacity 16 and retain 8, we
        // re-process the second half of each window with the first half of
        // the next — every new frame is seen by exactly two inferences.
        const val AUTOAVSR_SLIDE_RETAIN = 8


        // V3 backend (AV-HuBERT seq2seq). Pulled by setup_libs.sh from
        // HereLiesAz/liperty-avhubert-encoder. Encoder + decoder + dict
        // are mean/std-compatible with the Auto-AVSR preprocessing above,
        // so the camera pipeline doesn't need to change. The decoder is
        // autoregressive (Transformer with cross-attention) — see
        // docs/AVHUBERT_V3_BACKEND.md for the rationale.
        const val AVHUBERT_V3_ENCODER_MODEL = "avhubert_base_vox_433h_visual_encoder.onnx"
        const val AVHUBERT_V3_DECODER_MODEL = "avhubert_base_vox_433h_decoder.onnx"
        const val AVHUBERT_V3_DICT = "avhubert_base_vox_433h_dict.txt"
        // Fairseq's canonical special-token indices for AV-HuBERT's 1000-token
        // unigram BPE vocab: 0=<s>(bos), 1=<pad>, 2=</s>(eos), 3=<unk>.
        const val AVHUBERT_V3_BOS = 0
        const val AVHUBERT_V3_EOS = 2
        // Encoder window: trained on ~50-frame clips. Matches V2's FrameBuffer.
        const val AVHUBERT_V3_NUM_FRAMES = 50
        // Beyond ~50 BPE tokens is unrealistic for a Liperty utterance and
        // protects against decoder runaway when the encoder features carry
        // no signal (e.g. user out of frame).
        const val AVHUBERT_V3_MAX_DECODE = 50

        // Compile-time switch. V3 stays off by default until on-device WER
        // beats V2 (Phase 4 in docs/AVHUBERT_V3_BACKEND.md). Flip to true
        // for a research build that exercises the new path end-to-end.
        const val USE_V3_BACKEND = false

        // KenLM n-gram language model for n-best rescoring on top of CTC
        // beam search (Phase A — see docs/AVHUBERT_V3_BACKEND.md). Pulled
        // by setup_libs.sh from HereLiesAz/liperty-lm. LibriSpeech 3-gram
        // pruned 1e-7, KenLM trie+q8 binary, ~27 MB. The vocabulary is
        // UPPERCASE — KenLmScorer.score() uppercases input automatically.
        const val KENLM_MODEL = "librispeech_3gram.bin"
        // Conventional shallow-fusion weight for n-gram LMs in ASR (0.3-0.7).
        // Tune on a held-out dev set; 0.5 is a sensible starting point.
        const val KENLM_WEIGHT = 0.5f

        // Viseme rescorer assets (built offline by tools/build_viseme_index.py
        // and shipped in the APK). Cmudict-derived: 126k words across 30k
        // unique viseme sequences. ~2 MB on disk.
        const val VISEME_INDEX = "viseme_index.json"
    }

    private lateinit var cameraManager: CameraManager
    private lateinit var faceLandmarkerHelper: FaceLandmarkerHelper
    private lateinit var handGestureHelper: HandGestureHelper
    private lateinit var overlayView: OverlayView
    private lateinit var previewView: PreviewView
    private lateinit var vsrInference: VSRInference
    /** V3 AV-HuBERT seq2seq backend. Non-null only when [USE_V3_BACKEND]
     *  is true and the assets are present. Research path; not yet
     *  validated on real users. See docs/AVHUBERT_V3_BACKEND.md. */
    private var avhubertV3: AvHubertSeq2SeqInference? = null
    /** Viseme-aware post-rescorer. Non-null when both the viseme index
     *  and the KenLM language model loaded successfully. See
     *  [VisemeRescorer] for the algorithm — it's the "Chaplin's second
     *  AI" tailored for visual ASR (swaps in viseme-equivalent words
     *  that score higher under the LM). */
    private var visemeRescorer: VisemeRescorer? = null
    private lateinit var ssrInference: SSRInference
    private lateinit var frameBuffer: FrameBuffer
    private val lipBoxFilter = RectKalmanFilter()
    private val opticalFlowTracker = com.hereliesaz.liperty.utils.OpticalFlowTracker()

    private val transcriptionManager by lazy { TranscriptionManager(this) }
    private lateinit var voiceManager: VoiceManager
    private lateinit var laryngealSensor: LaryngealSensor

    // Compose State
    private val transcriptionWords = mutableStateOf(listOf<String>())
    private val wordConfidences = mutableStateOf(listOf<Float>())
    /** Words from the latest inference window that haven't yet been confirmed
     *  by a subsequent window. Rendered as a dimmer live-preview tail so the
     *  user sees speech-as-it's-mouthed before the next inference commits it. */
    private val liveTranscriptionWords = mutableStateOf(listOf<String>())
    private val liveWordConfidences = mutableStateOf(listOf<Float>())
    private val selectedWordIndex = mutableStateOf(-1)
    private val isRecordingState = mutableStateOf(false)
    private val isBCModeState = mutableStateOf(false)
    private val carrierF0State = mutableStateOf(120f)
    private val isLipReadModeState = mutableStateOf(true)
    private val isELModeState = mutableStateOf(false)

    // Camera State
    private var currentLensFacing = CameraSelector.LENS_FACING_FRONT
    private var telephotoPreference = true

    // Sensitivity State
    private val vsrSensitivity = mutableStateOf(0.5f)
    private val larynxSensitivity = mutableStateOf(0.5f)
    private val isDarkTheme = mutableStateOf(true)

    // Flag to prevent overlapping inference calls
    @Volatile private var isInferencing = false
    private val isPausedState = mutableStateOf(false)
    @Volatile private var frameCount = 0
    @Volatile private var calibrationCallback: ((Bitmap) -> Unit)? = null

    // True only when the try block in onCreate completed without throwing.
    // Gates every code path that touches the ML/camera lateinit fields so the
    // app can still launch (and show the init-error Toast) when assets are
    // missing instead of force-closing.
    @Volatile private var coreInitialized = false

    // Tracks whether BC/EL was active before onPause so we can restore on resume
    @Volatile private var wasBCActiveBeforePause = false
    @Volatile private var wasELActiveBeforePause = false

    private val requestPermissionsLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { results ->
            val allGranted = results.values.all { it }
            if (allGranted) {
                checkConsentAndStart()
            } else {
                val denied = results.filter { !it.value }.keys.joinToString(", ") {
                    it.removePrefix("android.permission.")
                }
                Toast.makeText(this, getString(R.string.common_permissions_denied, denied), Toast.LENGTH_LONG).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        ImageUtils.initializeOpenCV(this)

        // Initialize Views programmatically for Compose AndroidView
        previewView = PreviewView(this)
        overlayView = OverlayView(this, null).apply {
             layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        }

        try {
            cameraManager = CameraManager(this)
            faceLandmarkerHelper = FaceLandmarkerHelper(this)
            handGestureHelper = HandGestureHelper(this)
            laryngealSensor = LaryngealSensor(this)
            // Auto-AVSR (Chaplin's LRS3_V_WER19.1, exported via
            // tools/export_autoavsr_to_onnx.ipynb) is the active backend.
            // 88x88 grayscale mouth ROI, normalized to mean=0.421 std=0.165,
            // emits log-softmax over 5049 SentencePiece subword tokens.
            val vsrEngine = OnnxModelEngine(this, AUTOAVSR_MODEL)
            val subwordVocab = VocabularyLoader.loadFromAssets(this, AUTOAVSR_VOCAB, blank = "<blank>")
            // Beam search over the 5049-token vocabulary. Beam width 8 picked
            // to match Chaplin's effective recognizer cost (Chaplin uses
            // beam_size=40 at greedy LM-scoring, but we have no LM and a
            // smaller-width search is enough to recover from greedy's
            // path-collapse mistakes).
            // Try to load the KenLM language model for n-best rescoring on
            // top of CTC beam search. Gracefully no-ops if either the .bin
            // file isn't in assets (setup_libs.sh failure / older builds) or
            // libkenlm.so isn't packaged (JNI not yet wired — Phase A3b/c).
            val kenLmScorer: KenLmScorer? = try {
                // KenLM expects a real on-disk path; stream the asset out so
                // build_binary's mmap path works. Matches OnnxModelEngine's
                // approach so the same caching semantics apply.
                val dst = java.io.File(filesDir, KENLM_MODEL)
                // Use input stream's available() instead of openFd() because
                // aapt2 compresses .bin assets by default, and openFd only
                // works on uncompressed entries. The decompressed size shows
                // up as available() on the AssetInputStream.
                val assetSize = assets.open(KENLM_MODEL).use { it.available().toLong() }
                if (!dst.exists() || dst.length() != assetSize) {
                    assets.open(KENLM_MODEL).use { input ->
                        dst.outputStream().use { output -> input.copyTo(output) }
                    }
                }
                KenLmScorer.tryLoad(dst.absolutePath).also {
                    Log.i("MainActivity", "KenLM scorer: isNativeLoaded=${it.isNativeLoaded}")
                }
            } catch (e: Exception) {
                Log.w("MainActivity", "KenLM scorer init skipped (${e.message})")
                null
            }
            val subwordDecoder = SubwordCtcBeamDecoder(
                vocabulary = subwordVocab,
                beamWidth = 8,
                blankIndex = 0,
                lmScorer = kenLmScorer,
                lmWeight = if (kenLmScorer?.isNativeLoaded == true) KENLM_WEIGHT else 0f,
            )

            // Viseme rescorer is independent of the JNI status — it only
            // needs the viseme index asset and a LanguageModelScorer (any
            // one will do; if KenLM isn't native-loaded, the rescorer
            // gracefully no-ops because every candidate scores 0).
            visemeRescorer = try {
                val idx = VisemeIndex.loadFromAssets(this, VISEME_INDEX)
                val lm = kenLmScorer ?: com.hereliesaz.liperty.ml.NoopLanguageModelScorer()
                VisemeRescorer(idx, lm).also {
                    Log.i("MainActivity", "VisemeRescorer ready (LM=${if (kenLmScorer?.isNativeLoaded == true) "kenlm" else "noop"})")
                }
            } catch (e: Exception) {
                Log.w("MainActivity", "VisemeRescorer init skipped (${e.message})")
                null
            }
            vsrInference = VSRInference(
                engine = vsrEngine,
                pixelMean = AUTOAVSR_PIXEL_MEAN,
                pixelStd = AUTOAVSR_PIXEL_STD,
                customDecoder = subwordDecoder::decode,
            )
            ssrInference = SSRInference(TFLiteEngine(this, "ssr_model.tflite"))
            // Auto-AVSR's encoder accepts variable T; 16 frames keeps cadence
            // close to the legacy VideoMAE pipeline.
            frameBuffer = FrameBuffer(capacity = 16)

            if (USE_V3_BACKEND) {
                // Optional: build the AV-HuBERT seq2seq backend in parallel.
                // Asset misses (model files not on disk) are non-fatal —
                // we leave avhubertV3 null and the pipeline falls back to V2.
                try {
                    avhubertV3 = AvHubertSeq2SeqInference.create(this).also {
                        Log.i("MainActivity", "V3 AV-HuBERT seq2seq backend constructed")
                    }
                } catch (e: Exception) {
                    Log.w("MainActivity", "V3 backend init skipped (${e.message}); using V2")
                    avhubertV3 = null
                }
            }
            coreInitialized = true
        } catch (e: Exception) {
            Log.e("MainActivity", "Failed to initialize components", e)
            Toast.makeText(this, getString(R.string.common_init_error, e.message ?: ""), Toast.LENGTH_LONG).show()
        }

        // Initialize TFLite in background — only if the lateinit fields above
        // were successfully constructed. Touching them otherwise would throw
        // UninitializedPropertyAccessException and crash the activity.
        if (coreInitialized) {
            lifecycleScope.launch(Dispatchers.Default) {
                vsrInference.initialize()
                ssrInference.initialize()
                avhubertV3?.let {
                    if (!it.initialize()) {
                        Log.w("MainActivity", "V3 backend.initialize() returned false; falling back to V2")
                        avhubertV3 = null
                    }
                }
            }
        }

        // Initialize VoiceManager. PocketTTSEngine's ctor calls
        // OrtEnvironment.getEnvironment() as a property initializer, which
        // triggers ONNX Runtime native-lib loading and throws RuntimeException
        // if the .so isn't packaged. Without this guard, that takes down the
        // whole activity — onDestroy already tolerates voiceManager being
        // uninitialized (::voiceManager.isInitialized).
        try {
            voiceManager = VoiceManager(this) { ready ->
                if (ready) {
                    Log.i("MainActivity", "VoiceManager initialized.")
                } else {
                    Log.e("MainActivity", "VoiceManager initialization failed.")
                }
            }
        } catch (t: Throwable) {
            Log.e("MainActivity", "VoiceManager construction failed — TTS/voice cloning disabled", t)
            Toast.makeText(this, getString(R.string.common_voice_unavailable), Toast.LENGTH_LONG).show()
        }

        setContent {
            LipertyApp(
                previewView = previewView,
                overlayView = overlayView,
                transcriptionWords = transcriptionWords.value,
                liveWords = liveTranscriptionWords.value,
                liveWordConfidences = liveWordConfidences.value,
                selectedWordIndex = selectedWordIndex.value,
                onWordClick = { index ->
                    transcriptionManager.selectWord(index)
                    updateTranscriptionUI()
                },
                wordConfidences = wordConfidences.value,
                isRecording = isRecordingState.value,
                onSwitchCamera = { toggleCamera() },
                onOpenSettings = { startActivity(Intent(this, SettingsActivity::class.java)) },
                onClearTranscript = {
                    transcriptionManager.clear()
                    updateTranscriptionUI()
                    if (coreInitialized) frameBuffer.clearAndRecycle()
                },
                onSpeak = { speakText() },
                onToggleBC = { toggleBCMode() },
                onToggleLipRead = { isLipReadModeState.value = !isLipReadModeState.value },
                onToggleEL = { toggleELMode() },
                isPaused = isPausedState.value,
                isBCActive = isBCModeState.value,
                isLipReadActive = isLipReadModeState.value,
                isELActive = isELModeState.value,
                currentLensFacing = if (currentLensFacing == CameraSelector.LENS_FACING_BACK) 1 else 0,
                vsrSensitivity = vsrSensitivity.value,
                onVsrSensitivityChange = { value ->
                    vsrSensitivity.value = value
                    if (coreInitialized) faceLandmarkerHelper.updateConfidence(value, value)
                },
                larynxSensitivity = larynxSensitivity.value,
                onLarynxSensitivityChange = {
                    larynxSensitivity.value = it
                    if (coreInitialized) laryngealSensor.setSensitivity(it)
                },
                carrierF0 = carrierF0State.value,
                onCarrierF0Change = { hz ->
                    carrierF0State.value = hz
                    if (coreInitialized) laryngealSensor.setCarrierF0(hz)
                },
                isDarkTheme = isDarkTheme.value,
                onRegisterCalibrationCallback = { cb ->
                    calibrationCallback = cb
                }
            )
        }

        if (allPermissionsGranted()) {
            checkConsentAndStart()
        } else {
            val perms = mutableListOf(
                Manifest.permission.CAMERA,
                Manifest.permission.RECORD_AUDIO
            )
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                perms.add(Manifest.permission.BLUETOOTH_CONNECT)
            }
            requestPermissionsLauncher.launch(perms.toTypedArray())
        }
    }

    override fun onPause() {
        super.onPause()
        if (!coreInitialized) return
        // Stop audio resources — camera stops automatically via CameraX lifecycle binding
        wasBCActiveBeforePause = isBCModeState.value
        wasELActiveBeforePause = isELModeState.value
        if (isBCModeState.value || isELModeState.value) {
            laryngealSensor.stop()
            isBCModeState.value = false
            isELModeState.value = false
        }
        isRecordingState.value = false
    }

    override fun onResume() {
        super.onResume()
        if (!coreInitialized) return
        applySettings()
        // Restart camera if consent was previously granted
        val consentGranted = getSharedPreferences("LipertyPrefs", Context.MODE_PRIVATE)
            .getBoolean("consent_granted", false)
        if (consentGranted && allPermissionsGranted()) {
            startCamera()
        }
        // Restore voice mode if it was active before pause
        if (wasBCActiveBeforePause) {
            wasBCActiveBeforePause = false
            toggleBCMode()
        } else if (wasELActiveBeforePause) {
            wasELActiveBeforePause = false
            toggleELMode()
        }
    }

    private fun applySettings() {
        val sharedPrefs = getSharedPreferences("LipertyPrefs", Context.MODE_PRIVATE)

        // Theme
        isDarkTheme.value = sharedPrefs.getBoolean("dark_theme", true)

        // Telephoto Preference
        val newTelephotoPref = sharedPrefs.getBoolean("telephoto_preference", true)
        if (newTelephotoPref != telephotoPreference) {
            telephotoPreference = newTelephotoPref
            // Restart camera if currently using back camera to apply lens change
            if (currentLensFacing == CameraSelector.LENS_FACING_BACK) {
                startCamera()
            }
        }
    }

    private fun checkConsentAndStart() {
        if (!coreInitialized) return
        val sharedPrefs = getSharedPreferences("LipertyPrefs", Context.MODE_PRIVATE)
        val consentGranted = sharedPrefs.getBoolean("consent_granted", false)
        val calibrationDone = sharedPrefs.getBoolean("calibration_complete", false)

        if (consentGranted) {
            startCamera()
            if (!calibrationDone) {
                // Ideally we should navigate to calibration route here.
                // For now, let the user trigger it from the rail or add logic to auto-open.
                Toast.makeText(this, getString(R.string.common_calibration_prompt), Toast.LENGTH_LONG).show()
            }
        } else {
            showConsentDialog()
        }
    }

    private fun showConsentDialog() {
        AlertDialog.Builder(this)
            .setTitle(getString(R.string.consent_title))
            .setMessage(getString(R.string.consent_message))
            .setPositiveButton(getString(R.string.consent_agree)) { _, _ ->
                val sharedPrefs = getSharedPreferences("LipertyPrefs", Context.MODE_PRIVATE)
                with(sharedPrefs.edit()) {
                    putBoolean("consent_granted", true)
                    apply()
                }
                startCamera()
            }
            .setNegativeButton(getString(R.string.consent_decline)) { _, _ ->
                Toast.makeText(this, getString(R.string.consent_declined_message), Toast.LENGTH_LONG).show()
                finish()
            }
            .setCancelable(false)
            .show()
    }

    private fun toggleCamera() {
        if (!coreInitialized) return
        currentLensFacing = if (currentLensFacing == CameraSelector.LENS_FACING_BACK) {
            CameraSelector.LENS_FACING_FRONT
        } else {
            CameraSelector.LENS_FACING_BACK
        }
        startCamera()
    }

    private fun toggleBCMode() {
        if (!coreInitialized) return
        if (isELModeState.value) {
            isBCModeState.value = false
            toggleELMode() // Mutually exclusive
        }
        val newMode = !isBCModeState.value
        isBCModeState.value = newMode

        if (newMode) {
            isLipReadModeState.value = false
            laryngealSensor.setCarrierF0(carrierF0State.value)
            laryngealSensor.startBCMode(
                onProcessedAudio = { pcmSamples ->
                    if (::voiceManager.isInitialized) voiceManager.playAudio(pcmSamples)
                },
                onVoicingState = { isVoicing ->
                    // Visual feedback for voice activity could be added here
                },
                onTranscriptionData = { audioData ->
                    // SSR inference on the captured audio for simultaneous text display
                    if (!isInferencing) {
                        lifecycleScope.launch(Dispatchers.Default) {
                            val result = ssrInference.runInference(audioData)
                            if (result.text.isNotEmpty()) {
                                withContext(Dispatchers.Main) {
                                    transcriptionManager.appendText(result.text, result.confidence)
                                    updateTranscriptionUI()
                                    if (::voiceManager.isInitialized) voiceManager.speakStreaming(result.text)
                                }
                            }
                        }
                    }
                }
            )
            // Badge UI shows "BC" state
        } else {
            laryngealSensor.stop()
        }
    }

    private fun toggleELMode() {
        if (!coreInitialized) return
        if (isBCModeState.value) {
            isELModeState.value = false
            toggleBCMode() // Mutually exclusive
        }
        val newMode = !isELModeState.value
        isELModeState.value = newMode
        
        if (newMode) {
            isLipReadModeState.value = false
            laryngealSensor.startELMode { pcmSamples ->
                if (::voiceManager.isInitialized) voiceManager.playAudio(pcmSamples)
            }
            // Badge UI shows "EL" state
        } else {
            laryngealSensor.stop()
        }
    }

    private fun updateTranscriptionUI() {
        // Split committed vs live: committed words render bright + clickable,
        // live words render dim/italic as a real-time preview tail. The Compose
        // layer concatenates them visually but styles them differently so the
        // user sees stability at a glance.
        transcriptionWords.value = transcriptionManager.getCommittedWords()
        wordConfidences.value = transcriptionManager.getCommittedConfidences()
        liveTranscriptionWords.value = transcriptionManager.getLiveWords()
        liveWordConfidences.value = transcriptionManager.getLiveConfidences()
        selectedWordIndex.value = transcriptionManager.getSelectedWordIndex()
    }

    private fun speakText() {
        if (!::voiceManager.isInitialized) return
        // Use the text from the manager
        val text = transcriptionManager.getCurrentSentence()
        if (text.isNotEmpty()) {
            voiceManager.speak(text)
        }
    }

    private fun startCamera() {
        if (!coreInitialized) return
        // We reuse the programmatically created previewView
        val analyzer = ImageAnalysis.Analyzer { imageProxy ->
            PerformanceMonitor.logFrame()
            frameCount++
            
            val rotationDegrees = imageProxy.imageInfo.rotationDegrees

            val rawBitmap = com.hereliesaz.liperty.utils.BitmapPool.get(imageProxy.width, imageProxy.height)
            com.hereliesaz.liperty.utils.ImageUtils.imageProxyToBitmap(imageProxy, rawBitmap)
            
            // Rotate bitmap to upright orientation
            var bitmap = if (rotationDegrees != 0) {
                val rotated = com.hereliesaz.liperty.utils.ImageUtils.rotateBitmap(rawBitmap, rotationDegrees.toFloat())
                com.hereliesaz.liperty.utils.BitmapPool.recycle(rawBitmap)
                rotated
            } else {
                rawBitmap
            }

            // If front camera, mirror to match mirrored preview perception
            if (currentLensFacing == CameraSelector.LENS_FACING_FRONT) {
                val mirrored = com.hereliesaz.liperty.utils.ImageUtils.mirrorBitmap(bitmap)
                if (bitmap === rawBitmap) {
                    com.hereliesaz.liperty.utils.BitmapPool.recycle(bitmap)
                } else {
                    bitmap.recycle()
                }
                bitmap = mirrored
            }

            // Wave-to-Pause hand gesture check
            if (frameCount % 5 == 0) {
                val gesture = handGestureHelper.detectSynchronously(bitmap)
                if (gesture == HandGestureHelper.HandGesture.WAVE_PAUSE) {
                    runOnUiThread {
                        isPausedState.value = !isPausedState.value
                        // Badge UI shows "PAUSED" state
                    }
                } else if (gesture == HandGestureHelper.HandGesture.AIR_SWIPE_LEFT) {
                    runOnUiThread {
                        transcriptionManager.cycleCurrentWord(-1)
                        updateTranscriptionUI()
                    }
                } else if (gesture == HandGestureHelper.HandGesture.AIR_SWIPE_RIGHT) {
                    runOnUiThread {
                        transcriptionManager.cycleCurrentWord(1)
                        updateTranscriptionUI()
                    }
                }
            }

            if (isPausedState.value) {
                imageProxy.close()
                if (bitmap === rawBitmap) {
                    com.hereliesaz.liperty.utils.BitmapPool.recycle(bitmap)
                } else {
                    bitmap.recycle()
                }
                return@Analyzer
            }

            // Synchronous detection
            val result = faceLandmarkerHelper.detectSynchronously(bitmap)

            imageProxy.close()

            if (frameCount % 30 == 0) {
                val faceCount = result?.faceLandmarks()?.size ?: 0
                Log.d("VSRPipeline", "frame=$frameCount lipRead=${isLipReadModeState.value} " +
                    "faces=$faceCount bufSize=${frameBuffer.size()} paused=${isPausedState.value}")
            }

            if (result != null) {
                processFrame(bitmap, result)
            } else {
                runOnUiThread { overlayView.clear() }
                frameBuffer.clearAndRecycle()
                lipBoxFilter.reset()
                opticalFlowTracker.reset()
            }

            if (bitmap === rawBitmap) {
                com.hereliesaz.liperty.utils.BitmapPool.recycle(bitmap)
            } else {
                bitmap.recycle()
            }
        }

        cameraManager.startCamera(this, previewView, analyzer, currentLensFacing)
        isRecordingState.value = true
    }

    private fun processFrame(bitmap: Bitmap, result: FaceLandmarkerResult) {
        if (!isLipReadModeState.value) {
            // Landmark overlay still useful but skip expensive crop + inference
            runOnUiThread { overlayView.clear() }
            frameBuffer.clearAndRecycle()
            return
        }

        val rawLipBox = faceLandmarkerHelper.extractLipBoundingBox(result, bitmap.width, bitmap.height)
        if (rawLipBox == null) {
            Log.d("VSRPipeline", "lip box null — face detected but no lips found")
        }

        if (rawLipBox != null) {
            // OpticalFlowTracker bypassed: accumulated drift over 50 frames was causing
            // the crop box to wander off-face by inference time. Kalman-only smoothing.
            val lipBox = lipBoxFilter.update(rawLipBox)

            // Compute the expanded mouth-crop box FIRST so the overlay can
            // draw the actual cropped region (not just the lip aperture).
            // Auto-AVSR / Chaplin training data shows mouth + chin + outer
            // lips + a strip of cheek on each side. We size the crop off the
            // full face bounding box rather than the tiny lip box because:
            //   - the lip box is only the lip APERTURE (very narrow),
            //     1.6x or 2x it doesn't reliably reach the chin
            //   - face_width is invariant to mouth-open vs mouth-closed,
            //     so the crop stays the same size across frames
            // Crop side = 0.55 * face_width, centered on the lip-box center.
            val faceBox = faceLandmarkerHelper.extractFaceBoundingBox(
                result, bitmap.width, bitmap.height, marginPct = 0f
            )
            val rotation = FaceLandmarkerHelper.calculateLipRotation(result)
            val expandedLipBox = run {
                val cx = lipBox.centerX()
                val cy = lipBox.centerY()
                val side = if (faceBox != null) {
                    (faceBox.width() * 0.55f).toInt()
                } else {
                    // Fallback if face bounding box failed: use 2.5x of the
                    // longest lip-box dimension.
                    (kotlin.math.max(lipBox.width(), lipBox.height()) * 2.5f).toInt()
                }
                val half = side / 2
                android.graphics.Rect(
                    (cx - half).coerceAtLeast(0),
                    (cy - half).coerceAtLeast(0),
                    (cx + half).coerceAtMost(bitmap.width),
                    (cy + half).coerceAtMost(bitmap.height)
                )
            }

            // Build overlay data on the camera thread, post to UI thread.
            // PreviewView uses FILL_CENTER (scale to fill, crop one axis).
            // We must apply the same transform so landmarks land on the
            // correct pixels of the displayed image.
            val rawLandmarks = result.faceLandmarks().firstOrNull()
            val imgW = bitmap.width.toFloat()
            val imgH = bitmap.height.toFloat()
            runOnUiThread {
                val vw = overlayView.width.toFloat()
                val vh = overlayView.height.toFloat()
                if (vw > 0 && vh > 0 && imgW > 0 && imgH > 0) {
                    // FILL_CENTER: scale so the image covers the whole view
                    val scale = maxOf(vw / imgW, vh / imgH)
                    val offsetX = (vw - imgW * scale) / 2f
                    val offsetY = (vh - imgH * scale) / 2f
                    if (frameCount % 90 == 0) {
                        Log.d("OverlayScale", "img=${imgW.toInt()}x${imgH.toInt()} " +
                            "overlay=${vw.toInt()}x${vh.toInt()} " +
                            "scale=${"%.3f".format(scale)} " +
                            "offset=(${offsetX.toInt()},${offsetY.toInt()})")
                    }

                    val points = rawLandmarks?.map { lm ->
                        PointF(lm.x() * imgW * scale + offsetX,
                               lm.y() * imgH * scale + offsetY)
                    } ?: emptyList()
                    // Show the ACTUAL crop region on the overlay, not the
                    // raw lip aperture box. That way what the user sees on
                    // screen matches what the model is being fed.
                    val scaledBox = RectF(
                        expandedLipBox.left   * scale + offsetX,
                        expandedLipBox.top    * scale + offsetY,
                        expandedLipBox.right  * scale + offsetX,
                        expandedLipBox.bottom * scale + offsetY
                    )
                    overlayView.setLandmarks(points, scaledBox)
                }
            }

            // Head Pose
            val matrix = faceLandmarkerHelper.extractFacialTransformationMatrix(result)
            if (matrix != null) {
                 FaceLandmarkerHelper.calculateHeadPose(matrix)
            }

            // Crop & Align — Auto-AVSR was trained on 88×88 grayscale crops
            // showing the mouth + chin + outer lips + cheek strip. The
            // expanded box above was sized off the face bounding box so we
            // capture that context consistently across mouth open/closed
            // states. VSRInference handles grayscale conversion (Rec.601
            // luma) at the byte-write stage.
            val cropSize = AUTOAVSR_CROP_SIZE
            val reusableBitmap = BitmapPool.get(cropSize, cropSize)
            var processedMouth = ImageUtils.alignAndCropMouth(
                bitmap, expandedLipBox, rotation, cropSize, cropSize, reusableBitmap
            )
            // Front camera produces a horizontally-mirrored image (the user
            // sees themselves as in a mirror). Auto-AVSR / LRS3 training
            // data is NOT mirrored (subjects facing camera directly), so
            // we flip the crop horizontally on the front-facing camera so
            // the model sees mouth motion in the same chirality it was
            // trained on. If we move to the back camera later this flip
            // is unnecessary -- gated on currentLensFacing.
            if (currentLensFacing == CameraSelector.LENS_FACING_FRONT) {
                val mirrorMatrix = android.graphics.Matrix().apply { preScale(-1f, 1f) }
                val mirrored = Bitmap.createBitmap(
                    processedMouth, 0, 0,
                    processedMouth.width, processedMouth.height,
                    mirrorMatrix, true
                )
                // The reused bitmap from BitmapPool is replaced; the
                // mirrored copy is now owned by the frame buffer flow.
                BitmapPool.recycle(processedMouth)
                processedMouth = mirrored
            }

            // Diagnostic: dump a MID-buffer frame (size==8 of 16) rather than the
            // first frame, so we see what the model is being fed at the middle
            // of an utterance rather than at the boundary where detection may
            // still be settling.
            if (frameBuffer.size() == 8) {
                val pixels = IntArray(cropSize * cropSize)
                processedMouth.getPixels(pixels, 0, cropSize, 0, 0, cropSize, cropSize)
                val mean = pixels.map { android.graphics.Color.red(it) }.average()
                Log.d("VSRInput", "frame0 mean_brightness=%.1f lipBox=$lipBox " +
                    "expanded=$expandedLipBox srcBitmap=${bitmap.width}x${bitmap.height} " +
                    "rotation=${"%.1f".format(rotation)}".format(mean))

                // Dump the actual 88x88 crop the model is being fed to
                // /sdcard/Download so we can pull it via `adb pull` and
                // visually verify whether the crop shows the user's mouth.
                // One file per inference window (overwrites previous).
                try {
                    val dumpFile = java.io.File(
                        android.os.Environment.getExternalStoragePublicDirectory(
                            android.os.Environment.DIRECTORY_DOWNLOADS),
                        "liperty_mouth_crop.png"
                    )
                    dumpFile.outputStream().use { out ->
                        processedMouth.compress(Bitmap.CompressFormat.PNG, 100, out)
                    }
                    // Also dump the full source bitmap with the expanded lip box
                    // overlaid so we can see what region was cropped from where.
                    val sourceDump = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                    val canvas = android.graphics.Canvas(sourceDump)
                    val boxPaint = android.graphics.Paint().apply {
                        color = android.graphics.Color.RED
                        style = android.graphics.Paint.Style.STROKE
                        strokeWidth = 4f
                    }
                    canvas.drawRect(expandedLipBox, boxPaint)
                    // Also dot every detected lip landmark (cyan) and every
                    // face landmark (faint green) so we can see what MediaPipe
                    // actually found, not just where our box ended up.
                    rawLandmarks?.let { lms ->
                        val allPaint = android.graphics.Paint().apply {
                            color = android.graphics.Color.argb(180, 100, 255, 100)
                            style = android.graphics.Paint.Style.FILL
                        }
                        val lipPaint = android.graphics.Paint().apply {
                            color = android.graphics.Color.argb(255, 0, 255, 255)
                            style = android.graphics.Paint.Style.FILL
                        }
                        val lipSet = com.hereliesaz.liperty.ml.FaceLandmarkerHelper.LIP_INDICES.toHashSet()
                        for ((i, lm) in lms.withIndex()) {
                            val x = lm.x() * bitmap.width
                            val y = lm.y() * bitmap.height
                            canvas.drawCircle(x, y, 2f, allPaint)
                            if (i in lipSet) canvas.drawCircle(x, y, 5f, lipPaint)
                        }
                    }
                    val sourceFile = java.io.File(
                        android.os.Environment.getExternalStoragePublicDirectory(
                            android.os.Environment.DIRECTORY_DOWNLOADS),
                        "liperty_source_with_box.png"
                    )
                    sourceFile.outputStream().use { out ->
                        sourceDump.compress(Bitmap.CompressFormat.PNG, 100, out)
                    }
                    sourceDump.recycle()
                    Log.d("VSRInput", "dumped crop+source PNGs to /sdcard/Download/")
                } catch (e: Exception) {
                    Log.w("VSRInput", "crop dump failed: ${e.message}")
                }
            }

            // Pass an explicitly copied bitmap to calibration to avoid lifecycle conflicts with FrameBuffer
            calibrationCallback?.let { cb ->
                val calibrationCopy = processedMouth.copy(processedMouth.config ?: Bitmap.Config.ARGB_8888, true)
                cb.invoke(calibrationCopy)
            }

            // DO NOT RECYCLE reusableBitmap HERE. It is now owned by frameBuffer via processedMouth reference!

            // Add to Buffer
            // Extract landmarks for LipCoordNet input: 40 coordinates (20 inner lip points x 2)
            val landmarkArray = FloatArray(40)
            rawLandmarks?.let { landmarkList ->
                // MediaPipe Face Mesh lip indices (inner lip specifically, approx 20 points)
                val innerLipIndices = com.hereliesaz.liperty.ml.MLConstants.INNER_LIP_INDICES
                for (i in innerLipIndices.indices) {
                    if (innerLipIndices[i] < landmarkList.size) {
                        landmarkArray[i * 2] = landmarkList[innerLipIndices[i]].x()
                        landmarkArray[i * 2 + 1] = landmarkList[innerLipIndices[i]].y()
                    }
                }
            }
            frameBuffer.addFrame(processedMouth, landmarkArray)

            // Inference
            if (frameBuffer.isFull() && !isInferencing) {
                isInferencing = true
                Log.d("VSRPipeline", "buffer full — launching inference")
                // Sliding window: caller takes ownership of the snapshot for
                // inference + recycling, FrameBuffer retains COPIES of the most
                // recent half (8 of 16) for the next window. Smooths output
                // across utterance boundaries by giving the model overlapping
                // context instead of dropping frames whole-window at a time.
                val bufferEntries = frameBuffer.slideAndGetFrames(retainCount = AUTOAVSR_SLIDE_RETAIN)
                val framesToProcess = bufferEntries.map { it.first }
                val landmarksToProcess = FloatArray(bufferEntries.size * 40)
                bufferEntries.forEachIndexed { index, entry ->
                    entry.second?.let { lms ->
                        if (lms.size == 40) {
                            lms.copyInto(landmarksToProcess, destinationOffset = index * 40)
                        }
                    }
                }

                lifecycleScope.launch(Dispatchers.Default) {
                    val v3 = avhubertV3
                    val vsrResult = if (v3 != null) {
                        // V3 path: AV-HuBERT seq2seq. Doesn't consume the lip
                        // landmarks — the encoder takes only mouth ROIs.
                        v3.runInference(framesToProcess)
                    } else {
                        vsrInference.runInference(framesToProcess, landmarksToProcess)
                    }
                    // Once inference is complete, explicitly recycle the frames
                    for (frame in framesToProcess) {
                        BitmapPool.recycle(frame)
                    }

                    // Viseme-aware rescoring ("Chaplin's second AI" for visual ASR):
                    // try to swap visually-equivalent words that score higher in
                    // the LM context. No-op when LM has no signal (Noop or
                    // missing KenLM JNI). Runs off the UI thread; the result is
                    // typically near-identical to input until KenLM is wired up.
                    val rescoredText = visemeRescorer?.let {
                        try {
                            it.rescore(vsrResult.text)
                        } catch (e: Exception) {
                            Log.w("MainActivity", "visemeRescorer failed: ${e.message}")
                            vsrResult.text
                        }
                    } ?: vsrResult.text

                    withContext(Dispatchers.Main) {
                        val rawText = rescoredText.replace("Pred: ", "").replace(Regex("\\(.*\\)"), "")
                        Log.d("VSRPipeline", "inference done: ctc='${vsrResult.text.take(80)}' rescored='${rescoredText.take(80)}' final='$rawText' conf=${"%.2f".format(vsrResult.confidence)}")
                        if (rawText.isNotBlank()) {
                            transcriptionManager.appendText(rawText, vsrResult.confidence)
                            updateTranscriptionUI()
                        }
                        isInferencing = false
                    }
                }
            }
        } else {
            runOnUiThread { overlayView.clear() }
            frameBuffer.clearAndRecycle()
        }
    }

    private fun allPermissionsGranted(): Boolean {
        val cameraOk = ContextCompat.checkSelfPermission(
            baseContext, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
        val micOk = ContextCompat.checkSelfPermission(
            baseContext, Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
        val btOk = Build.VERSION.SDK_INT < Build.VERSION_CODES.S ||
            ContextCompat.checkSelfPermission(
                baseContext, Manifest.permission.BLUETOOTH_CONNECT
            ) == PackageManager.PERMISSION_GRANTED
        return cameraOk && micOk && btOk
    }

    override fun onDestroy() {
        super.onDestroy()
        // Per-field guards (not a single coreInitialized check): the try block
        // in onCreate may throw partway through, leaving an arbitrary prefix of
        // these fields initialized. Skipping them all would leak the ones that
        // did get constructed.
        if (::cameraManager.isInitialized) cameraManager.shutdown()
        if (::faceLandmarkerHelper.isInitialized) faceLandmarkerHelper.close()
        if (::handGestureHelper.isInitialized) handGestureHelper.close()
        if (::laryngealSensor.isInitialized) laryngealSensor.stop()
        if (::vsrInference.isInitialized) vsrInference.close()
        if (::ssrInference.isInitialized) ssrInference.close()
        if (::frameBuffer.isInitialized) frameBuffer.clearAndRecycle()
        if (::voiceManager.isInitialized) {
            voiceManager.stop()
            voiceManager.shutdown()
        }
    }

    // Removed: onInit(status: Int) is now handled by VoiceManager
}
