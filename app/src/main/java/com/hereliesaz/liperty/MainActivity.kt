package com.hereliesaz.liperty

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Rect
import android.os.Bundle
import android.speech.tts.TextToSpeech
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
import com.hereliesaz.liperty.voicebox.ArtificialLarynx
import com.hereliesaz.liperty.ml.FrameBuffer
import com.hereliesaz.liperty.ml.TFLiteEngine
import com.hereliesaz.liperty.ml.VSRInference
import com.hereliesaz.liperty.ui.LipertyApp
import com.hereliesaz.liperty.ui.OverlayView
import com.hereliesaz.liperty.ui.SettingsActivity
import com.hereliesaz.liperty.ui.TranscriptionManager
import com.hereliesaz.liperty.utils.BitmapPool
import com.hereliesaz.liperty.utils.ImageUtils
import com.hereliesaz.liperty.utils.PerformanceMonitor
import com.hereliesaz.liperty.utils.RectKalmanFilter
import com.hereliesaz.liperty.voicebox.VoiceManager
import com.hereliesaz.liperty.voicebox.recording.VoiceRecorder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private lateinit var cameraManager: CameraManager
    private lateinit var faceLandmarkerHelper: FaceLandmarkerHelper
    private lateinit var handGestureHelper: HandGestureHelper
    private lateinit var overlayView: OverlayView
    private lateinit var previewView: PreviewView
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var vsrInference: VSRInference
    private lateinit var frameBuffer: FrameBuffer
    private val lipBoxFilter = RectKalmanFilter()

    private val transcriptionManager by lazy { TranscriptionManager(this) }
    private lateinit var voiceManager: VoiceManager
    private lateinit var laryngealSensor: LaryngealSensor

    // Compose State
    private val transcriptionWords = mutableStateOf(listOf<String>())
    private val selectedWordIndex = mutableStateOf(-1)
    private val isRecordingState = mutableStateOf(false)
    private val isSSIModeState = mutableStateOf(false)
    private val isLipReadModeState = mutableStateOf(true)

    // Camera State
    private var currentLensFacing = CameraSelector.LENS_FACING_FRONT
    private var telephotoPreference = true

    // Sensitivity State
    private val vsrSensitivity = mutableStateOf(0.5f)
    private val larynxSensitivity = mutableStateOf(0.5f)
    private val isDarkTheme = mutableStateOf(true)

    // Flag to prevent overlapping inference calls
    private var isInferencing = false
    private val isPausedState = mutableStateOf(false)
    private var frameCount = 0
    private var calibrationCallback: ((Bitmap) -> Unit)? = null

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (isGranted) {
                checkConsentAndStart()
            } else {
                Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize Views programmatically for Compose AndroidView
        previewView = PreviewView(this)
        overlayView = OverlayView(this, null).apply {
             layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        }

        cameraManager = CameraManager(this)
        faceLandmarkerHelper = FaceLandmarkerHelper(this)
        handGestureHelper = HandGestureHelper(this)
        laryngealSensor = LaryngealSensor(this)
        vsrInference = VSRInference(TFLiteEngine(this))
        frameBuffer = FrameBuffer(capacity = 50) // 2 seconds at 25fps
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Initialize TFLite in background
        lifecycleScope.launch(Dispatchers.Default) {
            vsrInference.initialize()
        }

        // Initialize VoiceManager
        voiceManager = VoiceManager(this) { ready ->
            if (ready) {
                Log.i("MainActivity", "VoiceManager initialized.")
            } else {
                Log.e("MainActivity", "VoiceManager initialization failed.")
            }
        }

        setContent {
            LipertyApp(
                previewView = previewView,
                overlayView = overlayView,
                transcriptionWords = transcriptionWords.value,
                selectedWordIndex = selectedWordIndex.value,
                onWordClick = { index ->
                    transcriptionManager.selectWord(index)
                    updateTranscriptionUI()
                },
                isRecording = isRecordingState.value,
                onSwitchCamera = { toggleCamera() },
                onOpenSettings = { startActivity(Intent(this, SettingsActivity::class.java)) },
                onClearTranscript = {
                    transcriptionManager.clear()
                    updateTranscriptionUI()
                    frameBuffer.clear()
                    Toast.makeText(this, "Transcript Cleared", Toast.LENGTH_SHORT).show()
                },
                onSpeak = { speakText() },
                onToggleSSI = { toggleSSIMode() },
                onToggleLipRead = { isLipReadModeState.value = !isLipReadModeState.value },
                isPaused = isPausedState.value,
                isSSIActive = isSSIModeState.value,
                isLipReadActive = isLipReadModeState.value,
                currentLensFacing = if (currentLensFacing == CameraSelector.LENS_FACING_BACK) 1 else 0,
                vsrSensitivity = vsrSensitivity.value,
                onVsrSensitivityChange = { vsrSensitivity.value = it },
                larynxSensitivity = larynxSensitivity.value,
                onLarynxSensitivityChange = { 
                    larynxSensitivity.value = it
                    laryngealSensor.setSensitivity(it)
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
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onResume() {
        super.onResume()
        applySettings()
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
        val sharedPrefs = getSharedPreferences("LipertyPrefs", Context.MODE_PRIVATE)
        val consentGranted = sharedPrefs.getBoolean("consent_granted", false)
        val calibrationDone = sharedPrefs.getBoolean("calibration_complete", false)

        if (consentGranted) {
            startCamera()
            if (!calibrationDone) {
                // Ideally we should navigate to calibration route here.
                // For now, let the user trigger it from the rail or add logic to auto-open.
                Toast.makeText(this, "Please personalize the model for better accuracy.", Toast.LENGTH_LONG).show()
            }
        } else {
            showConsentDialog()
        }
    }

    private fun showConsentDialog() {
        AlertDialog.Builder(this)
            .setTitle("Legal Consent Required")
            .setMessage("This application records video and processes facial landmarks to provide visual speech recognition.\n\n" +
                    "By proceeding, you consent to the real-time processing of your biometric data on this device. " +
                    "No data is sent to the cloud or permanently stored without your explicit action.\n\n" +
                    "Do you agree?")
            .setPositiveButton("I Agree") { _, _ ->
                val sharedPrefs = getSharedPreferences("LipertyPrefs", Context.MODE_PRIVATE)
                with(sharedPrefs.edit()) {
                    putBoolean("consent_granted", true)
                    apply()
                }
                startCamera()
            }
            .setNegativeButton("Decline") { _, _ ->
                Toast.makeText(this, "Consent declined. App cannot function.", Toast.LENGTH_LONG).show()
                finish()
            }
            .setCancelable(false)
            .show()
    }

    private fun toggleCamera() {
        currentLensFacing = if (currentLensFacing == CameraSelector.LENS_FACING_BACK) {
            CameraSelector.LENS_FACING_FRONT
        } else {
            CameraSelector.LENS_FACING_BACK
        }
        startCamera()
    }

    private fun toggleSSIMode() {
        val newMode = !isSSIModeState.value
        isSSIModeState.value = newMode
        
        if (newMode) {
            // SSI Mode: Phone vibrates, acts as sound source
            laryngealSensor.start(
                onProcessedAudio = { /* Real-time audio stream could be played or sent to ML */ },
                onVoicingState = { isVoicing ->
                    // Optionally show visual feedback for contact detection
                }
            )
            Toast.makeText(this, "Voice Box Mode Active: Press against throat.", Toast.LENGTH_LONG).show()
        } else {
            laryngealSensor.stop()
            Toast.makeText(this, "Voice Box Mode Inactive", Toast.LENGTH_SHORT).show()
        }
    }

    private fun updateTranscriptionUI() {
        transcriptionWords.value = transcriptionManager.getWords()
        selectedWordIndex.value = transcriptionManager.getSelectedWordIndex()
    }

    private fun speakText() {
        // Use the text from the manager
        val text = transcriptionManager.getCurrentSentence()
        if (text.isNotEmpty()) {
            voiceManager.speak(text)
        }
    }

    private fun startCamera() {
        // We reuse the programmatically created previewView
        val analyzer = ImageAnalysis.Analyzer { imageProxy ->
            PerformanceMonitor.logFrame()
            frameCount++
            
            val rotationDegrees = imageProxy.imageInfo.rotationDegrees
            val rawBitmap = com.hereliesaz.liperty.utils.ImageUtils.imageProxyToBitmap(imageProxy)
            
            // Rotate bitmap to upright orientation
            var bitmap = if (rotationDegrees != 0) {
                val rotated = com.hereliesaz.liperty.utils.ImageUtils.rotateBitmap(rawBitmap, rotationDegrees.toFloat())
                rawBitmap.recycle()
                rotated
            } else {
                rawBitmap
            }

            // If front camera, mirror to match mirrored preview perception
            if (currentLensFacing == CameraSelector.LENS_FACING_FRONT) {
                val mirrored = com.hereliesaz.liperty.utils.ImageUtils.mirrorBitmap(bitmap)
                bitmap.recycle()
                bitmap = mirrored
            }

            // Wave-to-Pause hand gesture check
            if (frameCount % 5 == 0) {
                val gesture = handGestureHelper.detectSynchronously(bitmap)
                if (gesture == HandGestureHelper.HandGesture.WAVE_PAUSE) {
                    isPausedState.value = !isPausedState.value
                    runOnUiThread {
                        val msg = if (isPausedState.value) "Paused" else "Resumed"
                        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
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
                bitmap.recycle()
                return@Analyzer
            }

            // Synchronous detection
            val result = faceLandmarkerHelper.detectSynchronously(bitmap)

            imageProxy.close()

            if (result != null) {
                processFrame(bitmap, result)
            } else {
                runOnUiThread { overlayView.clear() }
                frameBuffer.clear()
                lipBoxFilter.reset()
            }
        }

        cameraManager.startCamera(this, previewView, analyzer, currentLensFacing)
        isRecordingState.value = true
    }

    private fun processFrame(bitmap: Bitmap, result: FaceLandmarkerResult) {
        val rawLipBox = faceLandmarkerHelper.extractLipBoundingBox(result, bitmap.width, bitmap.height)

        if (rawLipBox != null) {
            val lipBox = lipBoxFilter.update(rawLipBox)

            runOnUiThread {
                val scaleX = overlayView.width.toFloat() / bitmap.width
                val scaleY = overlayView.height.toFloat() / bitmap.height

                val scaledRect = Rect(
                    (lipBox.left * scaleX).toInt(),
                    (lipBox.top * scaleY).toInt(),
                    (lipBox.right * scaleX).toInt(),
                    (lipBox.bottom * scaleY).toInt()
                )
                overlayView.setResults(emptyList(), listOf(scaledRect))
            }

            // Head Pose
            val matrix = faceLandmarkerHelper.extractFacialTransformationMatrix(result)
            if (matrix != null) {
                 FaceLandmarkerHelper.calculateHeadPose(matrix)
            }

            // Crop & Align
            val rotation = FaceLandmarkerHelper.calculateLipRotation(result)
            val reusableBitmap = BitmapPool.get(88, 88)
            val alignedMouth = ImageUtils.alignAndCropMouth(bitmap, lipBox, rotation, 88, reusableBitmap)
            
            // Optimized JNI Normalization (Blur + Histogram Equalization)
            val processedMouth = ImageUtils.normalizeForInference(alignedMouth)

            // Pass to calibration if active
            calibrationCallback?.invoke(processedMouth)

            BitmapPool.recycle(reusableBitmap)

            // Add to Buffer
            frameBuffer.addFrame(processedMouth)

            // Inference
            if (frameBuffer.isFull() && !isInferencing) {
                isInferencing = true
                val framesToProcess = frameBuffer.getFrames()

                lifecycleScope.launch(Dispatchers.Default) {
                    val vsrResult = vsrInference.runInference(framesToProcess)

                    withContext(Dispatchers.Main) {
                        val rawText = vsrResult.text.replace("Pred: ", "").replace(Regex("\\(.*\\)"), "")
                        if (rawText.isNotBlank()) {
                            transcriptionManager.appendText(rawText)
                            updateTranscriptionUI()
                        }
                        isInferencing = false
                        frameBuffer.clear()
                    }
                }
            }
        } else {
            runOnUiThread { overlayView.clear() }
            frameBuffer.clear()
        }
    }

    private fun allPermissionsGranted() = ContextCompat.checkSelfPermission(
        baseContext, Manifest.permission.CAMERA
    ) == PackageManager.PERMISSION_GRANTED

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        cameraManager.shutdown()
        faceLandmarkerHelper.close()
        vsrInference.close()
        voiceManager.stop()
        voiceManager.shutdown()
    }

    // Removed: onInit(status: Int) is now handled by VoiceManager
}
