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

    // Compose State
    private val transcriptionState = mutableStateOf("")
    private val isRecordingState = mutableStateOf(false)

    // Camera State
    private var currentLensFacing = CameraSelector.LENS_FACING_BACK
    private var telephotoPreference = true

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
                transcriptionText = transcriptionState.value,
                onTextChange = { newText ->
                    // Update manager but maybe don't overwrite if it's correction?
                    // For now, let's assume direct edit replaces current sentence or word.
                    // But TranscriptionManager logic is complex.
                    // We'll update state directly for UI feedback, and maybe sync with manager.
                    // Ideally, TranscriptionManager should be the source of truth.
                    // For now, update state.
                    transcriptionState.value = newText
                    // If we want to support full editing, we might need to update the manager's buffer.
                    // Let's assume for now simple text update.
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
                isPaused = isPausedState.value
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

        // Font Size
        val fontSize = sharedPrefs.getInt("font_size", 20)
        // transcriptionText.textSize = fontSize.toFloat() // Handled by Compose now?
        // Note: Compose AzTextBox might need font size param.
        // For now, we skip dynamic font size update in Compose unless we pass it.

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

        if (consentGranted) {
            startCamera()
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

    private fun updateTranscriptionUI() {
        transcriptionState.value = transcriptionManager.getCurrentSentence()
    }

    private fun speakText() {
        // Use the text from the state as it might have been edited by the user
        val text = transcriptionState.value
        if (text.isNotEmpty()) {
            voiceManager.speak(text)
        }
    }

    private fun startCamera() {
        // We reuse the programmatically created previewView
        val analyzer = ImageAnalysis.Analyzer { imageProxy ->
            PerformanceMonitor.logFrame()
            frameCount++
            
            // Note: CameraX 1.3.0+ toBitmap() ALREADY applies rotationDegrees.
            // Manual rotation here causes double-rotation (e.g. 270+270 = 180/upside down).
            val bitmap = com.hereliesaz.liperty.utils.ImageUtils.imageProxyToBitmap(imageProxy)
            
            // Wave-to-Pause hand gesture check
            if (frameCount % 5 == 0) {
                val gesture = handGestureHelper.detectSynchronously(bitmap)
                if (gesture == HandGestureHelper.HandGesture.WAVE_PAUSE) {
                    isPausedState.value = !isPausedState.value
                    runOnUiThread {
                        val msg = if (isPausedState.value) "Paused" else "Resumed"
                        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
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
