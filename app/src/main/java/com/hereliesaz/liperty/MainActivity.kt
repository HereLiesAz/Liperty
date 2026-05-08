package com.hereliesaz.liperty

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
 import android.graphics.PointF
import android.graphics.RectF
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
import com.hereliesaz.liperty.voicebox.ArtificialLarynx
import com.hereliesaz.liperty.ml.FrameBuffer
import com.hereliesaz.liperty.ml.TFLiteEngine
import com.hereliesaz.liperty.ml.VSRInference
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
import com.hereliesaz.liperty.voicebox.recording.VoiceRecorder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {

    private lateinit var cameraManager: CameraManager
    private lateinit var faceLandmarkerHelper: FaceLandmarkerHelper
    private lateinit var handGestureHelper: HandGestureHelper
    private lateinit var overlayView: OverlayView
    private lateinit var previewView: PreviewView
    private lateinit var vsrInference: VSRInference
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
    private val selectedWordIndex = mutableStateOf(-1)
    private val isRecordingState = mutableStateOf(false)
    private val isSSIModeState = mutableStateOf(false)
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

        ImageUtils.initializeOpenCV(this)

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
        // vsr_lora_model.tflite uses ops incompatible with LiteRT 2.x CompiledModel.
        // Always load the base model until LoRA export is fixed.
        val vsrEngine = TFLiteEngine(this, "vallr_model.tflite")
        
        vsrInference = VSRInference(vsrEngine)
        ssrInference = SSRInference(TFLiteEngine(this, "ssr_model.tflite"))
        // VoiceConverter is initialized lazily or in onCreate
        // For simplicity, let's use the one in LaryngealSensor if EL mode is active.
        // But MainActivity might need its own if it does other things.
        frameBuffer = FrameBuffer(capacity = 50) // VSR model input: [1, 50, 64, 128, 3] and landmarks: [1, 50, 40]

        // Initialize TFLite in background
        lifecycleScope.launch(Dispatchers.Default) {
            vsrInference.initialize()
            ssrInference.initialize()
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
                wordConfidences = wordConfidences.value,
                isRecording = isRecordingState.value,
                onSwitchCamera = { toggleCamera() },
                onOpenSettings = { startActivity(Intent(this, SettingsActivity::class.java)) },
                onClearTranscript = {
                    transcriptionManager.clear()
                    updateTranscriptionUI()
                    frameBuffer.clearAndRecycle()
                    Toast.makeText(this, "Transcript Cleared", Toast.LENGTH_SHORT).show()
                },
                onSpeak = { speakText() },
                onToggleSSI = { toggleSSIMode() },
                onToggleLipRead = { isLipReadModeState.value = !isLipReadModeState.value },
                onToggleEL = { toggleELMode() },
                isPaused = isPausedState.value,
                isSSIActive = isSSIModeState.value,
                isLipReadActive = isLipReadModeState.value,
                isELActive = isELModeState.value,
                currentLensFacing = if (currentLensFacing == CameraSelector.LENS_FACING_BACK) 1 else 0,
                vsrSensitivity = vsrSensitivity.value,
                onVsrSensitivityChange = { value ->
                    vsrSensitivity.value = value
                    faceLandmarkerHelper.updateConfidence(value, value)
                },
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
        if (isELModeState.value) {
            isSSIModeState.value = false
            toggleELMode() // Mutually exclusive
        }
        val newMode = !isSSIModeState.value
        isSSIModeState.value = newMode
        
        if (newMode) {
            isLipReadModeState.value = false // Focus on SSI
            laryngealSensor.start(
                onProcessedAudio = { pcmSamples -> voiceManager.playAudio(pcmSamples) },
                onVoicingState = { isVoicing ->
                    // Optionally show visual feedback for contact detection
                },
                onVibrationData = { vibrationSignal ->
                    // Phase 9: SSR Inference
                    if (!isInferencing) {
                        lifecycleScope.launch(Dispatchers.Default) {
                            val result = ssrInference.runInference(vibrationSignal)
                            if (result.text.isNotEmpty()) {
                                withContext(Dispatchers.Main) {
                                    transcriptionManager.appendText(result.text, result.confidence)
                                    updateTranscriptionUI()
                                    // Ultra-low latency synthesis:
                                    voiceManager.speakStreaming(result.text)
                                }
                            }
                        }
                    }
                }
            )
            Toast.makeText(this, "Voice Box Mode Active: Press against throat.", Toast.LENGTH_LONG).show()
        } else {
            laryngealSensor.stop()
            Toast.makeText(this, "Voice Box Mode Inactive", Toast.LENGTH_SHORT).show()
        }
    }

    private fun toggleELMode() {
        if (isSSIModeState.value) {
            isELModeState.value = false
            toggleSSIMode() // Mutually exclusive
        }
        val newMode = !isELModeState.value
        isELModeState.value = newMode
        
        if (newMode) {
            isLipReadModeState.value = false
            laryngealSensor.startELMode { pcmSamples -> voiceManager.playAudio(pcmSamples) }
            Toast.makeText(this, "EL Translator Active", Toast.LENGTH_LONG).show()
        } else {
            laryngealSensor.stop()
            Toast.makeText(this, "EL Translator Inactive", Toast.LENGTH_SHORT).show()
        }
    }

    private fun updateTranscriptionUI() {
        transcriptionWords.value = transcriptionManager.getWords()
        wordConfidences.value = transcriptionManager.getWordConfidences()
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
                    val scaledBox = RectF(
                        lipBox.left   * scale + offsetX,
                        lipBox.top    * scale + offsetY,
                        lipBox.right  * scale + offsetX,
                        lipBox.bottom * scale + offsetY
                    )
                    overlayView.setLandmarks(points, scaledBox)
                }
            }

            // Head Pose
            val matrix = faceLandmarkerHelper.extractFacialTransformationMatrix(result)
            if (matrix != null) {
                 FaceLandmarkerHelper.calculateHeadPose(matrix)
            }

            // Crop & Align — crop to 128x64 to match the model's native input size directly,
            // avoiding a second redundant scale step inside VSRInference.
            val rotation = FaceLandmarkerHelper.calculateLipRotation(result)
            val cropWidth = 128
            val cropHeight = 64
            val reusableBitmap = BitmapPool.get(cropWidth, cropHeight)
            val alignedMouth = ImageUtils.alignAndCropMouth(bitmap, lipBox, rotation, cropWidth, cropHeight, reusableBitmap)

            // Normalization bypassed: applyNormalizationNative (JNI) was modifying frames
            // in-place in a way that may make all frames identical (histogram equalization
            // collapses per-frame contrast; native implementation is opaque). Pass raw crop.
            val processedMouth = alignedMouth

            // Diagnostic: log mean pixel brightness of first frame in each batch to confirm input varies
            if (frameBuffer.size() == 0) {
                val pixels = IntArray(cropWidth * cropHeight)
                processedMouth.getPixels(pixels, 0, cropWidth, 0, 0, cropWidth, cropHeight)
                val mean = pixels.map { android.graphics.Color.red(it) }.average()
                Log.d("VSRInput", "frame0 mean_brightness=%.1f lipBox=$rawLipBox".format(mean))
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
                // Takes ownership of the frames
                val bufferEntries = frameBuffer.clearAndGetFrames()
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
                    val vsrResult = vsrInference.runInference(framesToProcess, landmarksToProcess)
                    // Once inference is complete, explicitly recycle the frames
                    for (frame in framesToProcess) {
                        BitmapPool.recycle(frame)
                    }

                    withContext(Dispatchers.Main) {
                        val rawText = vsrResult.text.replace("Pred: ", "").replace(Regex("\\(.*\\)"), "")
                        Log.d("VSRPipeline", "inference done: raw='${vsrResult.text.take(80)}' filtered='$rawText' conf=${"%.2f".format(vsrResult.confidence)}")
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

    private fun allPermissionsGranted() = ContextCompat.checkSelfPermission(
        baseContext, Manifest.permission.CAMERA
    ) == PackageManager.PERMISSION_GRANTED

    override fun onDestroy() {
        super.onDestroy()
        cameraManager.shutdown()
        faceLandmarkerHelper.close()
        handGestureHelper.close()
        laryngealSensor.stop()
        vsrInference.close()
        ssrInference.close()
        voiceManager.stop()
        voiceManager.shutdown()
    }

    // Removed: onInit(status: Int) is now handled by VoiceManager
}
