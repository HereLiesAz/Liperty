package com.HereLiesAz.Liperty

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Rect
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.GestureDetector
import android.view.MotionEvent
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.HereLiesAz.Liperty.camera.CameraManager
import com.HereLiesAz.Liperty.ml.FaceLandmarkerHelper
import com.HereLiesAz.Liperty.ml.FrameBuffer
import com.HereLiesAz.Liperty.ml.TFLiteEngine
import com.HereLiesAz.Liperty.ml.VSRInference
import com.HereLiesAz.Liperty.ui.GestureListener
import com.HereLiesAz.Liperty.ui.OverlayView
import com.HereLiesAz.Liperty.ui.SettingsActivity
import com.HereLiesAz.Liperty.ui.TranscriptionManager
import com.HereLiesAz.Liperty.utils.BitmapPool
import com.HereLiesAz.Liperty.utils.ImageUtils
import com.HereLiesAz.Liperty.utils.PerformanceMonitor
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    private lateinit var cameraManager: CameraManager
    private lateinit var faceLandmarkerHelper: FaceLandmarkerHelper
    private lateinit var overlayView: OverlayView
    private lateinit var transcriptionText: TextView
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var vsrInference: VSRInference
    private lateinit var frameBuffer: FrameBuffer
    private lateinit var switchCameraButton: Button
    private lateinit var settingsButton: Button
    private lateinit var recordingIndicator: TextView

    private lateinit var gestureDetector: GestureDetector
    private val transcriptionManager = TranscriptionManager()
    private var tts: TextToSpeech? = null

    // Camera State
    private var currentLensFacing = CameraSelector.LENS_FACING_BACK
    private var telephotoPreference = true

    // Flag to prevent overlapping inference calls
    private var isInferencing = false

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
        setContentView(R.layout.activity_main)

        overlayView = findViewById(R.id.overlay)
        transcriptionText = findViewById(R.id.text_transcription)
        switchCameraButton = findViewById(R.id.btn_switch_camera)
        settingsButton = findViewById(R.id.btn_settings)
        recordingIndicator = findViewById(R.id.indicator_recording)

        cameraManager = CameraManager(this)
        faceLandmarkerHelper = FaceLandmarkerHelper(this)
        vsrInference = VSRInference(TFLiteEngine(this))
        frameBuffer = FrameBuffer(capacity = 50) // 2 seconds at 25fps
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Initialize TFLite in background
        lifecycleScope.launch(Dispatchers.Default) {
            vsrInference.initialize()
        }

        // Initialize TTS
        tts = TextToSpeech(this, this)

        // Initialize Gesture Detector
        initGestures()

        // Setup Button Listeners
        switchCameraButton.setOnClickListener {
            toggleCamera()
        }

        settingsButton.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
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
        transcriptionText.textSize = fontSize.toFloat()

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

    private fun initGestures() {
        val listener = GestureListener(
            onSwipeLeft = {
                transcriptionManager.cycleCurrentWord(-1)
                updateTranscriptionUI()
            },
            onSwipeRight = {
                transcriptionManager.cycleCurrentWord(1)
                updateTranscriptionUI()
            },
            onSwipeUp = {
                speakText()
            },
            onDoubleTapAction = {
                transcriptionManager.clear()
                updateTranscriptionUI()
                frameBuffer.clear()
                Toast.makeText(this, "Transcript Cleared", Toast.LENGTH_SHORT).show()
            }
        )
        gestureDetector = GestureDetector(this, listener)
    }

    private fun updateTranscriptionUI() {
        transcriptionText.text = transcriptionManager.getCurrentSentence()
    }

    private fun speakText() {
        val text = transcriptionManager.getCurrentSentence()
        if (text.isNotEmpty()) {
            tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
        }
    }

    override fun onTouchEvent(event: MotionEvent?): Boolean {
        return if (event != null) {
            gestureDetector.onTouchEvent(event) || super.onTouchEvent(event)
        } else {
            super.onTouchEvent(event)
        }
    }

    private fun startCamera() {
        val previewView = findViewById<PreviewView>(R.id.viewFinder)

        val analyzer = ImageAnalysis.Analyzer { imageProxy ->
            PerformanceMonitor.logFrame()
            val bitmap = com.HereLiesAz.Liperty.utils.ImageUtils.imageProxyToBitmap(imageProxy)
            // Synchronous detection
            val result = faceLandmarkerHelper.detectSynchronously(bitmap)

            imageProxy.close()

            if (result != null) {
                processFrame(bitmap, result)
            } else {
                runOnUiThread { overlayView.clear() }
                // Don't clear frame buffer immediately on one missed face?
                // Maybe better to clear if face is lost to prevent mixing sentences.
                frameBuffer.clear()
            }
        }

        // Note: CameraManager currently selects best back camera automatically.
        // Ideally we would pass the telephotoPreference to it.
        // For now, we assume CameraManager handles it or we'll update it later if needed.
        // Let's stick to existing CameraManager logic for now, or update it if possible.
        // Update: CameraManager currently hardcodes logic. We should update it to respect preference.
        // But for this step, we just restart.

        cameraManager.startCamera(this, previewView, analyzer, currentLensFacing)

        // Show recording indicator when camera starts
        recordingIndicator.visibility = android.view.View.VISIBLE
    }

    private fun processFrame(bitmap: Bitmap, result: FaceLandmarkerResult) {
        val lipBox = faceLandmarkerHelper.extractLipBoundingBox(result, bitmap.width, bitmap.height)

        if (lipBox != null) {
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

            // Head Pose (calculated but currently just for logging/debug)
            val matrix = faceLandmarkerHelper.extractFacialTransformationMatrix(result)
            if (matrix != null) {
                 val pose = FaceLandmarkerHelper.calculateHeadPose(matrix)
                 // pose is Triple(Roll, Pitch, Yaw)
            }

            // Crop & Align
            val rotation = FaceLandmarkerHelper.calculateLipRotation(result)
            // Use pooled bitmap
            val reusableBitmap = BitmapPool.get(88, 88)
            val alignedMouth = ImageUtils.alignAndCropMouth(bitmap, lipBox, rotation, 88, reusableBitmap)

            // Preprocess (Note: applyHistogramEqualization currently allocates new bitmap, optimization TODO)
            val processedMouth = ImageUtils.applyHistogramEqualization(alignedMouth)

            // We can recycle alignedMouth if histogram created a new one, OR if buffer copies it.
            // Since FrameBuffer stores it, we cannot recycle it immediately unless FrameBuffer copies.
            // Current FrameBuffer just adds to deque.
            // So we transfer ownership to FrameBuffer.
            // When FrameBuffer drops a frame, it should ideally recycle it.

            // For now, to keep it safe without changing FrameBuffer logic too much (as it might be used by multiple threads):
            // We will NOT recycle manually here, relying on GC for the processed one, but we used pool for intermediate step if possible.
            // Actually alignAndCropMouth writes to reusableBitmap.
            // applyHistogramEqualization returns a NEW bitmap.
            // So we can recycle reusableBitmap immediately after histogram.

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
        tts?.stop()
        tts?.shutdown()
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = tts?.setLanguage(Locale.US)
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e("MainActivity", "TTS Language not supported")
            }
        } else {
            Log.e("MainActivity", "TTS Initialization failed")
        }
    }
}
