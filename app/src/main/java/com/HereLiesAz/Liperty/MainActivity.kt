package com.HereLiesAz.Liperty

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Rect
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.ImageAnalysis
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.HereLiesAz.Liperty.camera.CameraManager
import com.HereLiesAz.Liperty.ml.FaceLandmarkerHelper
import com.HereLiesAz.Liperty.ml.FrameBuffer
import com.HereLiesAz.Liperty.ml.VSRInference
import com.HereLiesAz.Liperty.ui.OverlayView
import com.HereLiesAz.Liperty.utils.ImageUtils
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), FaceLandmarkerHelper.FaceLandmarkerListener {

    private lateinit var cameraManager: CameraManager
    private lateinit var faceLandmarkerHelper: FaceLandmarkerHelper
    private lateinit var overlayView: OverlayView
    private lateinit var transcriptionText: TextView
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var vsrInference: VSRInference
    private lateinit var frameBuffer: FrameBuffer

    // Cached dummy bitmap for VSR placeholder to avoid garbage collection churn
    private val dummyBitmap: Bitmap by lazy {
        Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)
    }

    // Flag to prevent overlapping inference calls
    private var isInferencing = false

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (isGranted) {
                startCamera()
            } else {
                Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        overlayView = findViewById(R.id.overlay)
        transcriptionText = findViewById(R.id.text_transcription)

        cameraManager = CameraManager(this)
        faceLandmarkerHelper = FaceLandmarkerHelper(this, this)
        vsrInference = VSRInference(this)
        frameBuffer = FrameBuffer(capacity = 50) // 2 seconds at 25fps
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Initialize TFLite in background
        lifecycleScope.launch(Dispatchers.Default) {
            vsrInference.initialize()
        }

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val previewView = findViewById<PreviewView>(R.id.viewFinder)

        val analyzer = ImageAnalysis.Analyzer { imageProxy ->
            val bitmap = com.HereLiesAz.Liperty.utils.ImageUtils.imageProxyToBitmap(imageProxy)
            faceLandmarkerHelper.detectLiveStream(bitmap)
            imageProxy.close()
        }

        cameraManager.startCamera(this, previewView, analyzer)
    }

    private fun allPermissionsGranted() = ContextCompat.checkSelfPermission(
        baseContext, Manifest.permission.CAMERA
    ) == PackageManager.PERMISSION_GRANTED

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        faceLandmarkerHelper.close()
        vsrInference.close()
    }

    // FaceLandmarkerListener Implementation
    override fun onError(error: String) {
        runOnUiThread {
            Toast.makeText(this, error, Toast.LENGTH_SHORT).show()
        }
    }

    override fun onResults(result: FaceLandmarkerResult) {
        runOnUiThread {
            // Draw bounding boxes for lips/face
            val lipBox = faceLandmarkerHelper.extractLipBoundingBox(result, overlayView.width, overlayView.height)

            if (lipBox != null) {
                overlayView.setResults(emptyList(), listOf(lipBox))

                // 1. Calculate Rotation
                val rotation = FaceLandmarkerHelper.calculateLipRotation(result)

                // 2. Crop & Align (Placeholder logic using dummyBitmap as previously requested)
                val alignedMouth = ImageUtils.alignAndCropMouth(dummyBitmap, lipBox, rotation, 88)

                // 3. Preprocess
                val processedMouth = ImageUtils.applyHistogramEqualization(alignedMouth)

                // 4. Add to Buffer
                frameBuffer.addFrame(processedMouth)

                // 5. Run Inference if ready and not currently running
                if (frameBuffer.isFull() && !isInferencing) {
                    isInferencing = true
                    // Clone buffer or pass data safely to background thread
                    // For prototype, we pass the current buffer content (List copy)
                    val framesToProcess = frameBuffer.getFrames()

                    lifecycleScope.launch(Dispatchers.Default) {
                        val vsrResult = vsrInference.runInference(framesToProcess)

                        withContext(Dispatchers.Main) {
                            transcriptionText.text = vsrResult.text
                            isInferencing = false
                            // Optional: Clear buffer or slide window?
                            // Current logic assumes sliding window, but runInference might take time.
                            // If sliding window, we keep adding frames.
                        }
                    }
                } else if (!frameBuffer.isFull()) {
                    transcriptionText.text = "Buffering... (${frameBuffer.getFrames().size}/50)"
                }
            } else {
                overlayView.clear()
                transcriptionText.text = "No face detected"
                frameBuffer.clear()
            }
        }
    }
}
