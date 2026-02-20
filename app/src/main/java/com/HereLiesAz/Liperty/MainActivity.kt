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
import com.HereLiesAz.Liperty.camera.CameraManager
import com.HereLiesAz.Liperty.ml.FaceLandmarkerHelper
import com.HereLiesAz.Liperty.ml.VSRInference
import com.HereLiesAz.Liperty.ui.OverlayView
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), FaceLandmarkerHelper.FaceLandmarkerListener {

    private lateinit var cameraManager: CameraManager
    private lateinit var faceLandmarkerHelper: FaceLandmarkerHelper
    private lateinit var overlayView: OverlayView
    private lateinit var transcriptionText: TextView
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var vsrInference: VSRInference

    // Cached dummy bitmap for VSR placeholder to avoid garbage collection churn
    private val dummyBitmap: Bitmap by lazy {
        Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
    }

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
        cameraExecutor = Executors.newSingleThreadExecutor()

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val previewView = findViewById<PreviewView>(R.id.viewFinder)

        // Setup Image Analyzer to pipe frames to MediaPipe
        val analyzer = ImageAnalysis.Analyzer { imageProxy ->
            // Convert ImageProxy to Bitmap using the optimized CameraX extension (via ImageUtils)
            val bitmap = com.HereLiesAz.Liperty.utils.ImageUtils.imageProxyToBitmap(imageProxy)

            // Note: For actual VSR, we would crop the lip region here and convert to grayscale.
            // We should reuse a shared Bitmap for grayscale conversion to avoid allocation churn.

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
        // Explicitly close FaceLandmarker to release native resources and GPU delegates
        faceLandmarkerHelper.close()
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
            // We use overlayView dimensions to scale the normalized landmarks to screen coordinates
            val lipBox = faceLandmarkerHelper.extractLipBoundingBox(result, overlayView.width, overlayView.height)

            if (lipBox != null) {
                overlayView.setResults(emptyList(), listOf(lipBox))

                // Dummy VSR call with placeholder bitmap
                val vsrResult = vsrInference.runInference(dummyBitmap)
                transcriptionText.text = vsrResult.text
            } else {
                overlayView.clear()
                transcriptionText.text = "No face detected"
            }
        }
    }
}
