package com.example.lipread

import android.Manifest
import android.content.pm.PackageManager
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
import com.example.lipread.camera.CameraManager
import com.example.lipread.ml.FaceLandmarkerHelper
import com.example.lipread.ui.OverlayView
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), FaceLandmarkerHelper.FaceLandmarkerListener {

    private lateinit var cameraManager: CameraManager
    private lateinit var faceLandmarkerHelper: FaceLandmarkerHelper
    private lateinit var overlayView: OverlayView
    private lateinit var transcriptionText: TextView
    private lateinit var cameraExecutor: ExecutorService

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
            // Convert ImageProxy to Bitmap for MediaPipe (Note: This is slow, optimize later)
            // For now, assume we have a utility or can use the bitmap directly if format supports it
            // Or use imageProxy directly if FaceLandmarker supports it (it usually takes MPImage)

            val bitmap = com.example.lipread.utils.ImageUtils.imageProxyToBitmap(imageProxy)
            faceLandmarkerHelper.detectLiveStream(bitmap, false) // false = rear camera default

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
            // This is simplified. Actual mapping from normalized coordinates to View coordinates needed.
            // val rects = result.faceLandmarks().map { ... }
            // overlayView.setResults(rects)

            // Here we would also feed the cropped lip region to the VSR model
            transcriptionText.text = "Tracking ${result.faceLandmarks().size} faces..."
        }
    }
}
