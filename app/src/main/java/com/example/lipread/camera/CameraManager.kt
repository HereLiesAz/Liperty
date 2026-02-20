package com.example.lipread.camera

import android.content.Context
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.Executors

class CameraManager(private val context: Context) {

    private val executor = Executors.newSingleThreadExecutor()

    fun startCamera(
        lifecycleOwner: LifecycleOwner,
        previewView: PreviewView,
        analyzer: ImageAnalysis.Analyzer
    ) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // Select Lens: Prefer Telephoto for Rear Camera to reduce distortion (Research)
            val cameraSelector = selectBestCamera(cameraProvider)

            // Preview Use Case
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            // Image Analysis Use Case
            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(executor, analyzer)
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    preview,
                    imageAnalysis
                )
            } catch (exc: Exception) {
                Log.e("CameraManager", "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(context))
    }

    /**
     * Attempts to find a Telephoto lens if available, otherwise defaults to standard back camera.
     * Research indicates telephoto lenses reduce perspective distortion ("moustache distortion").
     */
    private fun selectBestCamera(cameraProvider: ProcessCameraProvider): CameraSelector {
        // This is a simplified selection logic. In production, you'd iterate through available cameras
        // and check camera characteristics for LENS_INFO_AVAILABLE_FOCAL_LENGTHS.

        // For now, we default to DEFAULT_BACK_CAMERA but leave a placeholder for explicit telephoto selection.
        return CameraSelector.DEFAULT_BACK_CAMERA
    }
}
