package com.HereLiesAz.Liperty.camera

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.util.Log
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
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

    fun shutdown() {
        executor.shutdown()
    }

    /**
     * Starts the camera with the specified lens facing.
     * @param lensFacing Use CameraSelector.LENS_FACING_BACK (default) or LENS_FACING_FRONT
     */
    fun startCamera(
        lifecycleOwner: LifecycleOwner,
        previewView: PreviewView,
        analyzer: ImageAnalysis.Analyzer,
        lensFacing: Int = CameraSelector.LENS_FACING_BACK
    ) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // Select Lens
            val cameraSelector = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
                selectBestBackCamera(cameraProvider)
            } else {
                CameraSelector.DEFAULT_FRONT_CAMERA
            }

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
    @androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
    private fun selectBestBackCamera(cameraProvider: ProcessCameraProvider): CameraSelector {
        var bestCameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
        var maxFocalLength = 0f
        var foundTelephoto = false

        for (cameraInfo in cameraProvider.availableCameraInfos) {
            try {
                val camera2CameraInfo = Camera2CameraInfo.from(cameraInfo)
                val lensFacing = camera2CameraInfo.getCameraCharacteristic(CameraCharacteristics.LENS_FACING)

                if (lensFacing == CameraCharacteristics.LENS_FACING_BACK) {
                    val focalLengths = camera2CameraInfo.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)

                    if (focalLengths != null && focalLengths.isNotEmpty()) {
                        // Get the maximum focal length for this camera (telephoto has larger focal length)
                        val currentFocalLength = focalLengths.maxOrNull() ?: 0f

                        // If this camera has a longer focal length than what we've seen, pick it.
                        if (currentFocalLength > maxFocalLength) {
                            maxFocalLength = currentFocalLength
                            // Create a selector specifically for this camera
                            bestCameraSelector = CameraSelector.Builder()
                                .addCameraFilter { _ -> listOf(cameraInfo) }
                                .build()
                            foundTelephoto = true
                        }
                    }
                }
            } catch (e: Exception) {
                Log.w("CameraManager", "Failed to inspect camera info: ${e.message}")
            }
        }

        if (foundTelephoto) {
            Log.i("CameraManager", "Selected Telephoto Camera with focal length: $maxFocalLength")
        } else {
            Log.i("CameraManager", "No dedicated Telephoto found, using default back camera.")
        }

        return bestCameraSelector
    }
}
