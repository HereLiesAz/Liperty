package com.HereLiesAz.Liperty.ml

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult

class FaceLandmarkerHelper(
    val context: Context,
    val faceLandmarkerListener: FaceLandmarkerListener
) {
    private var faceLandmarker: FaceLandmarker? = null

    init {
        setupFaceLandmarker()
    }

    private fun setupFaceLandmarker() {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("face_landmarker.task") // This file must be in assets/
            .build()

        val options = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setMinFaceDetectionConfidence(0.5f)
            .setMinFaceTrackingConfidence(0.5f)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setResultListener(this::returnLivestreamResult)
            .setErrorListener(this::returnLivestreamError)
            .build()

        try {
            faceLandmarker = FaceLandmarker.createFromOptions(context, options)
        } catch (e: IllegalStateException) {
            faceLandmarkerListener.onError("Face Landmarker failed to initialize: ${e.message}")
        }
    }

    fun detectLiveStream(imageBitmap: Bitmap) {
        if (faceLandmarker == null) return

        val frameTime = SystemClock.uptimeMillis()

        // Convert Bitmap to MP Image
        val mpImage = BitmapImageBuilder(imageBitmap).build()

        faceLandmarker?.detectAsync(mpImage, frameTime)
    }

    private fun returnLivestreamResult(
        result: FaceLandmarkerResult,
        input: com.google.mediapipe.framework.image.MPImage
    ) {
        faceLandmarkerListener.onResults(result)
    }

    private fun returnLivestreamError(error: RuntimeException) {
        faceLandmarkerListener.onError(error.message ?: "Unknown error")
    }

    fun close() {
        faceLandmarker?.close()
        faceLandmarker = null
    }

    interface FaceLandmarkerListener {
        fun onError(error: String)
        fun onResults(result: FaceLandmarkerResult)
    }
}
