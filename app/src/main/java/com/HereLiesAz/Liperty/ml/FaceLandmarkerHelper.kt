package com.HereLiesAz.Liperty.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.os.SystemClock
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import java.util.Optional

class FaceLandmarkerHelper(
    val context: Context,
    val faceLandmarkerListener: FaceLandmarkerListener
) {
    private var faceLandmarker: FaceLandmarker? = null

    init {
        setupFaceLandmarker()
    }

    private fun setupFaceLandmarker() {
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("face_landmarker.task") // This file must be in assets/
                .build()

            val options = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setMinFaceDetectionConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setOutputFacialTransformationMatrixes(true)
                .setResultListener(this::returnLivestreamResult)
                .setErrorListener(this::returnLivestreamError)
                .build()

            faceLandmarker = FaceLandmarker.createFromOptions(context, options)
        } catch (e: Exception) {
            faceLandmarkerListener.onError("Face Landmarker failed to initialize: ${e.message}")
            Log.e("FaceLandmarkerHelper", "Error initializing FaceLandmarker", e)
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
        input: MPImage
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

    fun extractLipBoundingBox(result: FaceLandmarkerResult, imageWidth: Int, imageHeight: Int): Rect? {
        if (result.faceLandmarks().isEmpty()) return null
        val landmarks = result.faceLandmarks()[0] // Use first face

        var minX = Float.MAX_VALUE
        var minY = Float.MAX_VALUE
        var maxX = Float.MIN_VALUE
        var maxY = Float.MIN_VALUE

        for (index in LIP_INDICES) {
            if (index < landmarks.size) {
                val landmark = landmarks[index]
                minX = minOf(minX, landmark.x())
                minY = minOf(minY, landmark.y())
                maxX = maxOf(maxX, landmark.x())
                maxY = maxOf(maxY, landmark.y())
            }
        }

        // Return null if min/max were not updated (empty indices or invalid landmarks)
        if (minX == Float.MAX_VALUE) return null

        val left = (minX * imageWidth).toInt().coerceAtLeast(0)
        val top = (minY * imageHeight).toInt().coerceAtLeast(0)
        val right = (maxX * imageWidth).toInt().coerceAtMost(imageWidth)
        val bottom = (maxY * imageHeight).toInt().coerceAtMost(imageHeight)

        return Rect(left, top, right, bottom)
    }

    fun extractFacialTransformationMatrix(result: FaceLandmarkerResult): FloatArray? {
        val matrixOptional = result.facialTransformationMatrixes()
        if (matrixOptional.isPresent && matrixOptional.get().isNotEmpty()) {
            val matrix = matrixOptional.get()[0] // First face
            // It appears matrix is already a FloatArray based on compiler error analysis
            return matrix
        }
        return null
    }

    interface FaceLandmarkerListener {
        fun onError(error: String)
        fun onResults(result: FaceLandmarkerResult)
    }

    companion object {
        // MediaPipe Face Mesh Lip Landmark Indices
        private val LIP_INDICES = listOf(
            0, 13, 14, 17, 37, 39, 40, 61, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 317, 318, 321, 375, 402, 405, 409
        )
    }
}
