package com.hereliesaz.liperty.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import java.io.File
import kotlin.math.atan2

class FaceLandmarkerHelper(
    val context: Context
) {
    private var faceLandmarker: FaceLandmarker? = null

    private var minFaceDetectionConfidence = 0.5f
    private var minFaceTrackingConfidence = 0.5f

    init {
        setupFaceLandmarker()
    }

    /**
     * Re-initializes the FaceLandmarker with new confidence thresholds.
     * Higher confidence = fewer false positives but might miss faces in bad lighting.
     * BlazeFace (the underlying model) works best with defaults, but adjustment helps in VSR.
     */
    @Synchronized
    fun updateConfidence(detectionConfidence: Float, trackingConfidence: Float) {
        if (detectionConfidence != minFaceDetectionConfidence || trackingConfidence != minFaceTrackingConfidence) {
            minFaceDetectionConfidence = detectionConfidence
            minFaceTrackingConfidence = trackingConfidence
            closeInternal()
            setupFaceLandmarker()
        }
    }

    @Synchronized
    private fun setupFaceLandmarker() {
        try {
            // Prefer downloaded model in filesDir; fall back to bundled asset (dev builds)
            val modelFile = File(context.filesDir, "face_landmarker.task")
            val modelPath = if (modelFile.exists() && modelFile.length() > 1000)
                modelFile.absolutePath
            else
                "face_landmarker.task"

            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(modelPath)
                .build()

            val options = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setMinFaceDetectionConfidence(minFaceDetectionConfidence)
                .setMinTrackingConfidence(minFaceTrackingConfidence)
                .setRunningMode(RunningMode.IMAGE)
                .setOutputFacialTransformationMatrixes(true)
                .build()

            faceLandmarker = FaceLandmarker.createFromOptions(context, options)
        } catch (e: Exception) {
            Log.e("FaceLandmarkerHelper", "Error initializing FaceLandmarker", e)
        }
    }

    @Synchronized
    fun detectSynchronously(imageBitmap: Bitmap): FaceLandmarkerResult? {
        if (faceLandmarker == null) return null

        val mpImage = BitmapImageBuilder(imageBitmap).build()

        return try {
            faceLandmarker?.detect(mpImage)
        } catch (e: Exception) {
            Log.e("FaceLandmarkerHelper", "Error during detection", e)
            null
        }
    }

    @Synchronized
    fun close() {
        closeInternal()
    }

    private fun closeInternal() {
        faceLandmarker?.close()
        faceLandmarker = null
    }

    /**
     * Bounding box covering the whole face (all 468 landmarks), expanded by
     * [marginPct] on each side and squared to the longest side. Matches the
     * full-face crop used by VALLR's offline preprocessing pipeline, which is
     * what the deployed VideoMAE checkpoint was trained on.
     */
    fun extractFaceBoundingBox(
        result: FaceLandmarkerResult,
        imageWidth: Int,
        imageHeight: Int,
        marginPct: Float = 0.15f
    ): Rect? {
        val bounds = landmarkBoundsNormalized(result) { _, _ -> true } ?: return null
        val minX = bounds[0]; val minY = bounds[1]; val maxX = bounds[2]; val maxY = bounds[3]
        val cx = ((minX + maxX) * 0.5f) * imageWidth
        val cy = ((minY + maxY) * 0.5f) * imageHeight
        val side = kotlin.math.max((maxX - minX) * imageWidth, (maxY - minY) * imageHeight) *
            (1f + 2f * marginPct)
        val half = side * 0.5f
        val left = (cx - half).toInt().coerceAtLeast(0)
        val top = (cy - half).toInt().coerceAtLeast(0)
        val right = (cx + half).toInt().coerceAtMost(imageWidth)
        val bottom = (cy + half).toInt().coerceAtMost(imageHeight)
        if (right <= left || bottom <= top) return null
        return Rect(left, top, right, bottom)
    }

    fun extractLipBoundingBox(result: FaceLandmarkerResult, imageWidth: Int, imageHeight: Int): Rect? {
        val lipSet = LIP_INDICES.toHashSet()
        val bounds = landmarkBoundsNormalized(result) { idx, _ -> idx in lipSet }
            ?: return null
        val left = (bounds[0] * imageWidth).toInt().coerceAtLeast(0)
        val top = (bounds[1] * imageHeight).toInt().coerceAtLeast(0)
        val right = (bounds[2] * imageWidth).toInt().coerceAtMost(imageWidth)
        val bottom = (bounds[3] * imageHeight).toInt().coerceAtMost(imageHeight)
        return Rect(left, top, right, bottom)
    }

    /**
     * Returns `[minX, minY, maxX, maxY]` in normalized [0,1] coords over the
     * first detected face's landmarks for which [include] returns true, or null
     * if no face / no matching landmarks.
     */
    private inline fun landmarkBoundsNormalized(
        result: FaceLandmarkerResult,
        include: (index: Int, total: Int) -> Boolean
    ): FloatArray? {
        if (result.faceLandmarks().isEmpty()) return null
        val landmarks = result.faceLandmarks()[0]
        if (landmarks.isEmpty()) return null
        var minX = Float.MAX_VALUE
        var minY = Float.MAX_VALUE
        var maxX = Float.NEGATIVE_INFINITY
        var maxY = Float.NEGATIVE_INFINITY
        for (i in landmarks.indices) {
            if (!include(i, landmarks.size)) continue
            val lm = landmarks[i]
            if (lm.x() < minX) minX = lm.x()
            if (lm.y() < minY) minY = lm.y()
            if (lm.x() > maxX) maxX = lm.x()
            if (lm.y() > maxY) maxY = lm.y()
        }
        if (minX == Float.MAX_VALUE) return null
        return floatArrayOf(minX, minY, maxX, maxY)
    }

    fun extractFacialTransformationMatrix(result: FaceLandmarkerResult): FloatArray? {
        val matrixOptional = result.facialTransformationMatrixes()
        if (matrixOptional.isPresent && matrixOptional.get().isNotEmpty()) {
            return matrixOptional.get()[0] // First face
        }
        return null
    }

    companion object {
        // MediaPipe Face Mesh Lip Landmark Indices
        val LIP_INDICES = listOf(
            0, 13, 14, 17, 37, 39, 40, 61, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 317, 318, 321, 375, 402, 405, 409
        )

        /**
         * Calculates Head Pose (Roll, Pitch, Yaw) from the transformation matrix.
         * The matrix is a 4x4 row-major array.
         * Returns Triple(Roll, Pitch, Yaw) in radians.
         */
        fun calculateHeadPose(matrix: FloatArray): Triple<Float, Float, Float>? {
            if (matrix.size != 16) return null

            // Rotation matrix elements (top-left 3x3)
            val r11 = matrix[0]; val r21 = matrix[4]; val r31 = matrix[8]
            val r32 = matrix[9]; val r33 = matrix[10]

            // Calculating Euler angles
            // Pitch (X-axis rotation)
            val pitch = atan2(r32, r33)

            // Yaw (Y-axis rotation)
            val yaw = atan2(-r31, kotlin.math.sqrt(r32 * r32 + r33 * r33))

            // Roll (Z-axis rotation)
            val roll = atan2(r21, r11)

            return Triple(roll, pitch, yaw)
        }

        /**
         * Calculates the rotation angle (in degrees) required to align the lips horizontally.
         * Uses landmarks 61 (left corner) and 291 (right corner).
         */
        fun calculateLipRotation(result: FaceLandmarkerResult): Float {
            if (result.faceLandmarks().isEmpty()) return 0f
            val landmarks = result.faceLandmarks()[0]

            if (landmarks.size <= 291) return 0f

            val leftCorner = landmarks[61]
            val rightCorner = landmarks[291]

            val deltaX = rightCorner.x() - leftCorner.x()
            val deltaY = rightCorner.y() - leftCorner.y() // Note: y-axis points down in image coordinates

            val radians = atan2(deltaY, deltaX)
            return Math.toDegrees(radians.toDouble()).toFloat()
        }
    }
}
