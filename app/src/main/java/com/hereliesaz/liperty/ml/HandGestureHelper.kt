package com.hereliesaz.liperty.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.io.File

class HandGestureHelper(val context: Context) {

    private var handLandmarker: HandLandmarker? = null
    private val positionHistory = mutableListOf<Float>()
    private val HISTORY_SIZE = 10 // Track position over last 10 frames

    // Gesture detection parameters
    private val WAVE_OSCILLATIONS = 2 // Needs to change direction twice for a wave
    private val AIR_SWIPE_THRESHOLD_X = 0.2f // Must travel 20% of screen width

    init {
        setupHandLandmarker()
    }

    private fun setupHandLandmarker() {
        try {
            val modelFile = File(context.filesDir, "hand_landmarker.task")
            val modelPath = if (modelFile.exists() && modelFile.length() > 1000)
                modelFile.absolutePath
            else
                "hand_landmarker.task"

            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(modelPath)
                .build()

            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setNumHands(1) // Only track one hand
                .setRunningMode(RunningMode.IMAGE)
                .build()

            handLandmarker = HandLandmarker.createFromOptions(context, options)
        } catch (e: Exception) {
            Log.e("HandGestureHelper", "Error initializing HandLandmarker. Make sure hand_landmarker.task is in assets", e)
        }
    }

    fun detectSynchronously(imageBitmap: Bitmap): HandGesture? {
        if (handLandmarker == null) return null

        val mpImage = BitmapImageBuilder(imageBitmap).build()
        val result = handLandmarker?.detect(mpImage)

        return result?.let { processResult(it) }
    }

    private fun processResult(result: HandLandmarkerResult): HandGesture? {
        if (result.landmarks().isEmpty()) {
            positionHistory.clear() // No hand, clear history
            return null
        }

        // Use wrist landmark (index 0) as the reference point for position
        val wristLandmark = result.landmarks()[0][0]
        val currentX = wristLandmark.x()

        // Add current position to history
        positionHistory.add(currentX)
        if (positionHistory.size > HISTORY_SIZE) {
            positionHistory.removeAt(0)
        }

        // Not enough data to detect a gesture yet
        if (positionHistory.size < HISTORY_SIZE) {
            return null
        }

        return detectWaveOrSwipe()
    }

    private fun detectWaveOrSwipe(): HandGesture? {
        val directions = mutableListOf<Int>()
        for (i in 1 until positionHistory.size) {
            val diff = positionHistory[i] - positionHistory[i-1]
            if (kotlin.math.abs(diff) > 0.01f) { // Ignore minor jitter
                directions.add(if (diff > 0) 1 else -1) // 1 for right, -1 for left
            }
        }

        // Count direction changes for wave detection
        var oscillations = 0
        for (i in 1 until directions.size) {
            if (directions[i] != directions[i-1]) {
                oscillations++
            }
        }

        if (oscillations >= WAVE_OSCILLATIONS) {
            positionHistory.clear() // Reset after gesture
            return HandGesture.WAVE_PAUSE
        }

        // Check for directional swipe
        val startX = positionHistory.first()
        val endX = positionHistory.last()
        val displacement = endX - startX

        if (kotlin.math.abs(displacement) > AIR_SWIPE_THRESHOLD_X) {
            positionHistory.clear() // Reset after gesture
            return if (displacement > 0) HandGesture.AIR_SWIPE_RIGHT else HandGesture.AIR_SWIPE_LEFT
        }

        return null
    }


    fun close() {
        handLandmarker?.close()
    }

    enum class HandGesture {
        WAVE_PAUSE,
        AIR_SWIPE_LEFT,
        AIR_SWIPE_RIGHT
    }
}
