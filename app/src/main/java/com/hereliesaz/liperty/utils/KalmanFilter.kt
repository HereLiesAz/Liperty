package com.hereliesaz.liperty.utils

import android.graphics.Rect

/**
 * Simple 1D Kalman Filter for smoothing noisy sensor data.
 */
class KalmanFilter(
    private val processNoise: Float = 0.03f,
    private val measurementNoise: Float = 0.8f
) {
    private var predictedEstimate = 0f
    private var estimateError = 1f
    private var isInitialized = false

    fun update(measurement: Float): Float {
        if (!isInitialized) {
            predictedEstimate = measurement
            isInitialized = true
            return measurement
        }

        // Prediction update
        val priorEstimate = predictedEstimate
        val priorError = estimateError + processNoise

        // Measurement update
        val kalmanGain = priorError / (priorError + measurementNoise)
        predictedEstimate = priorEstimate + kalmanGain * (measurement - priorEstimate)
        estimateError = (1 - kalmanGain) * priorError
        estimateError = estimateError.coerceAtLeast(1e-6f)

        return predictedEstimate
    }

    fun reset() {
        isInitialized = false
    }
}

/**
 * Kalman Filter specialized for Rect coordinates (smoothing bounding boxes).
 */
class RectKalmanFilter(
    processNoise: Float = 0.05f,
    measurementNoise: Float = 1.0f
) {
    private val kfLeft = KalmanFilter(processNoise, measurementNoise)
    private val kfTop = KalmanFilter(processNoise, measurementNoise)
    private val kfRight = KalmanFilter(processNoise, measurementNoise)
    private val kfBottom = KalmanFilter(processNoise, measurementNoise)

    fun update(rect: Rect): Rect {
        val left = kfLeft.update(rect.left.toFloat()).toInt()
        val top = kfTop.update(rect.top.toFloat()).toInt()
        val right = kfRight.update(rect.right.toFloat()).toInt()
        val bottom = kfBottom.update(rect.bottom.toFloat()).toInt()
        return Rect(left, top, right, bottom)
    }

    fun reset() {
        kfLeft.reset()
        kfTop.reset()
        kfRight.reset()
        kfBottom.reset()
    }
}
