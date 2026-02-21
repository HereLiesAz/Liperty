package com.HereLiesAz.Liperty.utils

import android.os.SystemClock

object PerformanceMonitor {

    private var lastFrameTime = 0L
    private var frameCount = 0
    private var fps = 0.0f

    private var inferenceCount = 0
    private var totalInferenceTime = 0L
    private var avgInferenceTime = 0.0f

    fun startFrame() {
        // Optional: Measure frame processing duration if needed
    }

    fun endFrame() {
        val currentTime = SystemClock.uptimeMillis()
        if (lastFrameTime == 0L) {
            lastFrameTime = currentTime
            return
        }

        frameCount++
        val diff = currentTime - lastFrameTime
        if (diff >= 1000) { // Update every second
            fps = (frameCount * 1000.0f) / diff
            frameCount = 0
            lastFrameTime = currentTime
        }
    }

    fun recordInferenceTime(timeMs: Long) {
        totalInferenceTime += timeMs
        inferenceCount++
        avgInferenceTime = totalInferenceTime.toFloat() / inferenceCount
    }

    fun getStats(): String {
        return "FPS: %.1f | Avg Inf: %.1f ms".format(fps, avgInferenceTime)
    }
}
