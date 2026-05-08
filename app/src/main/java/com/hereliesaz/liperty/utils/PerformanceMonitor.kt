package com.hereliesaz.liperty.utils

import android.os.SystemClock
import android.util.Log

object PerformanceMonitor {
    private const val TAG = "PerformanceMonitor"

    private var frameCount = 0
    @Volatile private var lastFpsTime = 0L

    fun logInferenceTime(timeMs: Long) {
        Log.d(TAG, "Inference Time: $timeMs ms")
    }

    fun logFrame() {
        val now = SystemClock.uptimeMillis()

        if (lastFpsTime == 0L) {
            lastFpsTime = now
            return
        }

        frameCount++

        if (now - lastFpsTime >= 1000) {
            val fps = frameCount * 1000.0f / (now - lastFpsTime)
            Log.d(TAG, "FPS: $fps")

            frameCount = 0
            lastFpsTime = now
        }
    }
}
