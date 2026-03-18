package com.hereliesaz.liperty.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

/**
 * Full-face mesh overlay.
 *
 * Draws all 468 MediaPipe face landmarks as small dots, then overlays
 * the lip contour connections (outer + inner) in bright cyan, and
 * optionally a bounding box around the lip region.
 *
 * All coordinates are passed pre-scaled to the view's pixel space
 * so this class has no dependency on MediaPipe types.
 */
class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    // ── Data ─────────────────────────────────────────────────────────────────

    @Volatile private var allLandmarks: List<PointF> = emptyList()
    @Volatile private var lipBox: RectF? = null

    // ── Paints ───────────────────────────────────────────────────────────────

    /** Very faint dots for the full 468-point mesh — just enough to show tracking */
    private val meshDotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(45, 180, 220, 255)
        style = Paint.Style.FILL
    }

    /** Subtle dots on lip landmarks */
    private val lipDotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(120, 0, 220, 255)
        style = Paint.Style.FILL
    }

    /** Lines connecting lip contour landmarks — semi-transparent */
    private val lipLinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(110, 0, 220, 255)
        style = Paint.Style.STROKE
        strokeWidth = 1.8f
        strokeJoin = Paint.Join.ROUND
        strokeCap = Paint.Cap.ROUND
    }

    /** Faint bounding box around the lip region */
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(55, 0, 200, 255)
        style = Paint.Style.STROKE
        strokeWidth = 2f
    }

    // ── Lip topology ─────────────────────────────────────────────────────────

    /**
     * MediaPipe face mesh lip contour index groups.
     * Source: mediapipe/python/solutions/face_mesh_connections.py
     */
    private val outerUpperLip = intArrayOf(61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291)
    private val outerLowerLip = intArrayOf(291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61)
    private val innerUpperLip = intArrayOf(78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308)
    private val innerLowerLip = intArrayOf(308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78)

    private val allLipIndices: Set<Int> = (
        outerUpperLip.toList() + outerLowerLip.toList() +
        innerUpperLip.toList() + innerLowerLip.toList()
    ).toSet()

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Update the overlay with new landmark data.
     *
     * @param landmarks All 468 face landmarks, each already scaled to the
     *                  view's pixel coordinate space (multiply normalized
     *                  x/y by [View.getWidth]/[View.getHeight] before calling).
     * @param lip       Optional bounding box around the lip region, same space.
     */
    fun setLandmarks(landmarks: List<PointF>, lip: RectF?) {
        allLandmarks = landmarks
        lipBox = lip
        postInvalidate()
    }

    /** Backwards-compatible no-op that simply clears the overlay. */
    fun setResults(faces: List<Any>, lips: List<Any>) = clear()

    fun clear() {
        allLandmarks = emptyList()
        lipBox = null
        postInvalidate()
    }

    // ── Drawing ───────────────────────────────────────────────────────────────

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val landmarks = allLandmarks
        if (landmarks.isEmpty()) return

        // 1. Full mesh — all 468 points as tiny dots
        for (pt in landmarks) {
            canvas.drawCircle(pt.x, pt.y, 2f, meshDotPaint)
        }

        // 2. Lip contour connections
        fun drawContour(indices: IntArray) {
            if (indices.any { it >= landmarks.size }) return
            for (i in 0 until indices.size - 1) {
                val a = landmarks[indices[i]]
                val b = landmarks[indices[i + 1]]
                canvas.drawLine(a.x, a.y, b.x, b.y, lipLinePaint)
            }
        }
        drawContour(outerUpperLip)
        drawContour(outerLowerLip)
        drawContour(innerUpperLip)
        drawContour(innerLowerLip)

        // 3. Lip landmark dots (on top of lines)
        for (idx in allLipIndices) {
            if (idx < landmarks.size) {
                canvas.drawCircle(landmarks[idx].x, landmarks[idx].y, 4f, lipDotPaint)
            }
        }

        // 4. Bounding box
        lipBox?.let { canvas.drawRect(it, boxPaint) }
    }
}
