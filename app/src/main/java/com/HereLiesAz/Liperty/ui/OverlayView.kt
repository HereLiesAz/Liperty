package com.HereLiesAz.Liperty.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.AttributeSet
import android.view.View

class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    private var results: List<Rect> = emptyList()
    private val boxPaint = Paint()

    init {
        initPaints()
    }

    private fun initPaints() {
        boxPaint.color = Color.GREEN
        boxPaint.style = Paint.Style.STROKE
        boxPaint.strokeWidth = 8f
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        for (rect in results) {
            canvas.drawRect(rect, boxPaint)
        }
    }

    fun setResults(detectionResults: List<Rect>) {
        results = detectionResults
        invalidate()
    }
}
