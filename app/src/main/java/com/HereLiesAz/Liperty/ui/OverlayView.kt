package com.HereLiesAz.Liperty.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.AttributeSet
import android.view.View

class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    private var faceResults: List<Rect> = emptyList()
    private var lipResults: List<Rect> = emptyList()
    private val facePaint = Paint()
    private val lipPaint = Paint()

    init {
        initPaints()
    }

    private fun initPaints() {
        facePaint.color = Color.GREEN
        facePaint.style = Paint.Style.STROKE
        facePaint.strokeWidth = 8f

        lipPaint.color = Color.BLUE
        lipPaint.style = Paint.Style.STROKE
        lipPaint.strokeWidth = 8f
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        for (rect in faceResults) {
            canvas.drawRect(rect, facePaint)
        }
        for (rect in lipResults) {
            canvas.drawRect(rect, lipPaint)
        }
    }

    fun setResults(faces: List<Rect>, lips: List<Rect>) {
        faceResults = faces
        lipResults = lips
        invalidate()
    }

    fun clear() {
        faceResults = emptyList()
        lipResults = emptyList()
        invalidate()
    }
}
