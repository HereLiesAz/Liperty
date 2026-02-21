package com.HereLiesAz.Liperty.ui

import android.view.GestureDetector
import android.view.MotionEvent
import kotlin.math.abs

class GestureListener(
    private val onSwipeLeft: () -> Unit,
    private val onSwipeRight: () -> Unit,
    private val onSwipeUp: () -> Unit,
    private val onDoubleTapAction: () -> Unit
) : GestureDetector.SimpleOnGestureListener() {

    private val SWIPE_THRESHOLD = 100
    private val SWIPE_VELOCITY_THRESHOLD = 100

    override fun onDown(e: MotionEvent): Boolean {
        return true
    }

    override fun onFling(
        e1: MotionEvent?,
        e2: MotionEvent,
        velocityX: Float,
        velocityY: Float
    ): Boolean {
        if (e1 == null) return false

        val diffY = e2.y - e1.y
        val diffX = e2.x - e1.x

        if (abs(diffX) > abs(diffY)) {
            // Horizontal Swipe
            if (abs(diffX) > SWIPE_THRESHOLD && abs(velocityX) > SWIPE_VELOCITY_THRESHOLD) {
                if (diffX > 0) {
                    onSwipeRight()
                } else {
                    onSwipeLeft()
                }
                return true
            }
        } else {
            // Vertical Swipe
            if (abs(diffY) > SWIPE_THRESHOLD && abs(velocityY) > SWIPE_VELOCITY_THRESHOLD) {
                if (diffY < 0) {
                    // Swipe Up
                    onSwipeUp()
                    return true
                }
            }
        }
        return false
    }

    override fun onDoubleTap(e: MotionEvent): Boolean {
        onDoubleTapAction()
        return true
    }
}
