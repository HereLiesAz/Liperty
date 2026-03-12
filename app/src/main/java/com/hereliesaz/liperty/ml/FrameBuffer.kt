package com.HereLiesAz.liperty.ml

import android.graphics.Bitmap
import java.util.ArrayDeque

class FrameBuffer(private val capacity: Int) {

    private val buffer = ArrayDeque<Bitmap>(capacity)

    /**
     * Adds a frame to the buffer.
     * If the buffer is full, the oldest frame is removed.
     * Thread-safe.
     */
    @Synchronized
    fun addFrame(bitmap: Bitmap) {
        if (buffer.size >= capacity) {
            buffer.removeFirst()
        }
        buffer.addLast(bitmap)
    }

    @Synchronized
    fun getFrames(): List<Bitmap> {
        return buffer.toList()
    }

    @Synchronized
    fun isFull(): Boolean {
        return buffer.size == capacity
    }

    @Synchronized
    fun clear() {
        buffer.clear()
    }
}
