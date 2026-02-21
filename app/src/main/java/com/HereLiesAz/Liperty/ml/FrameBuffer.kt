package com.HereLiesAz.Liperty.ml

import android.graphics.Bitmap
import java.util.ArrayDeque

class FrameBuffer(private val capacity: Int) {

    private val buffer = ArrayDeque<Bitmap>(capacity)

    /**
     * Adds a frame to the buffer.
     * If the buffer is full, the oldest frame is removed.
     */
    fun addFrame(bitmap: Bitmap) {
        if (buffer.size >= capacity) {
            // Remove the oldest frame
            // Ideally we should recycle the bitmap if we manage the pool,
            // but for now we let GC handle it or the caller manages reuse.
            // In a real VSR pipeline, we might be storing FloatArrays (tensors) instead of Bitmaps to save memory.
            buffer.removeFirst()
        }
        buffer.addLast(bitmap)
    }

    fun getFrames(): List<Bitmap> {
        return buffer.toList()
    }

    fun isFull(): Boolean {
        return buffer.size == capacity
    }

    fun clear() {
        buffer.clear()
    }
}
