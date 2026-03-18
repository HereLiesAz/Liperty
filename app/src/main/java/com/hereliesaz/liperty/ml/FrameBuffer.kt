package com.hereliesaz.liperty.ml

import android.graphics.Bitmap
import java.util.ArrayDeque

class FrameBuffer(private val capacity: Int) {

    private val buffer = ArrayDeque<Bitmap>(capacity)

    /**
     * Adds a frame to the buffer.
     * If the buffer is full, the oldest frame is removed and recycled securely via BitmapPool.
     * Thread-safe.
     */
    @Synchronized
    fun addFrame(bitmap: Bitmap) {
        if (buffer.size >= capacity) {
            val oldBitmap = buffer.removeFirst()
            com.hereliesaz.liperty.utils.BitmapPool.recycle(oldBitmap)
        }
        buffer.addLast(bitmap)
    }

    @Synchronized
    fun getFrames(): List<Bitmap> {
        return buffer.toList()
    }

    /**
     * Clears the buffer and transfers ownership of the Bitmaps to the caller.
     * The caller is now responsible for recycling the returned Bitmaps.
     */
    @Synchronized
    fun clearAndGetFrames(): List<Bitmap> {
        val frames = buffer.toList()
        buffer.clear()
        return frames // Caller assumes ownership for recycling
    }

    @Synchronized
    fun size(): Int = buffer.size

    @Synchronized
    fun isFull(): Boolean {
        return buffer.size == capacity
    }

    /**
     * Clears the buffer and explicitly recycles all contained Bitmaps back into the pool.
     */
    @Synchronized
    fun clearAndRecycle() {
        for (bitmap in buffer) {
            com.hereliesaz.liperty.utils.BitmapPool.recycle(bitmap)
        }
        buffer.clear()
    }
}
