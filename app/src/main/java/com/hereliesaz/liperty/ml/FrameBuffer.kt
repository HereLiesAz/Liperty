package com.hereliesaz.liperty.ml

import android.graphics.Bitmap
import java.util.ArrayDeque

class FrameBuffer(private val capacity: Int) {

    private val buffer = ArrayDeque<Pair<Bitmap, FloatArray?>>(capacity)

    /**
     * Adds a frame and an optional array of landmarks to the buffer.
     * If the buffer is full, the oldest frame is removed and recycled securely via BitmapPool.
     * Thread-safe.
     */
    @Synchronized
    fun addFrame(bitmap: Bitmap, landmarks: FloatArray? = null) {
        if (buffer.size >= capacity) {
            val oldEntry = buffer.removeFirst()
            com.hereliesaz.liperty.utils.BitmapPool.recycle(oldEntry.first)
        }
        buffer.addLast(Pair(bitmap, landmarks))
    }

    @Synchronized
    fun getFrames(): List<Pair<Bitmap, FloatArray?>> {
        return buffer.toList()
    }

    /**
     * Clears the buffer and transfers ownership of the Bitmaps to the caller.
     * The caller is now responsible for recycling the returned Bitmaps.
     */
    @Synchronized
    fun clearAndGetFrames(): List<Pair<Bitmap, FloatArray?>> {
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
        for (entry in buffer) {
            com.hereliesaz.liperty.utils.BitmapPool.recycle(entry.first)
        }
        buffer.clear()
    }

    /**
     * Sliding-window inference primitive. Returns a snapshot of all currently
     * buffered frames (caller takes ownership and is responsible for recycling)
     * while internally retaining COPIES of the most recent [retainCount] frames
     * for the next inference window.
     *
     * The copies are necessary because the caller will recycle the snapshot's
     * Bitmaps after inference completes; without copies, the retained tail
     * would point at recycled memory and the next window's frames would be
     * corrupted.
     *
     * @throws IllegalArgumentException if [retainCount] is negative or exceeds
     *   the current buffer size.
     */
    @Synchronized
    fun slideAndGetFrames(retainCount: Int): List<Pair<Bitmap, FloatArray?>> {
        require(retainCount >= 0) { "retainCount must be non-negative, got $retainCount" }
        require(retainCount <= buffer.size) {
            "retainCount $retainCount exceeds buffer size ${buffer.size}"
        }

        val snapshot = buffer.toList()

        val toRetainSnapshots = if (retainCount == 0) emptyList()
                                else snapshot.takeLast(retainCount)
        val retainedCopies = toRetainSnapshots.map { (bm, lm) ->
            val copy = com.hereliesaz.liperty.utils.BitmapPool.get(bm.width, bm.height)
            android.graphics.Canvas(copy).drawBitmap(bm, 0f, 0f, null)
            Pair(copy, lm)
        }
        buffer.clear()
        for (entry in retainedCopies) buffer.addLast(entry)

        return snapshot
    }
}
