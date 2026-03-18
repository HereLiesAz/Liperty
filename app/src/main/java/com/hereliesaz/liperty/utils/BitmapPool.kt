package com.hereliesaz.liperty.utils

import android.graphics.Bitmap
import java.util.ArrayDeque

/**
 * Config-aware Bitmap pool for zero-GC-churn image reuse.
 * Keyed by "widthxheightxconfig" so ALPHA_8 grayscale mouth ROIs
 * are pooled separately from ARGB_8888 full-frame bitmaps.
 */
object BitmapPool {
    private val pools = HashMap<String, ArrayDeque<Bitmap>>()
    private const val MAX_POOL_SIZE = 6

    fun get(width: Int, height: Int, config: Bitmap.Config = Bitmap.Config.ARGB_8888): Bitmap {
        val key = key(width, height, config)
        synchronized(pools) {
            val deque = pools[key]
            while (deque != null && deque.isNotEmpty()) {
                val candidate = deque.pollFirst()!!
                if (!candidate.isRecycled) return candidate
            }
        }
        return Bitmap.createBitmap(width, height, config)
    }

    fun recycle(bitmap: Bitmap) {
        if (bitmap.isRecycled) return
        val config = bitmap.config ?: return // Don't pool bitmaps without a config
        val key = key(bitmap.width, bitmap.height, config)
        synchronized(pools) {
            val deque = pools.getOrPut(key) { ArrayDeque() }
            if (deque.size < MAX_POOL_SIZE) deque.addLast(bitmap)
        }
    }

    fun clear() {
        synchronized(pools) { pools.clear() }
    }

    private fun key(w: Int, h: Int, c: Bitmap.Config) = "${w}x${h}x${c.name}"
}
