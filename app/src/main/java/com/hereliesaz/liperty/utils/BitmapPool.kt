package com.hereliesaz.liperty.utils

import android.graphics.Bitmap
import java.util.ArrayDeque

object BitmapPool {
    private val pool = ArrayDeque<Bitmap>()
    private const val MAX_POOL_SIZE = 10

    fun get(width: Int, height: Int): Bitmap {
        synchronized(pool) {
            val iterator = pool.iterator()
            while (iterator.hasNext()) {
                val bitmap = iterator.next()
                if (bitmap.width == width && bitmap.height == height) {
                    iterator.remove()
                    // Clear the bitmap if needed, or assume caller handles overwriting
                    // bitmap.eraseColor(0)
                    return bitmap
                }
            }
        }
        return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    }

    fun recycle(bitmap: Bitmap) {
        synchronized(pool) {
            if (pool.size < MAX_POOL_SIZE && !pool.contains(bitmap)) {
                pool.add(bitmap)
            } else {
                // If pool is full, let GC handle it (don't explicitly recycle as it might be used elsewhere briefly, though standard practice is to recycle if not pooling)
                // bitmap.recycle() // Risky if used in UI
            }
        }
    }

    fun clear() {
        synchronized(pool) {
            pool.clear()
        }
    }
}
