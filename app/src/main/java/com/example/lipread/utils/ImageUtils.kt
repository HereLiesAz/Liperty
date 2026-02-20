package com.example.lipread.utils

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.graphics.Rect
import androidx.camera.core.ImageProxy

object ImageUtils {

    /**
     * Converts a YUV_420_888 ImageProxy to a Bitmap.
     * Uses CameraX's optimized toBitmap() extension.
     */
    fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        return image.toBitmap()
    }

    fun rotateBitmap(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
        val matrix = android.graphics.Matrix()
        matrix.postRotate(rotationDegrees.toFloat())
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    /**
     * Crops the bitmap to the specified rectangle.
     */
    fun cropBitmap(bitmap: Bitmap, rect: Rect): Bitmap {
        // Ensure rect is within bounds
        val safeLeft = rect.left.coerceAtLeast(0)
        val safeTop = rect.top.coerceAtLeast(0)
        val safeWidth = rect.width().coerceAtMost(bitmap.width - safeLeft)
        val safeHeight = rect.height().coerceAtMost(bitmap.height - safeTop)

        return Bitmap.createBitmap(bitmap, safeLeft, safeTop, safeWidth, safeHeight)
    }

    /**
     * Converts Bitmap to Grayscale (as required by most VSR models).
     * Reuses the destination Bitmap to avoid allocation churn.
     */
    fun toGrayscale(src: Bitmap, dest: Bitmap) {
        if (src.width != dest.width || src.height != dest.height) {
             throw IllegalArgumentException("Source and destination bitmaps must have the same dimensions: src=${src.width}x${src.height}, dest=${dest.width}x${dest.height}")
        }
        val c = Canvas(dest)
        val paint = Paint()
        val cm = ColorMatrix()
        cm.setSaturation(0f)
        val f = ColorMatrixColorFilter(cm)
        paint.colorFilter = f
        c.drawBitmap(src, 0f, 0f, paint)
    }
}
