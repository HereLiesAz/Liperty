package com.HereLiesAz.Liperty.utils

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.graphics.Rect
import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer
import kotlin.math.max
import kotlin.math.min

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

        if (safeWidth <= 0 || safeHeight <= 0) {
             return Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888) // Return 1x1 empty bitmap to avoid crash
        }

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

    /**
     * Applies a simple box blur to the bitmap.
     * A lightweight alternative to Gaussian Blur for mobile devices without using RenderScript.
     * Radius must be odd.
     */
    fun applyBlur(bitmap: Bitmap, radius: Int): Bitmap {
        if (radius < 1) return bitmap

        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val newPixels = IntArray(width * height)
        val kernelSize = radius * 2 + 1

        // Horizontal Pass
        for (y in 0 until height) {
            for (x in 0 until width) {
                var r = 0; var g = 0; var b = 0; var count = 0

                for (i in -radius..radius) {
                    val nx = x + i
                    if (nx in 0 until width) {
                        val pixel = pixels[y * width + nx]
                        r += Color.red(pixel)
                        g += Color.green(pixel)
                        b += Color.blue(pixel)
                        count++
                    }
                }
                newPixels[y * width + x] = Color.rgb(r / count, g / count, b / count)
            }
        }

        // Vertical Pass (using newPixels as source)
        val finalPixels = IntArray(width * height)
        for (x in 0 until width) {
            for (y in 0 until height) {
                var r = 0; var g = 0; var b = 0; var count = 0

                for (i in -radius..radius) {
                    val ny = y + i
                    if (ny in 0 until height) {
                        val pixel = newPixels[ny * width + x]
                        r += Color.red(pixel)
                        g += Color.green(pixel)
                        b += Color.blue(pixel)
                        count++
                    }
                }
                finalPixels[y * width + x] = Color.rgb(r / count, g / count, b / count)
            }
        }

        return Bitmap.createBitmap(finalPixels, width, height, Bitmap.Config.ARGB_8888)
    }

    /**
     * Applies Histogram Equalization to a Grayscale bitmap.
     * Improves contrast by stretching the intensity range.
     */
    fun applyHistogramEqualization(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        if (width == 0 || height == 0) return bitmap

        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        // 1. Calculate Histogram
        val histogram = IntArray(256)
        for (pixel in pixels) {
            // Assume grayscale, take red channel
            val intensity = Color.red(pixel)
            histogram[intensity]++
        }

        // 2. Calculate CDF (Cumulative Distribution Function)
        val cdf = IntArray(256)
        cdf[0] = histogram[0]
        for (i in 1 until 256) {
            cdf[i] = cdf[i - 1] + histogram[i]
        }

        // 3. Normalize CDF
        val minCdf = cdf.firstOrNull { it > 0 } ?: 0
        val totalPixels = width * height
        // Avoid division by zero if all pixels are uniform
        val denominator = max(1, totalPixels - minCdf)
        val scale = 255.0f / denominator

        val equalizedMap = IntArray(256)
        for (i in 0 until 256) {
             equalizedMap[i] = (((cdf[i] - minCdf) * scale).toInt()).coerceIn(0, 255)
        }

        // 4. Map pixels
        val newPixels = IntArray(width * height)
        for (i in pixels.indices) {
            val oldVal = Color.red(pixels[i])
            val newVal = equalizedMap[oldVal]
            newPixels[i] = Color.rgb(newVal, newVal, newVal)
        }

        return Bitmap.createBitmap(newPixels, width, height, Bitmap.Config.ARGB_8888)
    }
}
