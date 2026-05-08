package com.hereliesaz.liperty.utils

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.graphics.Rect
import android.util.Log
import androidx.camera.core.ImageProxy
import org.opencv.android.OpenCVLoader
import kotlin.math.max

object ImageUtils {

    private var openCVLoaded = false


    /**
     * Initializes OpenCV. Call this in Application.onCreate or Activity.onCreate.
     */
    fun initializeOpenCV(context: Context) {
        if (!openCVLoaded) {
            try {
                if (!OpenCVLoader.initLocal()) {
                    Log.e("ImageUtils", "OpenCV initialization failed — native library not bundled")
                    return
                }
                System.loadLibrary("liperty_cv")
                openCVLoaded = true
                Log.i("ImageUtils", "OpenCV and 'liperty_cv' loaded successfully.")
            } catch (e: UnsatisfiedLinkError) {
                Log.e("ImageUtils", "Failed to load native library: ${e.message}")
            }
        }
    }

    /**
     * Converts a YUV_420_888 ImageProxy to a Bitmap.
     * Uses CameraX's optimized toBitmap() extension.
     */
    fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        return image.toBitmap()
    }

    // Pre-allocated buffers for zero-allocation YUV to ARGB conversion.
    // NOTE: These fields are NOT thread-safe and must only be accessed from a single
    // camera/processing thread. Do not call imageProxyToBitmap(ImageProxy, Bitmap)
    // concurrently from multiple threads.
    private var nv21Buffer: ByteArray? = null
    private var argbBuffer: IntArray? = null

    /**
     * Extracts YUV data from ImageProxy and converts it to ARGB into a pre-allocated Bitmap.
     * Reduces garbage collection overhead by reusing the bitmap and buffers.
     */
    fun imageProxyToBitmap(image: ImageProxy, destBitmap: Bitmap) {
        val yPlane = image.planes[0].buffer
        val uPlane = image.planes[1].buffer
        val vPlane = image.planes[2].buffer

        val ySize = yPlane.remaining()
        val uSize = uPlane.remaining()
        val vSize = vPlane.remaining()

        val totalSize = ySize + uSize + vSize
        val buffer = nv21Buffer
        val nv21 = if (buffer == null || buffer.size < totalSize) {
            ByteArray(totalSize).also { nv21Buffer = it }
        } else {
            buffer
        }
        val nv21Array = nv21 as ByteArray

        yPlane.get(nv21Array, 0, ySize)
        vPlane.get(nv21Array, ySize, vSize)
        uPlane.get(nv21Array, ySize + vSize, uSize)

        val width = image.width
        val height = image.height
        val totalPixels = width * height

        if (argbBuffer == null || argbBuffer!!.size < totalPixels) {
            argbBuffer = IntArray(totalPixels)
        }
        val argb = argbBuffer!!

        val yRowStride = image.planes[0].rowStride
        val uvRowStride = image.planes[1].rowStride
        val uvPixelStride = image.planes[1].pixelStride

        var yp = 0
        for (j in 0 until height) {
            val pY = yRowStride * j
            val pUV = uvRowStride * (j shr 1)

            for (i in 0 until width) {
                val uvOffset = pUV + (i shr 1) * uvPixelStride

                val y = nv21Array[pY + i].toInt() and 0xFF
                val v = nv21Array[ySize + uvOffset].toInt() and 0xFF
                val u = nv21Array[ySize + vSize + uvOffset].toInt() and 0xFF

                var r = y + (1.370705 * (v - 128)).toInt()
                var g = y - (0.698001 * (v - 128)).toInt() - (0.337633 * (u - 128)).toInt()
                var b = y + (1.732446 * (u - 128)).toInt()

                r = r.coerceIn(0, 255)
                g = g.coerceIn(0, 255)
                b = b.coerceIn(0, 255)

                argb[yp++] = Color.argb(255, r, g, b)
            }
        }
        destBitmap.setPixels(argb, 0, width, 0, 0, width, height)
    }

    fun rotateBitmap(bitmap: Bitmap, rotationDegrees: Float): Bitmap {
        val matrix = android.graphics.Matrix()
        matrix.postRotate(rotationDegrees)
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    /**
     * Mirrors the bitmap horizontally.
     */
    fun mirrorBitmap(bitmap: Bitmap): Bitmap {
        val matrix = android.graphics.Matrix()
        matrix.postScale(-1f, 1f, bitmap.width / 2f, bitmap.height / 2f)
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
     * Aligns and crops the mouth region from the input bitmap.
     */
    @Deprecated(
        "Use the two-arg overload with a pooled destBitmap",
        ReplaceWith("alignAndCropMouth(bitmap, destBitmap)")
    )
    fun alignAndCropMouth(bitmap: Bitmap, mouthRect: Rect, rotationDegrees: Float, targetWidth: Int, targetHeight: Int): Bitmap {
        val targetBitmap = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
        return alignAndCropMouth(bitmap, mouthRect, rotationDegrees, targetWidth, targetHeight, targetBitmap)
    }

    /**
     * Overload that writes to a reused bitmap to reduce allocation.
     */
    fun alignAndCropMouth(bitmap: Bitmap, mouthRect: Rect, rotationDegrees: Float, targetWidth: Int, targetHeight: Int, destBitmap: Bitmap): Bitmap {
        // Ensure destBitmap is correct size, if not, recreate (though typically caller ensures this)
        val targetBitmap = if (destBitmap.width != targetWidth || destBitmap.height != targetHeight) {
            Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
        } else {
            destBitmap
        }

        val canvas = Canvas(targetBitmap)
        val scale = targetWidth.toFloat() / max(mouthRect.width(), 1) // Base scale on width primarily

        val drawingMatrix = android.graphics.Matrix()
        drawingMatrix.postTranslate(-mouthRect.centerX().toFloat(), -mouthRect.centerY().toFloat())
        drawingMatrix.postRotate(rotationDegrees)
        drawingMatrix.postScale(scale, scale)
        drawingMatrix.postTranslate(targetWidth / 2f, targetHeight / 2f)

        val paint = Paint()
        paint.isFilterBitmap = true

        canvas.drawBitmap(bitmap, drawingMatrix, paint)

        return targetBitmap
    }

    /**
     * Applies Histogram Equalization to a Grayscale bitmap.
     * Improves contrast by stretching the intensity range.
     * Uses Native OpenCV implementation for hardware acceleration.
     */
    fun applyHistogramEqualization(bitmap: Bitmap): Bitmap {
        if (openCVLoaded) {
            // Modify in-place for zero-allocation performance
            applyHistogramEqualizationNative(bitmap)
            return bitmap
        }

        // Fallback Kotlin implementation
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

    /**
     * Native method declarations.
     * These methods modify the bitmap in-place using Android's Bitmap locking.
     */
    private external fun applyHistogramEqualizationNative(bitmap: Bitmap)
    private external fun applyNormalizationNative(bitmap: Bitmap)

    /**
     * Applies full normalization (Blur + Equalization) in a single JNI pass.
     */
    fun normalizeForInference(bitmap: Bitmap): Bitmap {
        if (openCVLoaded) {
            // Apply in-place for zero-allocation
            applyNormalizationNative(bitmap)
            return bitmap
        }
        // Fallback to sequential Kotlin steps
        var processed = applyBlur(bitmap, 1)
        processed = applyHistogramEqualization(processed)
        return processed
    }

    /**
     * Applies a simple box blur to the bitmap.
     */
    fun applyBlur(bitmap: Bitmap, radius: Int): Bitmap {
        if (radius < 1) return bitmap

        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val newPixels = IntArray(width * height)

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
