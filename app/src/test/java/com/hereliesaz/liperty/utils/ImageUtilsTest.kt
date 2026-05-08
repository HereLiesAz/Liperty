package com.hereliesaz.liperty.utils

import android.graphics.Bitmap
import android.graphics.Rect
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config

@RunWith(AndroidJUnit4::class)
@Config(sdk = [34])
class ImageUtilsTest {

    @Test
    fun testCropBitmap() {
        val bitmap = Bitmap.createBitmap(100, 100, Bitmap.Config.ARGB_8888)
        val rect = Rect(10, 10, 50, 50)
        val cropped = ImageUtils.cropBitmap(bitmap, rect)

        assertEquals(40, cropped.width)
        assertEquals(40, cropped.height)
    }

    @Test
    fun testAlignAndCropMouth() {
        val bitmap = Bitmap.createBitmap(100, 100, Bitmap.Config.ARGB_8888)
        val rect = Rect(40, 40, 60, 60)
        val targetWidth = 128
        val targetHeight = 64

        val aligned = ImageUtils.alignAndCropMouth(bitmap, rect, 0f, targetWidth, targetHeight)

        assertEquals(targetWidth, aligned.width)
        assertEquals(targetHeight, aligned.height)
    }

    @Test
    fun testToGrayscale() {
        val src = Bitmap.createBitmap(10, 10, Bitmap.Config.ARGB_8888)
        val dest = Bitmap.createBitmap(10, 10, Bitmap.Config.ARGB_8888)

        ImageUtils.toGrayscale(src, dest)
        // Just ensure no crash
    }

    @Test
    fun testApplyBlur() {
        val bitmap = Bitmap.createBitmap(3, 3, Bitmap.Config.ARGB_8888)
        // Center white, others black/transparent (0)
        bitmap.setPixel(1, 1, Color.WHITE)

        val blurred = ImageUtils.applyBlur(bitmap, 1)

        val center = blurred.getPixel(1, 1)
        // 255 (center) + 0 (8 neighbors) / 9 = 28.
        val r = Color.red(center)
        assertEquals(28, r)
    }

    @Test
    fun testApplyHistogramEqualization() {
        val width = 100
        val height = 1
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        // 50 pixels intensity 10, 50 pixels intensity 20
        for (i in 0 until 50) bitmap.setPixel(i, 0, Color.rgb(10, 10, 10))
        for (i in 50 until 100) bitmap.setPixel(i, 0, Color.rgb(20, 20, 20))

        val equalized = ImageUtils.applyHistogramEqualization(bitmap)

        // Expected: 10 -> 0, 20 -> 255 (approx)
        val val1 = Color.red(equalized.getPixel(0, 0))
        val val2 = Color.red(equalized.getPixel(99, 0))

        // Allow variance due to float conversion
        assertEquals(0.0, val1.toDouble(), 1.0)
        assertEquals(255.0, val2.toDouble(), 1.0)
    }
}
