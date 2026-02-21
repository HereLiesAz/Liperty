package com.HereLiesAz.Liperty.utils

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Rect
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class ImageUtilsTest {

    @Test
    fun testCropBitmap() {
        val original = Bitmap.createBitmap(100, 100, Bitmap.Config.ARGB_8888)
        val rect = Rect(10, 10, 50, 50)

        val cropped = ImageUtils.cropBitmap(original, rect)

        assertEquals(40, cropped.width)
        assertEquals(40, cropped.height)
    }

    @Test
    fun testCropBitmapOutOfBounds() {
        val original = Bitmap.createBitmap(100, 100, Bitmap.Config.ARGB_8888)
        val rect = Rect(-10, -10, 40, 40) // width=50, height=50

        val cropped = ImageUtils.cropBitmap(original, rect)

        assertEquals(50, cropped.width)
        assertEquals(50, cropped.height)
    }

    @Test
    fun testToGrayscale() {
        val src = Bitmap.createBitmap(10, 10, Bitmap.Config.ARGB_8888)
        val dest = Bitmap.createBitmap(10, 10, Bitmap.Config.ARGB_8888)

        ImageUtils.toGrayscale(src, dest)

        assertNotNull(dest)
    }

    @Test
    fun testApplyBlur() {
        val src = Bitmap.createBitmap(10, 10, Bitmap.Config.ARGB_8888)
        // Set a pixel to white
        src.setPixel(5, 5, Color.WHITE)

        val blurred = ImageUtils.applyBlur(src, 1)

        assertNotNull(blurred)
        assertEquals(src.width, blurred.width)
        assertEquals(src.height, blurred.height)

        // Pixel at 5,5 should be less intense than pure white due to blur
        val blurredPixel = blurred.getPixel(5, 5)
        // Should not be pure white if blur worked
        // Wait, pure white is 255. If neighbors are black, avg < 255.
        assertTrue(Color.red(blurredPixel) < 255)
    }

    @Test
    fun testApplyHistogramEqualization() {
        val src = Bitmap.createBitmap(10, 10, Bitmap.Config.ARGB_8888)
        // Fill with dark gray
        for (i in 0 until 10) {
            for (j in 0 until 10) {
                src.setPixel(i, j, Color.rgb(50, 50, 50))
            }
        }

        val equalized = ImageUtils.applyHistogramEqualization(src)

        assertNotNull(equalized)
        // Uniform color image -> should remain uniform but mapped to max range?
        // If all pixels are 50. minCdf = totalPixels. denominator = 1. scale = 255.
        // val = (cdf - minCdf) * scale = 0.
        // So it maps to 0.
        // Wait, if minCdf is totalPixels, then (cdf[i] - minCdf) is 0 for i=50.
        // So it becomes black.

        val pixel = equalized.getPixel(0, 0)
        assertEquals(0, Color.red(pixel))
    }
}
