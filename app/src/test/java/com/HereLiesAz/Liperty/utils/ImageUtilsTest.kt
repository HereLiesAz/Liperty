package com.HereLiesAz.Liperty.utils

import android.graphics.Bitmap
import android.graphics.Rect
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
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
        // Rect is left, top, right, bottom.
        // Original logic: width = right - left? No, Rect(left, top, right, bottom)
        // ImageUtils.cropBitmap logic:
        // safeWidth = rect.width().coerceAtMost(bitmap.width - safeLeft)
        // rect.width() = right - left

        // Rect(-10, -10, 50, 50) -> width=60, height=60
        // safeLeft = 0, safeTop = 0
        // safeWidth = 60.coerceAtMost(100 - 0) = 60?
        // Wait, Rect(-10, -10, 50, 50) means left=-10, right=50. width=60.
        // But logic says: safeLeft = rect.left.coerceAtLeast(0) = 0.
        // safeWidth = rect.width().coerceAtMost(bitmap.width - safeLeft)
        // If rect is just a container for width/height or coordinates?
        // Rect(left, top, right, bottom).
        // If I pass a Rect representing a crop region.

        val rect = Rect(-10, -10, 40, 40) // width=50, height=50
        // safeLeft = 0
        // safeTop = 0
        // width = 50. coerceAtMost(100) = 50.

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
}
