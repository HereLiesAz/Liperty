package com.hereliesaz.liperty.ml

import android.graphics.Bitmap
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class FrameBufferTest {

    @Test
    fun testBufferLogic() {
        val capacity = 5
        val buffer = FrameBuffer(capacity)
        val bitmap = Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)

        // Add frames up to capacity
        for (i in 1..capacity) {
            buffer.addFrame(bitmap)
            if (i < capacity) {
                assertFalse("Buffer should not be full yet", buffer.isFull())
            } else {
                assertTrue("Buffer should be full", buffer.isFull())
            }
        }

        assertEquals(capacity, buffer.getFrames().size)

        // Add one more, size should remain capacity
        buffer.addFrame(bitmap)
        assertEquals(capacity, buffer.getFrames().size)
    }

    @Test
    fun testClear() {
        val buffer = FrameBuffer(5)
        buffer.addFrame(Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888))
        buffer.clearAndRecycle()
        assertEquals(0, buffer.getFrames().size)
        assertFalse(buffer.isFull())
    }
}
