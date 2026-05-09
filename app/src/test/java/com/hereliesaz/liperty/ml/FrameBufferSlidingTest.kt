package com.hereliesaz.liperty.ml

import android.graphics.Bitmap
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotSame
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config

/**
 * Tests for [FrameBuffer.slideAndGetFrames] — the sliding-window inference
 * primitive. Each call returns a snapshot of all currently buffered frames
 * (caller takes ownership) and retains a copy of the most recent N frames
 * for the next inference window.
 *
 * Uses `@Config(sdk = [34])` because Robolectric on this project's
 * `targetSdk = 37` fails to load. Existing FrameBufferTest is broken for
 * the same reason; this file uses the working AndroidJUnit4+Config pattern.
 */
@RunWith(AndroidJUnit4::class)
@Config(sdk = [34])
class FrameBufferSlidingTest {

    private fun bm(): Bitmap = Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)

    @Test
    fun returnsSnapshotOfAllCurrentFrames() {
        val buffer = FrameBuffer(capacity = 4)
        repeat(4) { buffer.addFrame(bm()) }

        val snapshot = buffer.slideAndGetFrames(retainCount = 2)
        assertEquals(4, snapshot.size)
    }

    @Test
    fun retainsLastNFramesForNextWindow() {
        val buffer = FrameBuffer(capacity = 4)
        repeat(4) { buffer.addFrame(bm()) }

        buffer.slideAndGetFrames(retainCount = 2)
        assertEquals(2, buffer.size())
        // After adding 2 more, the buffer should be full again.
        buffer.addFrame(bm())
        buffer.addFrame(bm())
        assertEquals(4, buffer.size())
    }

    @Test
    fun retainCountZeroIsEquivalentToFullClear() {
        val buffer = FrameBuffer(capacity = 4)
        repeat(4) { buffer.addFrame(bm()) }

        val snapshot = buffer.slideAndGetFrames(retainCount = 0)
        assertEquals(4, snapshot.size)
        assertEquals(0, buffer.size())
    }

    @Test
    fun retainCountEqualToSizeKeepsAllFrames() {
        val buffer = FrameBuffer(capacity = 4)
        repeat(4) { buffer.addFrame(bm()) }

        val snapshot = buffer.slideAndGetFrames(retainCount = 4)
        assertEquals(4, snapshot.size)
        assertEquals(4, buffer.size())
    }

    @Test
    fun retainedFramesAreCopiesNotSharedReferences() {
        // The caller treats the snapshot as owned (recycles when done with
        // inference). The buffer must hold COPIES so the snapshot bitmaps
        // can be safely recycled without invalidating the next window.
        val buffer = FrameBuffer(capacity = 3)
        val original = bm()
        buffer.addFrame(original)
        buffer.addFrame(bm())
        buffer.addFrame(bm())

        val snapshot = buffer.slideAndGetFrames(retainCount = 1)
        val retained = buffer.getFrames()

        assertEquals(3, snapshot.size)
        assertEquals(1, retained.size)
        // The retained bitmap must NOT be the same object as any in the
        // snapshot; the caller will recycle the snapshot bitmaps.
        for ((sBm, _) in snapshot) {
            assertNotSame(retained[0].first, sBm)
        }
    }

    @Test
    fun rejectsRetainCountExceedingCurrentSize() {
        val buffer = FrameBuffer(capacity = 4)
        buffer.addFrame(bm())
        buffer.addFrame(bm())

        try {
            buffer.slideAndGetFrames(retainCount = 5)
            throw AssertionError("Expected IllegalArgumentException for retainCount > size")
        } catch (e: IllegalArgumentException) {
            // expected
        }
    }
}
