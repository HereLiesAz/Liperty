package com.hereliesaz.liperty.ml

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TrainingSampleTest {

    @Test
    fun equalSamplesAreEqual() {
        val a = TrainingSample(
            videoFrames = floatArrayOf(1f, 2f, 3f),
            numFrames = 1,
            targetFeatures = floatArrayOf(4f, 5f),
            numTargetFrames = 1,
        )
        val b = TrainingSample(
            videoFrames = floatArrayOf(1f, 2f, 3f),
            numFrames = 1,
            targetFeatures = floatArrayOf(4f, 5f),
            numTargetFrames = 1,
        )
        assertEquals(a, b)
        assertEquals(a.hashCode(), b.hashCode())
    }

    @Test
    fun differentContentIsNotEqual() {
        val a = TrainingSample(
            videoFrames = floatArrayOf(1f, 2f, 3f),
            numFrames = 1,
            targetFeatures = floatArrayOf(4f, 5f),
            numTargetFrames = 1,
        )
        val b = TrainingSample(
            videoFrames = floatArrayOf(9f, 8f, 7f),
            numFrames = 1,
            targetFeatures = floatArrayOf(4f, 5f),
            numTargetFrames = 1,
        )
        assertNotEquals(a, b)
    }

    @Test
    fun differentFrameCountIsNotEqual() {
        val a = TrainingSample(floatArrayOf(1f), 1, floatArrayOf(2f), 1)
        val b = TrainingSample(floatArrayOf(1f), 2, floatArrayOf(2f), 1)
        assertNotEquals(a, b)
    }

    @Test
    fun sameInstanceIsEqual() {
        val a = TrainingSample(floatArrayOf(1f), 1, floatArrayOf(2f), 1)
        assertTrue(a == a)
    }
}
