package com.hereliesaz.liperty.ml

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * Tests for [parseInputShape] — the helper that maps a 5-dim input shape from
 * a `ModelEngine` into the explicit (numFrames, inputHeight, inputWidth,
 * numChannels) tuple `VSRInference` uses.
 *
 * Adding NTCHW here so the Auto-AVSR ONNX (shape `[B, T, 1, 88, 88]`,
 * NTCHW layout) plugs into the existing `OnnxModelEngine` cleanly. NTHWC
 * and NCTHW were already supported.
 */
class InputLayoutTest {

    @Test
    fun nthwcShapeParsesCorrectly() {
        val dims = parseInputShape(intArrayOf(1, 50, 64, 128, 3), InputLayout.NTHWC)
        assertNotNull(dims)
        assertEquals(50, dims!!.numFrames)
        assertEquals(64, dims.inputHeight)
        assertEquals(128, dims.inputWidth)
        assertEquals(3, dims.numChannels)
    }

    @Test
    fun ncthwShapeParsesCorrectly() {
        val dims = parseInputShape(intArrayOf(1, 3, 16, 224, 224), InputLayout.NCTHW)
        assertNotNull(dims)
        assertEquals(3, dims!!.numChannels)
        assertEquals(16, dims.numFrames)
        assertEquals(224, dims.inputHeight)
        assertEquals(224, dims.inputWidth)
    }

    @Test
    fun ntchwShapeParsesAutoAvsrInput() {
        // Auto-AVSR / VSR-ML LRS3 visual encoder: (B, T, 1, 88, 88)
        val dims = parseInputShape(intArrayOf(1, 60, 1, 88, 88), InputLayout.NTCHW)
        assertNotNull(dims)
        assertEquals(60, dims!!.numFrames)
        assertEquals(1, dims.numChannels)
        assertEquals(88, dims.inputHeight)
        assertEquals(88, dims.inputWidth)
    }

    @Test
    fun shortShapeReturnsNull() {
        // Engine not yet initialized (or wrong shape) — caller must keep its
        // existing defaults.
        val dims = parseInputShape(intArrayOf(1, 16, 39), InputLayout.NTHWC)
        assertNull(dims)
    }
}
