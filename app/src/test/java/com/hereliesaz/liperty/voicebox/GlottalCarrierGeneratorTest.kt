package com.hereliesaz.liperty.voicebox

import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import kotlin.math.abs
import kotlin.math.roundToInt

class GlottalCarrierGeneratorTest {

    private lateinit var generator: GlottalCarrierGenerator

    @Before
    fun setUp() {
        generator = GlottalCarrierGenerator()
    }

    @Test
    fun `default F0 is 120 Hz`() {
        assertEquals(120f, generator.f0)
    }

    @Test
    fun `setF0 clamps to valid range`() {
        generator.setF0(50f)
        assertEquals(80f, generator.f0)

        generator.setF0(300f)
        assertEquals(200f, generator.f0)

        generator.setF0(150f)
        assertEquals(150f, generator.f0)
    }

    @Test
    fun `buffer amplitude stays within bounds`() {
        val buffer = FloatArray(4096)
        generator.generateBuffer(buffer)

        for (sample in buffer) {
            assertTrue("Sample $sample exceeds [-1, 1]", sample >= -1f && sample <= 1f)
        }
    }

    @Test
    fun `buffer contains non-zero samples during open phase`() {
        val buffer = FloatArray(4096)
        generator.generateBuffer(buffer)

        val nonZero = buffer.count { abs(it) > 0.001f }
        assertTrue("Expected some non-zero samples, got $nonZero", nonZero > 0)
    }

    @Test
    fun `open phase is approximately 40 percent of period`() {
        generator.setF0(100f)
        // At 100 Hz, period = 16000/100 = 160 samples, open phase = 64 samples
        val buffer = FloatArray(160) // exactly one period
        generator.generateBuffer(buffer)

        val nonZero = buffer.count { abs(it) > 0.001f }
        val expectedOpen = (160 * 0.4).roundToInt()
        // Allow +-2 samples tolerance for float rounding
        assertTrue(
            "Expected ~$expectedOpen non-zero samples in one period, got $nonZero",
            abs(nonZero - expectedOpen) <= 2
        )
    }

    @Test
    fun `phase continuity across multiple buffers`() {
        val buf1 = FloatArray(100)
        val buf2 = FloatArray(100)
        generator.generateBuffer(buf1)
        generator.generateBuffer(buf2)

        // Concatenate and check there are no discontinuities larger than
        // what a normal pulse transition would produce
        val lastOfBuf1 = buf1.last()
        val firstOfBuf2 = buf2.first()
        val jump = abs(lastOfBuf1 - firstOfBuf2)
        // A normal pulse has smooth transitions; jumps > 0.5 indicate a phase reset
        assertTrue(
            "Phase discontinuity detected: jump=$jump between buffers",
            jump < 0.5f
        )
    }

    @Test
    fun `period matches F0 frequency`() {
        generator.setF0(160f)
        // Period should be 16000/160 = 100 samples
        val buffer = FloatArray(1000) // ~10 periods
        generator.generateBuffer(buffer)

        // Find zero-crossings from positive to zero (end of open phase)
        var zeroCrossings = 0
        for (i in 1 until buffer.size) {
            if (buffer[i - 1] > 0.01f && abs(buffer[i]) < 0.01f) {
                zeroCrossings++
            }
        }

        // Expect ~10 periods in 1000 samples at 160 Hz
        assertTrue(
            "Expected ~10 pulse periods, detected $zeroCrossings zero crossings",
            zeroCrossings in 8..12
        )
    }
}
