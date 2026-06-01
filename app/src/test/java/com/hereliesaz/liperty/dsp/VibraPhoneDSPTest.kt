package com.hereliesaz.liperty.dsp

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config
import kotlin.math.cos
import kotlin.math.hypot

/**
 * Regression tests for [VibraPhoneDSP.trambaBandwidthExpansion].
 *
 * AndroidJUnit4 + sdk=34 because the class's companion `init` calls
 * `System.loadLibrary` (caught) and `android.util.Log` — both need Robolectric.
 */
@RunWith(AndroidJUnit4::class)
@Config(sdk = [34])
class VibraPhoneDSPTest {

    private val dsp = VibraPhoneDSP()

    /**
     * TRAMBA must only reconstruct frequencies at/above the 3 kHz cutoff; bins
     * strictly below the cutoff must pass through untouched. The previous
     * implementation iterated the full upper spectrum and, for negative-frequency
     * bins (i > N/2), wrote a "conjugate mirror" at N-i that landed in the LOW
     * half — clobbering the sub-cutoff content (including the input tone). This
     * test fails on that bug and passes on the fix.
     */
    @Test
    fun trambaPreservesBelowCutoffFrequencies() {
        val n = 512
        val sr = VibraPhoneDSP.SAMPLE_RATE          // 16000
        val cutoffIdx = (3000L * n / sr).toInt()    // 96 — bin of the 3 kHz cutoff

        // Pure cosine at bin 30 (~937 Hz), comfortably below the cutoff.
        val toneBin = 30
        assertTrue("tone must sit below the cutoff", toneBin < cutoffIdx)
        val input = FloatArray(n) { t -> cos(2.0 * Math.PI * toneBin * t / n).toFloat() }

        val inSpec = dsp.fft(input)
        val out = dsp.trambaBandwidthExpansion(input)
        assertEquals("output length preserved", n, out.size)
        val outSpec = dsp.fft(out)

        // Reference magnitude (the tone) used to scale tolerances.
        val toneMag = hypot(inSpec[2 * toneBin].toDouble(), inSpec[2 * toneBin + 1].toDouble())
        assertTrue("sanity: tone has energy", toneMag > 1.0)

        // Every positive-frequency bin strictly below the cutoff must be preserved.
        for (i in 1 until cutoffIdx) {
            val inMag = hypot(inSpec[2 * i].toDouble(), inSpec[2 * i + 1].toDouble())
            val outMag = hypot(outSpec[2 * i].toDouble(), outSpec[2 * i + 1].toDouble())
            assertEquals("below-cutoff bin $i must be preserved", inMag, outMag, toneMag * 0.02)
        }
    }

    /** Output stays finite and same-length for a multi-tone input. */
    @Test
    fun trambaOutputIsFiniteAndSameLength() {
        val n = 256
        val input = FloatArray(n) { t ->
            (cos(2.0 * Math.PI * 50 * t / n) + 0.3 * cos(2.0 * Math.PI * 100 * t / n)).toFloat()
        }
        val out = dsp.trambaBandwidthExpansion(input)
        assertEquals(n, out.size)
        for (v in out) assertTrue("sample must be finite", v.isFinite())
    }
}
