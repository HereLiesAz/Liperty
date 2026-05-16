package com.hereliesaz.liperty.voicebox.cloning

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Pure JUnit tests for [VoiceQualityTier.compute]. Drives the live
 * "voice quality" indicator shown to users during coached recording.
 */
class VoiceQualityTierTest {

    @Test
    fun `zero clips returns NONE`() {
        assertEquals(VoiceQualityTier.NONE, VoiceQualityTier.compute(0, 0))
    }

    @Test
    fun `one short clip returns NONE if under 5 seconds`() {
        assertEquals(VoiceQualityTier.NONE, VoiceQualityTier.compute(1, 4))
    }

    @Test
    fun `one clip past 5 seconds returns INITIAL`() {
        assertEquals(VoiceQualityTier.INITIAL, VoiceQualityTier.compute(1, 5))
        assertEquals(VoiceQualityTier.INITIAL, VoiceQualityTier.compute(1, 30))
        // One very long clip is still INITIAL — centroid stability
        // requires multiple clips, not just more audio.
        assertEquals(VoiceQualityTier.INITIAL, VoiceQualityTier.compute(1, 600))
    }

    @Test
    fun `three clips and 30 seconds returns BASIC`() {
        assertEquals(VoiceQualityTier.BASIC, VoiceQualityTier.compute(3, 30))
        // 2 clips no matter how long → stays at INITIAL.
        assertEquals(VoiceQualityTier.INITIAL, VoiceQualityTier.compute(2, 600))
        // 3 clips but only 20s total → stays at INITIAL.
        assertEquals(VoiceQualityTier.INITIAL, VoiceQualityTier.compute(3, 20))
    }

    @Test
    fun `six clips and 90 seconds returns GOOD`() {
        assertEquals(VoiceQualityTier.GOOD, VoiceQualityTier.compute(6, 90))
        assertEquals(VoiceQualityTier.BASIC, VoiceQualityTier.compute(5, 90))
        assertEquals(VoiceQualityTier.BASIC, VoiceQualityTier.compute(6, 60))
    }

    @Test
    fun `twelve clips and 240 seconds returns EXCELLENT`() {
        assertEquals(VoiceQualityTier.EXCELLENT, VoiceQualityTier.compute(12, 240))
        assertEquals(VoiceQualityTier.EXCELLENT, VoiceQualityTier.compute(50, 1000))
        // 11 clips at any duration → stays at GOOD.
        assertEquals(VoiceQualityTier.GOOD, VoiceQualityTier.compute(11, 1000))
        // 12 clips but only 180s → stays at GOOD.
        assertEquals(VoiceQualityTier.GOOD, VoiceQualityTier.compute(12, 180))
    }

    @Test
    fun `filledDots increase monotonically with rank`() {
        var prev = -1
        for (tier in VoiceQualityTier.values().sortedBy { it.rank }) {
            val dots = tier.filledDots
            assert(dots >= prev) {
                "${tier.name} has $dots dots but previous rank had $prev"
            }
            prev = dots
        }
        assertEquals(4, VoiceQualityTier.EXCELLENT.filledDots)
    }

    @Test
    fun `tier thresholds are strictly increasing`() {
        // Each tier should require at least as many clips AND at
        // least as much speech as the previous tier — otherwise the
        // bands aren't monotonic in user effort and the UI can flicker.
        val tiers = VoiceQualityTier.values().sortedBy { it.rank }
        for (i in 1 until tiers.size) {
            assert(tiers[i].minClips >= tiers[i - 1].minClips) {
                "${tiers[i].name} requires fewer clips than ${tiers[i - 1].name}"
            }
            assert(tiers[i].minSeconds >= tiers[i - 1].minSeconds) {
                "${tiers[i].name} requires fewer seconds than ${tiers[i - 1].name}"
            }
        }
    }
}
