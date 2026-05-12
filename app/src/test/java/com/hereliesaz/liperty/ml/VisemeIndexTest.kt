package com.hereliesaz.liperty.ml

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Tests for [VisemeIndex] — the runtime lookup structure that supports
 * [VisemeRescorer]. Built from the JSON output of
 * `tools/build_viseme_index.py` (`viseme_index.json` asset).
 *
 * The on-disk JSON is just the inverse map (viseme_seq → words);
 * [VisemeIndex] builds the forward map (word → viseme_seq) at load time
 * so the rescorer can do O(1) lookups in either direction.
 */
class VisemeIndexTest {

    /** Tiny inverse index mirroring the asset format produced by
     *  `tools/build_viseme_index.py`. The key is a space-separated
     *  viseme code sequence; the value is the list of words. */
    private val sample: Map<String, List<String>> = mapOf(
        "V2 V1 V4" to listOf("BAT", "MAT", "PAT"),       // bilabial + vowel + alveolar
        "V7 V1 V4" to listOf("FAT", "VAT"),              // labio-dental
        "V4 V1 V4 V4 V1" to listOf("NASTY", "TASTY"),    // alveolar
        "V1" to listOf("A", "I", "O"),                   // single vowel
    )

    @Test
    fun candidatesIncludeAllWordsSharingViseme() {
        val idx = VisemeIndex(sample)
        assertEquals(setOf("BAT", "MAT", "PAT"), idx.candidatesFor("BAT").toSet())
        assertEquals(setOf("BAT", "MAT", "PAT"), idx.candidatesFor("PAT").toSet())
        assertEquals(setOf("FAT", "VAT"), idx.candidatesFor("FAT").toSet())
    }

    @Test
    fun candidatesIncludeInputWordItself() {
        // The input word is part of its own equivalence class; the
        // rescorer needs to be able to pick it as the winner.
        val idx = VisemeIndex(sample)
        assertTrue("input word in candidate set", "NASTY" in idx.candidatesFor("NASTY"))
    }

    @Test
    fun candidatesAreUppercaseRegardlessOfInputCasing() {
        val idx = VisemeIndex(sample)
        // The index is built from uppercase cmudict words; queries must
        // normalize so callers don't have to remember.
        assertEquals(setOf("BAT", "MAT", "PAT"), idx.candidatesFor("bat").toSet())
        assertEquals(setOf("BAT", "MAT", "PAT"), idx.candidatesFor("Bat").toSet())
    }

    @Test
    fun unknownWordReturnsSingletonOfTheInput() {
        // Words not in the index (proper nouns, slang, neologisms) should
        // pass through unchanged so the rescorer doesn't drop them.
        val idx = VisemeIndex(sample)
        assertEquals(listOf("DROPMORE"), idx.candidatesFor("dropmore"))
    }

    @Test
    fun candidateCountCanBeCapped() {
        // Some viseme sequences (e.g. a single common vowel) match
        // hundreds of words; the rescorer should cap to keep scoring
        // tractable. Cmudict's actual max is 1673; we test the small
        // sample but the same code path handles huge sets.
        val idx = VisemeIndex(sample)
        val capped = idx.candidatesFor("BAT", maxCandidates = 2)
        assertEquals(2, capped.size)
        assertTrue("input word always included", "BAT" in capped)
    }

    @Test
    fun parseJsonReadsAssetFormat() {
        // The asset is a flat JSON object: {viseme_seq: [word1, word2, ...]}.
        val json = """{"V2 V1 V4":["BAT","MAT","PAT"],"V1":["A","I"]}"""
        val idx = VisemeIndex.fromJson(json)
        assertEquals(setOf("BAT", "MAT", "PAT"), idx.candidatesFor("BAT").toSet())
        assertEquals(setOf("A", "I"), idx.candidatesFor("A").toSet())
    }
}
