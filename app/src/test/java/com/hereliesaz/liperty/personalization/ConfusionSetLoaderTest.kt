package com.hereliesaz.liperty.personalization

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config

/**
 * Unit tests for [ConfusionSetLoader.parse].
 *
 * Pure JSON in -> data class list out, but [org.json.JSONObject] is an
 * Android framework class (not a JVM class), so we need a Robolectric
 * runtime to get a working implementation instead of the stub that
 * throws "not mocked" on every method call.
 *
 * `@Config(sdk = [34])` follows the project-wide pattern for
 * `targetSdk = 37` (see CLAUDE.md "Robolectric SDK 37 trap").
 */
@RunWith(AndroidJUnit4::class)
@Config(sdk = [34])
class ConfusionSetLoaderTest {

    private val sampleJson = """
        {
          "metadata": {
            "viseme_count": 12,
            "total_sets": 2,
            "version": "v1"
          },
          "sets": [
            {
              "viseme_seq": "V2 V1A V4",
              "words": ["back", "man", "might", "bad", "mac", "pass"],
              "phoneme_seqs": [
                ["B","AE","K"], ["M","AE","N"], ["M","AY","T"],
                ["B","AE","D"], ["M","AE","K"], ["P","AE","S"]
              ],
              "score": 10.30
            },
            {
              "viseme_seq": "V5 V1B",
              "words": ["the", "they"],
              "phoneme_seqs": [["DH","AH"], ["DH","EY"]],
              "score": 10.31
            }
          ]
        }
    """.trimIndent()

    @Test
    fun parsesAllSetsWhenNoCaps() {
        val sets = ConfusionSetLoader.parse(sampleJson)
        assertEquals(2, sets.size)
        assertEquals("V2 V1A V4", sets[0].visemeSeq)
        assertEquals(6, sets[0].words.size)
        assertEquals(listOf("back", "man", "might", "bad", "mac", "pass"), sets[0].words)
        assertEquals(10.30f, sets[0].score, 1e-3f)

        assertEquals("V5 V1B", sets[1].visemeSeq)
        assertEquals(listOf("the", "they"), sets[1].words)
    }

    @Test
    fun preservesScoreOrderFromFile() {
        // The build script writes sets sorted by score DESCENDING. The
        // loader preserves that order. The UI's "top-N most useful"
        // logic depends on this; if loader started sorting, the result
        // could quietly differ.
        val sets = ConfusionSetLoader.parse(sampleJson)
        assertTrue("loader must preserve file order", sets[0].visemeSeq == "V2 V1A V4")
        assertTrue("loader must preserve file order", sets[1].visemeSeq == "V5 V1B")
    }

    @Test
    fun appliesMaxSetsCap() {
        val sets = ConfusionSetLoader.parse(sampleJson, maxSets = 1)
        assertEquals(1, sets.size)
        assertEquals("V2 V1A V4", sets[0].visemeSeq)
    }

    @Test
    fun appliesMaxSetSizeCap() {
        val sets = ConfusionSetLoader.parse(sampleJson, maxSetSize = 3)
        assertEquals(2, sets.size)
        assertEquals(3, sets[0].words.size)
        assertEquals(listOf("back", "man", "might"), sets[0].words)
        // 2-member set unaffected (cap > size).
        assertEquals(2, sets[1].words.size)
    }

    @Test
    fun dropsSetsThatWouldFallBelowTwoWordsAfterCap() {
        // Defensive: if maxSetSize is 1 the resulting "set" has nothing
        // to discriminate — drop it.
        val sets = ConfusionSetLoader.parse(sampleJson, maxSetSize = 1)
        assertEquals(0, sets.size)
    }

    @Test
    fun handlesMissingScoreGracefully() {
        // Tolerate sparse JSON — score missing -> 0.0f, set still
        // parseable.
        val json = """
            {"sets":[{"viseme_seq":"V2","words":["a","b"]}]}
        """.trimIndent()
        val sets = ConfusionSetLoader.parse(json)
        assertEquals(1, sets.size)
        assertEquals(0f, sets[0].score, 1e-6f)
    }
}
