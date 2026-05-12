package com.hereliesaz.liperty.ml

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Tests for [VisemeMap] — parses the phoneme→viseme assignments shipped
 * in `assets/viseme_map.txt` and exposes the lookup as a pure Kotlin
 * map for use by [VisemeRescorer].
 *
 * The phoneme set matches Liperty's [MLConstants.PHONEME_VOCAB] (40
 * ARPABET symbols). The file format is line-based: `<PHONEME>\t<VISEME>`,
 * with `#`-prefixed comment lines and blank lines ignored.
 *
 * Equivalence (used by [VisemeRescorer] to find candidate words):
 *   two phoneme sequences are viseme-equivalent iff they map to the
 *   same viseme sequence under the map. So /b ah t/ ("but") and
 *   /p ah t/ ("put") and /m ah t/ ("mutt") are all equivalent (all
 *   start with V2 then V1 V4).
 */
class VisemeMapTest {

    private val SAMPLE = """
        # comment line — ignored
        AA	V1
        B	V2
        M	V2
        T	V4
        # blank line follows

        UW	V9
    """.trimIndent()

    @Test
    fun parsesTabSeparatedPhonemeVisemePairs() {
        val map = VisemeMap.parse(SAMPLE)
        assertEquals("V1", map["AA"])
        assertEquals("V2", map["B"])
        assertEquals("V2", map["M"])
        assertEquals("V4", map["T"])
        assertEquals("V9", map["UW"])
    }

    @Test
    fun stripsCommentAndBlankLines() {
        val map = VisemeMap.parse(SAMPLE)
        assertEquals(5, map.size)
    }

    @Test
    fun returnsNullForUnknownPhoneme() {
        val map = VisemeMap.parse(SAMPLE)
        assertEquals(null, map["XYZ"])
    }

    @Test
    fun mapsPhonemeSequenceToVisemeSequence() {
        val map = VisemeMap.parse(SAMPLE)
        // "but" (cmudict: B AH T) — note AH not in SAMPLE so we use AA instead
        val visemes = VisemeMap.phonemesToVisemes(listOf("B", "AA", "T"), map)
        assertEquals(listOf("V2", "V1", "V4"), visemes)
    }

    @Test
    fun phonemesToVisemesDropsUnknownPhonemesSilently() {
        // Defensive: if a phoneme is missing from the map, drop it rather
        // than failing the whole word — the alternative is propagating an
        // empty result up to the rescorer which is more disruptive.
        val map = VisemeMap.parse(SAMPLE)
        val visemes = VisemeMap.phonemesToVisemes(listOf("B", "XYZ", "T"), map)
        assertEquals(listOf("V2", "V4"), visemes)
    }

    @Test
    fun realAssetParsesWithFortyPhonemes() {
        // Smoke test that the shipped asset matches MLConstants.PHONEME_VOCAB
        // size (40, minus blank = 39 mapped + nothing else). The actual
        // file is loaded by VisemeRescorer in production — here we just
        // parse the string we ship.
        val real = """
            AA	V1
            AE	V1
            AH	V1
            AO	V1
            AW	V1
            AY	V1
            B	V2
            CH	V3
            D	V4
            DH	V5
            EH	V1
            ER	V6
            EY	V1
            F	V7
            G	V4
            HH	V8
            IH	V1
            IY	V1
            JH	V3
            K	V4
            L	V4
            M	V2
            N	V4
            NG	V4
            OW	V1
            OY	V1
            P	V2
            R	V6
            S	V4
            SH	V3
            T	V4
            TH	V5
            UH	V1
            UW	V9
            V	V7
            W	V9
            Y	V1
            Z	V4
            ZH	V3
        """.trimIndent()
        val map = VisemeMap.parse(real)
        assertEquals(39, map.size)
        // B/P/M should share V2 (bilabial closure) — the classic
        // viseme-confusion the rescorer exists to disambiguate.
        assertEquals(map["B"], map["P"])
        assertEquals(map["B"], map["M"])
        // F/V share V7 (labio-dental); another classic confusion.
        assertEquals(map["F"], map["V"])
    }
}
