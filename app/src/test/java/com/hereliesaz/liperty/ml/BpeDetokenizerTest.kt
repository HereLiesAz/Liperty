package com.hereliesaz.liperty.ml

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Tests for [BpeDetokenizer] — turns a sequence of SentencePiece BPE
 * token IDs (as produced by the AV-HuBERT seq2seq decoder) into text.
 *
 * The dictionary uses SentencePiece's `▁` (U+2581) as the explicit
 * word-boundary marker: any token starting with `▁` opens a new word;
 * tokens without `▁` are suffixes that attach to the previous one.
 * Special tokens (`<s>`, `<pad>`, `</s>`, `<unk>`) are dropped.
 */
class BpeDetokenizerTest {

    /** Minimal AV-HuBERT-style dict for tests. Indices 0..3 are the specials
     *  in fairseq's canonical order. */
    private val dict = listOf(
        "<s>", "<pad>", "</s>", "<unk>",
        "▁the", "▁cat", "▁sat", "▁on", "▁mat",
        "s", "ing", "▁i", "'", "ve",
    )

    @Test
    fun joinsWordPiecesIntoSpaceSeparatedText() {
        // [▁the, ▁cat, ▁sat, ▁on, ▁the, ▁mat]
        val ids = intArrayOf(4, 5, 6, 7, 4, 8)
        assertEquals("the cat sat on the mat", BpeDetokenizer.detokenize(ids, dict))
    }

    @Test
    fun attachesSuffixTokensWithoutSpace() {
        // [▁the, ▁cat, s] -> "the cats"
        val ids = intArrayOf(4, 5, 9)
        assertEquals("the cats", BpeDetokenizer.detokenize(ids, dict))
    }

    @Test
    fun chainsMultipleSuffixesOnSameWord() {
        // [▁sat, ing] -> "sating"  (BPE composition, regardless of whether
        // it makes English sense — the detokenizer's job is to reverse the
        // tokenization, not to validate the linguistics)
        val ids = intArrayOf(6, 10)
        assertEquals("sating", BpeDetokenizer.detokenize(ids, dict))
    }

    @Test
    fun mergesContractionTokensCorrectly() {
        // [▁i, ', ve] -> "i've"  (typical English contraction in BPE)
        val ids = intArrayOf(11, 12, 13)
        assertEquals("i've", BpeDetokenizer.detokenize(ids, dict))
    }

    @Test
    fun dropsBosEosPadUnkSpecials() {
        // [<s>, ▁the, ▁cat, </s>] -> "the cat"  (bos and eos at the boundaries
        // come out of the decoder's autoregressive loop; we strip them)
        val ids = intArrayOf(0, 4, 5, 2)
        assertEquals("the cat", BpeDetokenizer.detokenize(ids, dict))
        // <pad> and <unk> mid-sequence should also disappear silently.
        val ids2 = intArrayOf(0, 4, 1, 5, 3, 2)
        assertEquals("the cat", BpeDetokenizer.detokenize(ids2, dict))
    }

    @Test
    fun handlesEmptySequence() {
        assertEquals("", BpeDetokenizer.detokenize(intArrayOf(), dict))
    }

    @Test
    fun handlesOnlySpecials() {
        // [<s>, </s>] — common output when the decoder bails immediately
        // (encoder features carry no signal). Result is an empty string,
        // not a stray ▁.
        assertEquals("", BpeDetokenizer.detokenize(intArrayOf(0, 2), dict))
    }

    @Test
    fun dropsEspnetSpecialsForSyncVsrVocab() {
        // SyncVSR / Auto-AVSR use ESPnet's special-token convention:
        // <blank>, <unk>, <eos>. ESPnet's attention decoder reuses <eos>
        // as <sos>, so the greedy loop keeps the terminating <eos> in
        // its emitted token list — the detokenizer must drop it.
        val espnetDict = listOf(
            "<blank>", "<unk>", "▁the", "▁cat", "<eos>",
        )
        // [<eos>(bos), ▁the, ▁cat, <eos>] -> "the cat"
        val ids = intArrayOf(4, 2, 3, 4)
        assertEquals("the cat", BpeDetokenizer.detokenize(ids, espnetDict))
        // <blank> mid-sequence (rarely emitted by attention decoder, but
        // defensive) should also disappear.
        val idsWithBlank = intArrayOf(4, 2, 0, 3, 4)
        assertEquals("the cat", BpeDetokenizer.detokenize(idsWithBlank, espnetDict))
    }

    @Test
    fun ignoresOutOfRangeIds() {
        // Defensive: if the decoder somehow emits an id beyond the vocab,
        // skip it instead of crashing.
        val ids = intArrayOf(4, 9999, 5)
        assertEquals("the cat", BpeDetokenizer.detokenize(ids, dict))
        val idsNeg = intArrayOf(4, -1, 5)
        assertEquals("the cat", BpeDetokenizer.detokenize(idsNeg, dict))
    }
}
