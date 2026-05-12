package com.hereliesaz.liperty.ml

/**
 * Reverses the SentencePiece BPE tokenization that AV-HuBERT's seq2seq
 * decoder produces. See `BpeDetokenizerTest` for the contract.
 *
 * The fairseq dict that ships next to the V3 decoder has `▁` (U+2581) as
 * the explicit word-boundary marker. Tokens starting with `▁` open a new
 * word; everything else attaches to the previous one. The fairseq specials
 * `<s>`, `</s>`, `<pad>`, `<unk>` are dropped wherever they appear.
 */
object BpeDetokenizer {

    private const val SP_SPACE = "▁"   // SentencePiece `▁`

    // Union of fairseq (AV-HuBERT) and espnet (SyncVSR / Auto-AVSR)
    // special tokens. Stripped wherever they appear in the decoded
    // sequence. <eos> and <sos> matter here because espnet's attention
    // decoder reuses the same id for both sentence boundaries, and our
    // greedy loop keeps the terminating EOS in the token list it returns.
    private val SPECIALS = setOf(
        "<s>", "</s>", "<pad>", "<unk>",
        "<blank>", "<eos>", "<sos>", "<sos/eos>",
    )

    fun detokenize(tokenIds: IntArray, dict: List<String>): String {
        val sb = StringBuilder()
        for (id in tokenIds) {
            if (id < 0 || id >= dict.size) continue
            val sym = dict[id]
            if (sym in SPECIALS) continue
            sb.append(sym)
        }
        return sb.toString().replace(SP_SPACE, " ").trim()
    }
}
