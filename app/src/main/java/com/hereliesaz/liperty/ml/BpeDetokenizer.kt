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

    private val SPECIALS = setOf("<s>", "</s>", "<pad>", "<unk>")

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
