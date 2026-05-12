package com.hereliesaz.liperty.ml

/**
 * "Chaplin's second AI" specialized for visual ASR: takes a CTC-decoded
 * sentence, expands each word to its viseme-equivalent candidates
 * (words that produce indistinguishable lip motion), and uses an
 * external [LanguageModelScorer] to pick the most English-plausible
 * substitution.
 *
 * Why this exists: lip-reading encoders make systematic confusions
 * between visemically-equivalent phonemes (b/p/m, f/v, t/n/d, etc.).
 * A pure-CTC beam search has no way to recover — once the encoder
 * commits to a wrong viseme-equivalent, the decoder propagates it.
 * Post-decoding rescoring with viseme-aware candidate expansion lets
 * the LM "re-vote" within the visual ambiguity class.
 *
 * Worked example:
 *   CTC emits "bird flew tasty". /t/ and /n/ share a viseme (V4 —
 *   alveolar), so the lip motion that produced "tasty" is equally
 *   consistent with "nasty". The LM rescores both candidates in
 *   context. "bird flew nasty" is far more English-plausible than
 *   "bird flew tasty" (a child's lunch box and a horror sentence), so
 *   the rescorer flips it.
 *
 * Beam search over per-position candidate sets. Search complexity is
 * `O(N * K * B)` LM calls per sentence (N = words, K = candidates per
 * word, B = beam width). For typical N≤10, K=8, B=8, that's ~640 LM
 * calls — well under a second with KenLM on device.
 *
 * If the rescored best doesn't STRICTLY beat the original input score,
 * the original is returned (we err on the side of not over-correcting).
 */
class VisemeRescorer(
    private val index: VisemeIndex,
    private val lm: LanguageModelScorer,
    private val beamWidth: Int = 8,
    private val candidatesPerWord: Int = 8,
    /** Per-position bonus for matching the input word — breaks LM-score
     *  ties in favor of minimal changes. Conceptually the rescorer's
     *  "do no harm" prior: when the LM can't tell candidates apart, keep
     *  what the encoder produced. */
    private val inputBias: Float = 1e-4f,
) {
    private data class Beam(val words: List<String>, val score: Float)

    fun rescore(sentence: String): String {
        val tokens = sentence.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }
        if (tokens.isEmpty()) return ""

        val inputWords = tokens.map { it.lowercase() }

        // Per-position viseme-equivalent candidate sets (lowercased for output).
        val candidates: List<List<String>> = tokens.map { tok ->
            index.candidatesFor(tok, candidatesPerWord).map { it.lowercase() }
        }

        // Helper: cumulative input-bias bonus for a prefix of chosen words.
        fun bonus(prefix: List<String>): Float {
            var n = 0
            for (i in prefix.indices) if (prefix[i] == inputWords[i]) n++
            return n * inputBias
        }

        // Beam search left-to-right with input-bias tiebreak baked into each
        // beam's score.
        var beams: List<Beam> = candidates[0]
            .map { w ->
                val words = listOf(w)
                Beam(words, lm.score(words) + bonus(words))
            }
            .sortedByDescending { it.score }
            .take(beamWidth)

        for (pos in 1 until candidates.size) {
            val expanded = ArrayList<Beam>(beams.size * candidates[pos].size)
            for (b in beams) {
                for (next in candidates[pos]) {
                    val newWords = b.words + next
                    expanded.add(Beam(newWords, lm.score(newWords) + bonus(newWords)))
                }
            }
            beams = expanded.sortedByDescending { it.score }.take(beamWidth)
        }

        val inputScore = lm.score(inputWords) + bonus(inputWords)
        val best = beams.firstOrNull() ?: return inputWords.joinToString(" ")
        return if (best.score > inputScore) {
            best.words.joinToString(" ")
        } else {
            // Best alternative doesn't strictly beat the (input-biased) input
            // score. Keep what the decoder produced — over-correction is
            // worse than under-correction for accessibility text.
            inputWords.joinToString(" ")
        }
    }
}
