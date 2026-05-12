package com.hereliesaz.liperty.ml

import kotlin.math.ln

/**
 * CTC beam search over a SentencePiece-style subword vocabulary. Used by the
 * Auto-AVSR backend instead of [SubwordCTCDecoder] (greedy) when more than
 * one path through the lattice is worth exploring.
 *
 * The classical CTC prefix-search algorithm with two probabilities tracked
 * per beam:
 *   - p_blank:    the path's total log-prob ending in a blank
 *   - p_nonblank: the path's total log-prob ending in a non-blank token
 *
 * At each timestep, every active beam is extended by every vocab token. We
 * keep only the top [beamWidth] beams by total log-prob (logSumExp of
 * p_blank and p_nonblank).
 *
 * Detokenization is identical to [SubwordCTCDecoder]: beam token strings are
 * concatenated without separator (subword units join naturally) and `▁`
 * (U+2581) is replaced with a space at the end. Leading whitespace stripped.
 *
 * Why a separate class from the existing [BeamSearchDecoder]: that one joins
 * emissions with literal `" "` (right for ARPABET phonemes) and has
 * mandatory bigram-LM scoring (trained on phoneme bigrams). For
 * SentencePiece subwords we want concatenation + no LM.
 */
class SubwordCtcBeamDecoder(
    private val vocabulary: List<String>,
    private val beamWidth: Int = 8,
    private val blankIndex: Int = 0,
    /** Optional external language model for n-best rescoring. When non-null
     *  and [lmWeight] != 0, the surviving top-[beamWidth] beams are re-ranked
     *  by `CTC_logprob + lmWeight * LM.score(words)` before the winner is
     *  picked. Null (the default) disables rescoring — pure-CTC behavior. */
    private val lmScorer: LanguageModelScorer? = null,
    /** Weight on the LM score in the rescoring sum. Conventional values
     *  for n-gram shallow fusion are 0.3–0.7; tune offline on a dev set.
     *  Unused when [lmScorer] is null. */
    private val lmWeight: Float = 0.5f,
) {
    private data class Beam(
        val text: String,
        val lastIndex: Int,
        val logProbBlank: Double = NEG_INF,
        val logProbNonBlank: Double = NEG_INF,
    ) {
        val total: Double get() = logSumExp(logProbBlank, logProbNonBlank)
    }

    fun decode(probabilities: Array<FloatArray>): String {
        if (probabilities.isEmpty()) return ""

        // Initial beam: empty prefix, ending in blank with log-prob 0 (= prob 1).
        var beams: Map<Pair<String, Int>, Beam> = mapOf(
            ("" to -1) to Beam(text = "", lastIndex = -1, logProbBlank = 0.0)
        )

        for (timeStep in probabilities) {
            // Defensive: if this timestep has no probability mass on any
            // in-vocab token (blank included), propagate beams unchanged —
            // matches the greedy decoder's behavior of no-op'ing on
            // out-of-range argmax. Without this, malformed inputs (e.g.
            // logits that include OOR slots) collapse the beam to nothing.
            var inVocabMass = 0.0
            for (v in 0 until minOf(timeStep.size, vocabulary.size)) {
                if (timeStep[v] > 0f) { inVocabMass += timeStep[v] }
            }
            if (inVocabMass <= 0.0) continue

            val nextBeams = HashMap<Pair<String, Int>, Beam>()

            for ((_, beam) in beams) {
                // Extension 1: add a blank. Stays on the same prefix.
                val pBlank = timeStep.getOrZero(blankIndex)
                if (pBlank > 0.0) {
                    val key = beam.text to -1   // -1 marker = "ends with blank"
                    val existing = nextBeams[key]
                    val mergedBlank = logSumExp(
                        existing?.logProbBlank ?: NEG_INF,
                        beam.total + ln(pBlank),
                    )
                    nextBeams[key] = (existing ?: beam.copy(lastIndex = -1, logProbBlank = NEG_INF, logProbNonBlank = NEG_INF))
                        .copy(logProbBlank = mergedBlank)
                }

                // Extension 2: emit each non-blank token v.
                for (v in timeStep.indices) {
                    if (v == blankIndex) continue
                    if (v >= vocabulary.size) continue
                    val token = vocabulary[v]
                    if (token.isEmpty()) continue

                    val pToken = timeStep[v].toDouble()
                    if (pToken <= 0.0) continue
                    val logP = ln(pToken)

                    if (v == beam.lastIndex) {
                        // Same token as the last emission. Two paths:
                        //   (a) preceded by a blank → emit a NEW token (extends prefix)
                        //   (b) NOT preceded by a blank → CTC collapses; stays on same prefix
                        // (a) extend
                        val newText = beam.text + token
                        val keyA = newText to v
                        val existingA = nextBeams[keyA]
                        val mergedA = logSumExp(
                            existingA?.logProbNonBlank ?: NEG_INF,
                            beam.logProbBlank + logP,
                        )
                        nextBeams[keyA] = (existingA ?: Beam(newText, v))
                            .copy(logProbNonBlank = mergedA)
                        // (b) collapse — stays on current prefix, ending in non-blank
                        val keyB = beam.text to v
                        val existingB = nextBeams[keyB]
                        val mergedB = logSumExp(
                            existingB?.logProbNonBlank ?: NEG_INF,
                            beam.logProbNonBlank + logP,
                        )
                        nextBeams[keyB] = (existingB ?: beam.copy(lastIndex = v))
                            .copy(logProbNonBlank = mergedB)
                    } else {
                        // Different token. Always extends the prefix.
                        val newText = beam.text + token
                        val key = newText to v
                        val existing = nextBeams[key]
                        val merged = logSumExp(
                            existing?.logProbNonBlank ?: NEG_INF,
                            beam.total + logP,
                        )
                        nextBeams[key] = (existing ?: Beam(newText, v))
                            .copy(logProbNonBlank = merged)
                    }
                }
            }

            // Prune to top beamWidth by total log-prob.
            beams = nextBeams.entries
                .sortedByDescending { it.value.total }
                .take(beamWidth)
                .associate { it.toPair() }
        }

        // Optional n-best rescoring with an external LM. When disabled
        // (lmScorer null or lmWeight 0), this is a no-op and we keep
        // the original CTC argmax behavior.
        val best = if (lmScorer != null && lmWeight != 0f) {
            beams.values.maxByOrNull { beam ->
                beam.total + lmWeight * lmScorer.score(splitIntoWords(beam.text))
            }
        } else {
            beams.values.maxByOrNull { it.total }
        } ?: return ""
        return best.text
            .replace(WORD_BOUNDARY, ' ')
            .trimStart()
    }

    /**
     * SentencePiece-aware word splitter — raw beam text concatenates tokens
     * including their `▁` word-start markers (e.g. "▁the▁cat" or "▁i've"
     * with a contraction). Splitting on `▁` and dropping empties yields
     * `["the", "cat"]` — the LM's expected input.
     */
    private fun splitIntoWords(text: String): List<String> =
        text.split(WORD_BOUNDARY).filter { it.isNotEmpty() }

    private fun FloatArray.getOrZero(i: Int): Double =
        if (i in indices) this[i].toDouble() else 0.0

    companion object {
        const val WORD_BOUNDARY: Char = '▁'
        private val NEG_INF = Double.NEGATIVE_INFINITY

        private fun logSumExp(a: Double, b: Double): Double {
            if (a == NEG_INF) return b
            if (b == NEG_INF) return a
            val m = if (a > b) a else b
            return m + ln(Math.exp(a - m) + Math.exp(b - m))
        }
    }
}
