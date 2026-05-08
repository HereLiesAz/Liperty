package com.hereliesaz.liperty.ml

class BeamSearchDecoder(
    private val vocabulary: List<String>,
    private val beamWidth: Int = 10,
    private val blankIndex: Int = 0,
    private val languageModel: LanguageModel? = null,
    private val lmWeight: Float = 0.08f
) {

    constructor(beamWidth: Int = 10) : this(
        MLConstants.PHONEME_VOCAB, beamWidth, 0,
        languageModel = LanguageModel()
    )

    data class BeamState(
        val text: String,
        val lastToken: String = "",
        val logProbBlank: Double = Double.NEGATIVE_INFINITY,
        val logProbNonBlank: Double = Double.NEGATIVE_INFINITY
    ) {
        val logTotalProb: Double get() = logSumExp(logProbBlank, logProbNonBlank)
        val score: Double get() = logTotalProb
    }

    companion object {
        private fun logSumExp(a: Double, b: Double): Double {
            if (a == Double.NEGATIVE_INFINITY) return b
            if (b == Double.NEGATIVE_INFINITY) return a
            val max = maxOf(a, b)
            return max + Math.log(Math.exp(a - max) + Math.exp(b - max))
        }
    }

    /**
     * Decodes a sequence of probabilities (T x V) into a string using CTC Beam Search
     * with prefix merging and optional bigram language-model rescoring.
     *
     * Uses log-domain arithmetic throughout to avoid floating-point underflow on long
     * sequences. All probabilities are represented as natural logarithms.
     *
     * LM integration: when [languageModel] is non-null, each token extension receives
     *   an additive log bonus of `log(1 + lmWeight * P_lm(token | prevToken))`
     * so the LM gently steers the search without dominating the acoustic model.
     */
    fun decode(probabilities: Array<FloatArray>): String {
        if (probabilities.isEmpty()) return ""

        // Seed beam: the empty prefix has log(1.0) = 0.0 probability of ending in blank.
        var beams = listOf(
            BeamState(
                text = "",
                lastToken = "",
                logProbBlank = 0.0,
                logProbNonBlank = Double.NEGATIVE_INFINITY
            )
        )

        for (timeStep in probabilities) {
            val nextBeams = mutableMapOf<String, BeamState>()

            for (beam in beams) {
                // 1. Path continues through blank
                val pBlank = timeStep[blankIndex].toDouble()
                val logPBlank = if (pBlank > 0.0) Math.log(pBlank) else Double.NEGATIVE_INFINITY
                val curBlank = nextBeams.getOrDefault(beam.text, BeamState(beam.text, beam.lastToken))
                nextBeams[beam.text] = curBlank.copy(
                    logProbBlank = logSumExp(curBlank.logProbBlank, beam.logTotalProb + logPBlank)
                )

                // 2. Path extends with a non-blank token
                for (v in timeStep.indices) {
                    if (v == blankIndex) continue

                    val char = if (v < vocabulary.size) vocabulary[v] else ""
                    if (char.isEmpty()) continue

                    val pToken = timeStep[v].toDouble()
                    if (pToken <= 0.0) continue
                    var logPToken = Math.log(pToken)

                    // LM bonus: favour phoneme transitions common in English
                    if (languageModel != null && beam.lastToken.isNotEmpty()) {
                        val lmScore = languageModel.getScore(beam.lastToken, char)
                        val lmFactor = 1.0 + lmWeight * lmScore
                        if (lmFactor > 0.0) logPToken += Math.log(lmFactor)
                    }

                    val newText = if (beam.text.isEmpty()) char else beam.text + " " + char

                    if (beam.text.endsWith(char)) {
                        // Repeat token: only allowed if last arc was blank (CTC rule)
                        val repeatEntry = nextBeams.getOrDefault(newText, BeamState(newText, char))
                        nextBeams[newText] = repeatEntry.copy(
                            logProbNonBlank = logSumExp(
                                repeatEntry.logProbNonBlank,
                                beam.logProbBlank + logPToken
                            )
                        )
                        // Collapsing: same token following same token stays on current prefix
                        val stayEntry = nextBeams.getOrDefault(beam.text, BeamState(beam.text, beam.lastToken))
                        nextBeams[beam.text] = stayEntry.copy(
                            logProbNonBlank = logSumExp(
                                stayEntry.logProbNonBlank,
                                beam.logProbNonBlank + logPToken
                            )
                        )
                    } else {
                        val newEntry = nextBeams.getOrDefault(newText, BeamState(newText, char))
                        nextBeams[newText] = newEntry.copy(
                            logProbNonBlank = logSumExp(
                                newEntry.logProbNonBlank,
                                beam.logTotalProb + logPToken
                            )
                        )
                    }
                }
            }

            beams = nextBeams.values
                .sortedByDescending { it.score }
                .take(beamWidth)
        }

        return beams.firstOrNull()?.text ?: ""
    }
}
