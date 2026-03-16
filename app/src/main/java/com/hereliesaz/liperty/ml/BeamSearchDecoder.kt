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
        val lastToken: String = "",         // last non-blank token emitted (for LM scoring)
        val probBlank: Double = 0.0,        // probability that path ends in blank
        val probNonBlank: Double = 0.0      // probability that path ends in non-blank
    ) {
        val totalProb: Double get() = probBlank + probNonBlank
        val score: Double get() = if (totalProb > 0) Math.log(totalProb) else -Double.MAX_VALUE
    }

    /**
     * Decodes a sequence of probabilities (T × V) into a string using CTC Beam Search
     * with prefix merging and optional bigram language-model rescoring.
     *
     * LM integration: when [languageModel] is non-null, each token extension receives
     *   a multiplicative bonus of `(1 + lmWeight × P_lm(token | prevToken))`
     * so the LM gently steers the search without dominating the acoustic model.
     */
    fun decode(probabilities: Array<FloatArray>): String {
        if (probabilities.isEmpty()) return ""

        var beams = listOf(BeamState(text = "", lastToken = "", probBlank = 1.0, probNonBlank = 0.0))

        for (timeStep in probabilities) {
            val nextBeams = mutableMapOf<String, BeamState>()

            for (beam in beams) {
                // 1. Path continues through blank
                val pBlank = timeStep[blankIndex].toDouble()
                val cur = nextBeams.getOrDefault(beam.text, BeamState(beam.text, beam.lastToken))
                nextBeams[beam.text] = cur.copy(
                    probBlank = cur.probBlank + beam.totalProb * pBlank
                )

                // 2. Path extends with a non-blank token
                for (v in timeStep.indices) {
                    if (v == blankIndex) continue

                    val char = if (v < vocabulary.size) vocabulary[v] else ""
                    if (char.isEmpty()) continue

                    // LM bonus: favour phoneme transitions that are common in English
                    val lmFactor = if (languageModel != null && beam.lastToken.isNotEmpty()) {
                        1.0 + lmWeight * languageModel.getScore(beam.lastToken, char)
                    } else 1.0
                    val pToken = timeStep[v].toDouble() * lmFactor

                    val newText = beam.text + char

                    if (beam.text.endsWith(char)) {
                        // Repeat token: only allowed if last arc was blank (CTC rule)
                        val repeatEntry = nextBeams.getOrDefault(newText, BeamState(newText, char))
                        nextBeams[newText] = repeatEntry.copy(
                            probNonBlank = repeatEntry.probNonBlank + beam.probBlank * pToken
                        )
                        // Collapsing: same token following same token stays on current prefix
                        val stayEntry = nextBeams.getOrDefault(beam.text, BeamState(beam.text, beam.lastToken))
                        nextBeams[beam.text] = stayEntry.copy(
                            probNonBlank = stayEntry.probNonBlank + beam.probNonBlank * pToken
                        )
                    } else {
                        val newEntry = nextBeams.getOrDefault(newText, BeamState(newText, char))
                        nextBeams[newText] = newEntry.copy(
                            probNonBlank = newEntry.probNonBlank + beam.totalProb * pToken
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
