package com.hereliesaz.liperty.ml

import kotlin.math.log

class BeamSearchDecoder(
    private val vocabulary: List<String>,
    private val beamWidth: Int = 10,
    private val blankIndex: Int = 0
) {

    // Simple vocabulary for testing/prototype (Default)
    constructor(beamWidth: Int = 10) : this(
        listOf(
            "_", // Blank
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
            " "
        ),
        beamWidth,
        0
    )

    data class BeamState(
        val text: String,
        val score: Double, // Log probability
        val lastIndex: Int
    )

    /**
     * Decodes a sequence of probabilities (T x V) into a string using Beam Search.
     * T = time steps, V = vocabulary size.
     */
    fun decode(probabilities: Array<FloatArray>): String {
        if (probabilities.isEmpty()) return ""

        var beams = listOf(BeamState("", 0.0, -1))

        for (timeStep in probabilities) {
            val nextBeams = mutableListOf<BeamState>()

            for (beam in beams) {
                // For each token in vocabulary
                for (v in probabilities[0].indices) {
                    val prob = timeStep[v]
                    if (prob <= 0) continue // Skip zero probability to avoid log(0)

                    val logProb = log(prob.toDouble(), Math.E)
                    val newScore = beam.score + logProb

                    // CTC Logic
                    var newText = beam.text
                    if (v != blankIndex) {
                        if (v != beam.lastIndex || beam.lastIndex == blankIndex) {
                            // Append if different from last non-blank, or if separated by blank
                            // Note: This logic is slightly simplified. Standard CTC tracks two scores (blank ending vs non-blank ending).
                            // For a simple beam search prototype without merging identical paths perfectly:
                            if (v < vocabulary.size) {
                                newText += vocabulary[v]
                            }
                        }
                    }

                    nextBeams.add(BeamState(newText, newScore, v))
                }
            }

            // Pruning: Merge identical texts and keep top K
            // Ideally, we sum probabilities of identical texts, but here we take max for simplicity (Viterbi-like)
            val mergedBeams = nextBeams
                .groupBy { it.text }
                .map { (text, states) ->
                    // Standard CTC sums probabilities here (log-sum-exp).
                    // We will take the max score for this simplified version.
                    val bestState = states.maxByOrNull { it.score }!!
                    BeamState(text, bestState.score, bestState.lastIndex)
                }
                .sortedByDescending { it.score }
                .take(beamWidth)

            beams = mergedBeams
        }

        return beams.firstOrNull()?.text ?: ""
    }
}
