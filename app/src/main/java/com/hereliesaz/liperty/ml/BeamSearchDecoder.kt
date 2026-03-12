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
        val probBlank: Double = 0.0,    // Probability path ends in blank
        val probNonBlank: Double = 0.0  // Probability path ends in non-blank
    ) {
        val totalProb: Double get() = probBlank + probNonBlank
        val score: Double get() = if (totalProb > 0) Math.log(totalProb) else -Double.MAX_VALUE
    }

    /**
     * Decodes a sequence of probabilities (T x V) into a string using Beam Search with CTC prefix merging.
     * T = time steps, V = vocabulary size.
     */
    fun decode(probabilities: Array<FloatArray>): String {
        if (probabilities.isEmpty()) return ""

        // Initialize with empty prefix
        var beams = listOf(BeamState(text = "", probBlank = 1.0, probNonBlank = 0.0))

        for (timeStep in probabilities) {
            val nextBeams = mutableMapOf<String, BeamState>()

            for (beam in beams) {
                // 1. Path ends in blank
                val pBlank = timeStep[blankIndex].toDouble()
                val currentBlank = nextBeams.getOrDefault(beam.text, BeamState(beam.text))
                nextBeams[beam.text] = currentBlank.copy(
                    probBlank = currentBlank.probBlank + beam.totalProb * pBlank
                )

                // 2. Path ends in non-blank
                for (v in timeStep.indices) {
                    if (v == blankIndex) continue
                    
                    val pToken = timeStep[v].toDouble()
                    val char = if (v < vocabulary.size) vocabulary[v] else ""
                    val newText = beam.text + char
                    
                    if (char.isNotEmpty() && beam.text.endsWith(char)) {
                        // Repeat character: blank required between them to not collapse
                        // Current prefix (ending in non-blank) + repeat -> new prefix
                        val repeatEntry = nextBeams.getOrDefault(newText, BeamState(newText))
                        nextBeams[newText] = repeatEntry.copy(
                            probNonBlank = repeatEntry.probNonBlank + beam.probBlank * pToken
                        )
                        
                        // Current prefix (ending in non-blank) + same char -> stays current prefix (collapsed)
                        val stayEntry = nextBeams.getOrDefault(beam.text, BeamState(beam.text))
                        nextBeams[beam.text] = stayEntry.copy(
                            probNonBlank = stayEntry.probNonBlank + beam.probNonBlank * pToken
                        )
                    } else {
                        // Different character or doesn't end in same: can merge
                        val newEntry = nextBeams.getOrDefault(newText, BeamState(newText))
                        nextBeams[newText] = newEntry.copy(
                            probNonBlank = newEntry.probNonBlank + beam.totalProb * pToken
                        )
                    }
                }
            }

            // Pruning: Keep top K beams
            beams = nextBeams.values
                .sortedByDescending { it.score }
                .take(beamWidth)
        }

        return beams.firstOrNull()?.text ?: ""
    }
}
