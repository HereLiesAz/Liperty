package com.HereLiesAz.Liperty.ml

class GreedyDecoder {

    // Simple vocabulary for testing/prototype.
    // In production, this should match the specific model's vocabulary (e.g., LRS3).
    // Including ' ' as space and assuming index 0 might be blank or padding.
    // Let's assume standard CTC: [blank, A, B, ..., Z, space] or similar.
    // For this example: 0=blank, 1=A, ..., 26=Z, 27=space.
    private val vocab = listOf(
        "_", // Blank
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        " "
    )

    /**
     * Decodes a sequence of probabilities (T x V) into a string.
     * T = time steps, V = vocabulary size.
     */
    fun decode(probabilities: Array<FloatArray>): String {
        val result = StringBuilder()
        var prevIndex = -1

        for (timeStep in probabilities) {
            // Find index of max probability
            var maxIndex = -1
            var maxProb = -1.0f

            for (i in timeStep.indices) {
                if (timeStep[i] > maxProb) {
                    maxProb = timeStep[i]
                    maxIndex = i
                }
            }

            // CTC Greedy Decoding Logic:
            // 1. If maxIndex is not blank (0) AND
            // 2. If maxIndex is different from previous index (to handle repeated characters emitted over multiple frames)
            // Then append.

            // Note: If we want to support actual repeated letters (like "HELLO" -> L, L),
            // CTC requires a blank in between.
            // e.g. H, H, E, E, L, L, _, L, L, O -> HEL LO (wait, usually H, E, L, _, L, O)
            // If the sequence is just indexes: 8 (H), 8 (H), 5 (E), 12 (L), 12 (L), 0 (Blank), 12 (L), 15 (O)
            // Collapsing duplicates: H, E, L, Blank, L, O
            // Removing blanks: H, E, L, L, O

            if (maxIndex != prevIndex) {
                if (maxIndex != 0 && maxIndex < vocab.size) {
                    result.append(vocab[maxIndex])
                }
                prevIndex = maxIndex
            }
        }

        return result.toString()
    }

    // Allow custom vocab injection
    fun decode(probabilities: Array<FloatArray>, vocabulary: List<String>): String {
        val result = StringBuilder()
        var prevIndex = -1

        for (timeStep in probabilities) {
            var maxIndex = -1
            var maxProb = -1.0f

            for (i in timeStep.indices) {
                if (timeStep[i] > maxProb) {
                    maxProb = timeStep[i]
                    maxIndex = i
                }
            }

            if (maxIndex != prevIndex) {
                if (maxIndex != 0 && maxIndex < vocabulary.size) {
                    result.append(vocabulary[maxIndex])
                }
                prevIndex = maxIndex
            }
        }
        return result.toString()
    }
}
