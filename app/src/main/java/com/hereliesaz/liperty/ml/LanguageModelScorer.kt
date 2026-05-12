package com.hereliesaz.liperty.ml

/**
 * Scores word sequences under an external language model. The contract is
 * deliberately small so decoders can integrate an LM without coupling to
 * any particular implementation (n-gram via KenLM, Transformer LM via
 * ONNX, the [LlmTextCleaner]'s Gemma — all valid choices).
 *
 * Implementations must return a log10 probability (or any monotonic
 * log-likelihood) — `SubwordCtcBeamDecoder` adds `lmWeight * score` to
 * each beam's log-likelihood, so absolute magnitude is unitless as long
 * as the *relative* ordering of candidates is preserved.
 *
 * `score(emptyList())` must return a finite value (typically 0) so the
 * decoder can score empty hypotheses without special-casing.
 */
interface LanguageModelScorer {
    fun score(words: List<String>): Float
    fun close()
}

/** No-op scorer — returns 0 for every input. Wire this in when the LM
 *  artifact is missing so the decoder still works; the rescoring step
 *  becomes a no-op and the result is the pure CTC argmax-by-beam path. */
class NoopLanguageModelScorer : LanguageModelScorer {
    override fun score(words: List<String>): Float = 0f
    override fun close() {}
}
