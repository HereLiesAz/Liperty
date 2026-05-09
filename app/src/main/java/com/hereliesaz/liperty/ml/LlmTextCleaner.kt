package com.hereliesaz.liperty.ml

/**
 * Wraps a small on-device language model to clean up the noisy CTC output
 * from the visual-only Auto-AVSR encoder. Chaplin's actual recipe — the
 * 19.1% headline WER number requires a downstream LM step that we can
 * approximate with a 0.5B-2B-param instruction-tuned LLM running locally.
 *
 * Production wiring (not pulled into the build to avoid bloating the APK):
 *
 * ```kotlin
 * // build.gradle.kts: implementation("com.google.mediapipe:tasks-genai:0.10.20")
 * val options = LlmInferenceOptions.builder()
 *     .setModelPath(File(context.filesDir, "gemma-2b-it-cpu-int4.task").absolutePath)
 *     .setMaxTokens(128)
 *     .build()
 * val llm = LlmInference.createFromOptions(context, options)
 * val cleaner = LlmTextCleaner { prompt -> llm.generateResponse(prompt) }
 * val cleaned = cleaner.clean(rawVsrOutput)
 * ```
 *
 * The model file (~1.3 GB for Gemma-2B-it-cpu-int4) should be downloaded
 * on first run from a MediaPipe-compatible HF mirror, NOT bundled in the
 * APK. The cleaner gracefully falls back to the raw input if the model is
 * absent, throws, or returns empty — so enabling this feature is opt-in
 * via providing the model file.
 *
 * @param generate Suspend lambda that produces a model response for a prompt.
 *   In tests, pass a fake. In production, wrap your `LlmInference` instance.
 *   The lambda may throw; this class catches and falls back.
 */
class LlmTextCleaner(
    private val generate: suspend (String) -> String,
) {

    /**
     * Clean [rawText] via the LLM. Returns the cleaned text, or [rawText]
     * unchanged if the model is unavailable or produces nothing usable.
     */
    suspend fun clean(rawText: String): String {
        if (rawText.isBlank()) return ""
        val prompt = buildPrompt(rawText)
        val response = try {
            generate(prompt)
        } catch (_: Throwable) {
            // The LLM is an opt-in best-effort layer; if it errors out we
            // silently fall back to the raw VSR output. Caller can log
            // around clean() if loud errors are desired.
            return rawText
        }
        return parseResponse(response).ifBlank { rawText }
    }

    internal fun buildPrompt(rawText: String): String =
        // Few-shot-y wording, kept short to fit small-model context windows
        // and to discourage chatter. Sub-1B-param instruction models are
        // very prone to "Sure, here's the corrected text..." preambles
        // unless explicitly told not to.
        """You are a typo corrector for a silent speech recognition system.
The input below is the raw output of a visual-only lipreader and may
contain spelling errors and missing or extra letters. Output ONLY the
corrected English text, with no preamble, no quotes, no commentary,
nothing else.

Input: $rawText
Corrected:"""

    internal fun parseResponse(response: String): String {
        var s = response.trim()
        if (s.isEmpty()) return ""

        // Take the first non-empty line. Small models often add commentary
        // ("Let me know if you want me to refine this further!") on
        // subsequent lines.
        s = s.lineSequence()
            .map(String::trim)
            .firstOrNull { it.isNotEmpty() }
            ?: return ""

        // Strip common chatter prefixes the model might add to the first
        // line itself, even with the prompt asking for none.
        for (prefix in CHATTER_PREFIXES) {
            if (s.length > prefix.length && s.startsWith(prefix, ignoreCase = true)) {
                s = s.substring(prefix.length).trimStart()
            }
        }

        // Strip wrapping quotes / backticks the model sometimes adds.
        s = stripWrappingPair(s, '"')
        s = stripWrappingPair(s, '\'')
        s = stripWrappingPair(s, '`')

        return s.trim()
    }

    private fun stripWrappingPair(s: String, ch: Char): String {
        if (s.length >= 2 && s.first() == ch && s.last() == ch) {
            return s.substring(1, s.length - 1)
        }
        return s
    }

    private companion object {
        /** Listed longest-first so the longer prefixes match before shorter substrings. */
        val CHATTER_PREFIXES = listOf(
            "Sure, here's the corrected text:",
            "Sure, here is the corrected text:",
            "Here's the corrected text:",
            "Here is the corrected text:",
            "Corrected text:",
            "Corrected:",
            "Output:",
        )
    }
}
