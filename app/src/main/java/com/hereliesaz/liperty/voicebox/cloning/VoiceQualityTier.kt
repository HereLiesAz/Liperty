package com.hereliesaz.liperty.voicebox.cloning

/**
 * Discrete quality bands for a banked voice profile, derived from
 * the number of reference clips captured and their cumulative
 * speech duration.
 *
 * Used by the voice-coach UI to show users live progression as
 * they record more clips — the "more samples = more like you"
 * intuition made concrete. Each band corresponds to roughly the
 * point at which OpenVoice v2's tone-color converter stops
 * benefiting meaningfully from additional samples on the embedding
 * side alone (further gains require Tier-3 fine-tuning).
 *
 * Thresholds were chosen so that:
 * - [INITIAL] is reachable in one ~5 s clip (any user can hit it)
 * - [BASIC] requires about a minute of speech, ~3 clips — enough
 *   that the centroid is no longer dominated by one noisy estimate
 * - [GOOD] requires the bulk of a normal banking session
 * - [EXCELLENT] is the asymptote — past this, marginal returns on
 *   pure averaging fall below what a Tier-3 LoRA could deliver
 */
enum class VoiceQualityTier(
    /** Sequential rank used to compare tiers and render dots. */
    val rank: Int,
    /** Filled-dot count for the four-dot UI indicator. */
    val filledDots: Int,
    /** Minimum number of reference clips required to reach this tier. */
    val minClips: Int,
    /** Minimum cumulative speech (seconds) required to reach this tier. */
    val minSeconds: Int,
) {
    /** Nothing captured yet. */
    NONE(rank = 0, filledDots = 0, minClips = 0, minSeconds = 0),

    /** First clip captured — voice is recognizable but unstable. */
    INITIAL(rank = 1, filledDots = 1, minClips = 1, minSeconds = 5),

    /** ~3 clips, ~30 s — embedding centroid stabilizing. */
    BASIC(rank = 2, filledDots = 2, minClips = 3, minSeconds = 30),

    /** ~6 clips, ~90 s — full quality on pure averaging. */
    GOOD(rank = 3, filledDots = 3, minClips = 6, minSeconds = 90),

    /** ~12 clips, ~4 min — centroid stable; further gain needs Tier-3. */
    EXCELLENT(rank = 4, filledDots = 4, minClips = 12, minSeconds = 240);

    companion object {
        /**
         * Computes the highest tier a profile with [clipCount] clips
         * and [totalSeconds] cumulative speech qualifies for.
         *
         * Both thresholds must be met: a single 60-second clip is
         * still [INITIAL] because the centroid is dominated by one
         * estimate (the "more clips → more centroid stability"
         * argument requires multiple clips, not just more audio).
         */
        fun compute(clipCount: Int, totalSeconds: Int): VoiceQualityTier {
            val candidates = entries.filter { tier ->
                clipCount >= tier.minClips && totalSeconds >= tier.minSeconds
            }
            return candidates.maxByOrNull { it.rank } ?: NONE
        }
    }
}
