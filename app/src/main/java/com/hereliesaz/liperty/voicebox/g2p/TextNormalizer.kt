package com.hereliesaz.liperty.voicebox.g2p

/**
 * Normalizes English text before grapheme-to-phoneme conversion.
 *
 * Pure Kotlin object — no Android dependencies, fully testable on JVM.
 *
 * Processing order:
 * 1. Expand currency, percent, and common symbol forms.
 * 2. Expand common abbreviations (title words, etc.).
 * 3. Expand cardinal numbers (integers written as digits).
 * 4. Lowercase and strip non-ASCII punctuation, keeping [.,!?;:-] and spaces.
 * 5. Collapse consecutive whitespace.
 */
object TextNormalizer {

    // Abbreviations expanded before lowercasing so matching is case-insensitive.
    private val ABBREVIATIONS = listOf(
        Regex("\\bMr\\.\\s*", RegexOption.IGNORE_CASE) to "mister ",
        Regex("\\bMrs\\.\\s*", RegexOption.IGNORE_CASE) to "misses ",
        Regex("\\bDr\\.\\s*", RegexOption.IGNORE_CASE) to "doctor ",
        Regex("\\bSt\\.\\s*", RegexOption.IGNORE_CASE) to "saint ",
        Regex("\\bJr\\.\\s*", RegexOption.IGNORE_CASE) to "junior ",
        Regex("\\bSr\\.\\s*", RegexOption.IGNORE_CASE) to "senior ",
        Regex("\\bProf\\.\\s*", RegexOption.IGNORE_CASE) to "professor ",
        Regex("\\bvs\\.\\s*", RegexOption.IGNORE_CASE) to "versus "
    )

    // Currency symbols mapped to unit words (singular; plural added by number logic).
    private val CURRENCY = listOf(
        Regex("\\$([0-9]+)") to { m: MatchResult -> "${numberToWords(m.groupValues[1].toLong())} dollars" },
        Regex("£([0-9]+)") to { m: MatchResult -> "${numberToWords(m.groupValues[1].toLong())} pounds" },
        Regex("€([0-9]+)") to { m: MatchResult -> "${numberToWords(m.groupValues[1].toLong())} euros" }
    )

    /**
     * Main normalization entry point.
     *
     * Returns a lowercased string with numbers and symbols expanded.
     * Punctuation characters [.,!?;:-] are retained (CMU dict handles them
     * at the tokenizer boundary). All other non-ASCII or non-alphanumeric
     * characters are removed. Runs of whitespace are collapsed to a single space.
     */
    fun normalize(text: String): String {
        if (text.isBlank()) return ""

        var s = text

        // 1. Currency (must come before bare number expansion).
        for ((pattern, expander) in CURRENCY) {
            s = pattern.replace(s) { expander(it) }
        }

        // 2. Percent.
        s = Regex("([0-9]+)%").replace(s) { m ->
            "${numberToWords(m.groupValues[1].toLong())} percent"
        }

        // 3. Ampersand and other single-character symbols.
        s = s.replace("&", " and ")

        // 4. Abbreviations.
        for ((pattern, replacement) in ABBREVIATIONS) {
            s = pattern.replace(s, replacement)
        }

        // 5. Bare integers (after currency/percent so those are already expanded).
        s = Regex("\\b([0-9]+)\\b").replace(s) { m ->
            numberToWords(m.groupValues[1].toLong())
        }

        // 6. Lowercase.
        s = s.lowercase()

        // 7. Strip anything that is not alphanumeric, an apostrophe, or kept punctuation.
        //    Apostrophes are kept so contractions stay intact for the CMU dict.
        s = s.replace(Regex("[^a-z0-9 .,!?;:' -]"), "")

        // 8. Collapse whitespace.
        s = s.trim().replace(Regex("\\s+"), " ")

        return s
    }

    /**
     * Converts a non-negative cardinal integer (0 to 999,999,999) to English words.
     *
     * Values outside that range return the numeric string unchanged.
     */
    fun numberToWords(n: Long): String {
        if (n < 0 || n > 999_999_999L) return n.toString()
        if (n == 0L) return "zero"

        val ones = listOf(
            "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
            "seventeen", "eighteen", "nineteen"
        )
        val tens = listOf(
            "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
        )

        fun below1000(value: Long): String {
            return when {
                value == 0L -> ""
                value < 20L -> ones[value.toInt()]
                value < 100L -> {
                    val t = tens[(value / 10).toInt()]
                    val o = ones[(value % 10).toInt()]
                    if (o.isEmpty()) t else "$t-$o"
                }
                else -> {
                    val h = "${ones[(value / 100).toInt()]} hundred"
                    val remainder = value % 100
                    if (remainder == 0L) h else "$h ${below1000(remainder)}"
                }
            }
        }

        val parts = mutableListOf<String>()
        val millions = n / 1_000_000L
        val thousands = (n % 1_000_000L) / 1_000L
        val remainder = n % 1_000L

        if (millions > 0L) parts.add("${below1000(millions)} million")
        if (thousands > 0L) parts.add("${below1000(thousands)} thousand")
        if (remainder > 0L) parts.add(below1000(remainder))

        return parts.joinToString(" ")
    }
}
