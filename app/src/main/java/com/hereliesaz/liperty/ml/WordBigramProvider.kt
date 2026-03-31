package com.hereliesaz.liperty.ml

object WordBigramProvider {
    val map: Map<String, Map<String, Double>> = mapOf(
        "i" to mapOf("am" to 0.2, "will" to 0.1, "can" to 0.1, "have" to 0.1, "do" to 0.05, "don't" to 0.05, "was" to 0.05),
        "you" to mapOf("are" to 0.2, "will" to 0.1, "can" to 0.1, "have" to 0.1, "do" to 0.05, "don't" to 0.05, "were" to 0.05),
        "he" to mapOf("is" to 0.2, "will" to 0.1, "has" to 0.1, "was" to 0.05, "doesn't" to 0.05),
        "she" to mapOf("is" to 0.2, "will" to 0.1, "has" to 0.1, "was" to 0.05, "doesn't" to 0.05),
        "we" to mapOf("are" to 0.2, "will" to 0.1, "have" to 0.1, "can" to 0.1, "were" to 0.05),
        "they" to mapOf("are" to 0.2, "will" to 0.1, "have" to 0.1, "can" to 0.1, "were" to 0.05),
        "it" to mapOf("is" to 0.2, "was" to 0.1, "will" to 0.1, "has" to 0.1, "doesn't" to 0.05),

        "am" to mapOf("i" to 0.1, "going" to 0.1, "a" to 0.1, "the" to 0.05),
        "is" to mapOf("a" to 0.1, "the" to 0.1, "it" to 0.1, "not" to 0.1, "going" to 0.05),
        "are" to mapOf("you" to 0.1, "we" to 0.1, "they" to 0.1, "a" to 0.05, "the" to 0.05, "going" to 0.05),

        "to" to mapOf("be" to 0.1, "do" to 0.1, "go" to 0.1, "the" to 0.1, "a" to 0.05, "have" to 0.05),
        "the" to mapOf("same" to 0.05, "time" to 0.05, "first" to 0.05, "way" to 0.05, "end" to 0.05),
        "a" to mapOf("lot" to 0.05, "little" to 0.05, "good" to 0.05, "great" to 0.05, "new" to 0.05),

        "in" to mapOf("the" to 0.2, "a" to 0.1, "this" to 0.05, "that" to 0.05),
        "on" to mapOf("the" to 0.2, "a" to 0.1, "my" to 0.05, "this" to 0.05),
        "at" to mapOf("the" to 0.2, "a" to 0.1, "this" to 0.05, "that" to 0.05),
        "of" to mapOf("the" to 0.2, "a" to 0.1, "my" to 0.05, "this" to 0.05),

        "can" to mapOf("you" to 0.1, "we" to 0.1, "i" to 0.1, "be" to 0.1, "do" to 0.1),
        "will" to mapOf("be" to 0.2, "have" to 0.1, "you" to 0.1, "we" to 0.1, "i" to 0.1),
        "have" to mapOf("a" to 0.1, "the" to 0.1, "been" to 0.1, "to" to 0.1),

        "this" to mapOf("is" to 0.2, "was" to 0.1, "one" to 0.1, "time" to 0.05),
        "that" to mapOf("is" to 0.1, "was" to 0.1, "we" to 0.1, "you" to 0.1, "i" to 0.1)
    )
}
