package com.hereliesaz.liperty.ml

object MLConstants {
    val PHONEME_VOCAB = listOf(
        "_", "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY",
        "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P",
        "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"
    )

    // MediaPipe face mesh lip contour index groups.
    val OUTER_UPPER_LIP = intArrayOf(61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291)
    val OUTER_LOWER_LIP = intArrayOf(291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61)
    val INNER_UPPER_LIP = intArrayOf(78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308)
    val INNER_LOWER_LIP = intArrayOf(308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78)
    
    // 20 distinct points for inner lip coord tracking
    val INNER_LIP_INDICES = (INNER_UPPER_LIP.toList() + INNER_LOWER_LIP.toList()).distinct().toIntArray()
}
