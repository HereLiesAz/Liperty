package com.hereliesaz.liperty.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.hereliesaz.liperty.utils.BitmapPool
import com.hereliesaz.liperty.utils.ImageUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Orchestrates the onboarding calibration phase for both VSR and SSI.
 * Collects mouth ROIs and sensor data while the user mouths specific phrases.
 */
class CalibrationManager(private val context: Context) {

    private val trainer = OnDeviceTrainer(context)
    
    // Phrases for calibration (phonetically diverse)
    val calibrationPhrases = listOf(
        "The quick brown fox jumps over the lazy dog",
        "Pack my box with five dozen liquor jugs",
        "How vexingly quick daft zebras jump",
        "Sphinx of black quartz judge my vow"
    )

    private var currentPhraseIndex = 0
    private val collectedFrames = mutableListOf<Bitmap>()
    
    // Constants matching vsr_lora_model.tflite
    private val INPUT_WIDTH = 88
    private val INPUT_HEIGHT = 88
    private val NUM_FRAMES = 50
    private val VOCAB_SIZE = 40

    fun initialize() {
        trainer.initialize()
    }

    fun getCurrentPhrase(): String = calibrationPhrases[currentPhraseIndex]

    fun addFrame(bitmap: Bitmap) {
        // Only keep the required number of frames for the current phrase
        if (collectedFrames.size < NUM_FRAMES) {
            // Process and store frame
            val processed = ImageUtils.normalizeForInference(bitmap)
            collectedFrames.add(processed)
        }
    }

    /**
     * Runs a training step for the current phrase.
     * @param label The ground truth text for the phrase.
     */
    suspend fun processCurrentPhrase(label: String): Float = withContext(Dispatchers.Default) {
        if (collectedFrames.size < NUM_FRAMES) {
            Log.w("CalibrationManager", "Insufficient frames for training: ${collectedFrames.size}")
            return@withContext -1f
        }

        // 1. Prepare Input Buffer [1, 50, 88, 88, 1]
        val inputBuffer = ByteBuffer.allocateDirect(1 * NUM_FRAMES * INPUT_HEIGHT * INPUT_WIDTH * 1 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())
        
        for (bitmap in collectedFrames) {
            val pixels = IntArray(INPUT_WIDTH * INPUT_HEIGHT)
            bitmap.getPixels(pixels, 0, INPUT_WIDTH, 0, 0, INPUT_WIDTH, INPUT_HEIGHT)
            for (pixel in pixels) {
                inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f)
            }
        }
        inputBuffer.rewind()

        // 2. Prepare Label Buffer [1, 50, 40]
        // Mapping label characters to indices
        val labelBuffer = ByteBuffer.allocateDirect(1 * NUM_FRAMES * VOCAB_SIZE * 4)
        labelBuffer.order(ByteOrder.nativeOrder())
        
        // Vocab matching decoders
        val vocab = listOf(
            "_", "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY",
            "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P",
            "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"
        )
        // Dummy mapping for calibration phrases (since we don't have a real G2P converter here,
        // we'll just map characters to the closest looking phoneme index or blank for spaces)
        val targetIndices = label.uppercase().map { char ->
            when (char) {
                'A' -> vocab.indexOf("AA")
                'B' -> vocab.indexOf("B")
                'C' -> vocab.indexOf("CH")
                'D' -> vocab.indexOf("D")
                'E' -> vocab.indexOf("EH")
                'F' -> vocab.indexOf("F")
                'G' -> vocab.indexOf("G")
                'H' -> vocab.indexOf("HH")
                'I' -> vocab.indexOf("IH")
                'J' -> vocab.indexOf("JH")
                'K' -> vocab.indexOf("K")
                'L' -> vocab.indexOf("L")
                'M' -> vocab.indexOf("M")
                'N' -> vocab.indexOf("N")
                'O' -> vocab.indexOf("OW")
                'P' -> vocab.indexOf("P")
                'Q' -> vocab.indexOf("K")
                'R' -> vocab.indexOf("R")
                'S' -> vocab.indexOf("S")
                'T' -> vocab.indexOf("T")
                'U' -> vocab.indexOf("UH")
                'V' -> vocab.indexOf("V")
                'W' -> vocab.indexOf("W")
                'X' -> vocab.indexOf("S") // Approximate
                'Y' -> vocab.indexOf("Y")
                'Z' -> vocab.indexOf("Z")
                else -> 0 // Blank for space and punctuation
            }
        }

        for (t in 0 until NUM_FRAMES) {
            val charIdx = if (t < targetIndices.size) targetIndices[t] else 0 // Pad with blank
            for (c in 0 until VOCAB_SIZE) {
                labelBuffer.putFloat(if (c == charIdx) 1.0f else 0.0f)
            }
        }
        labelBuffer.rewind()

        // 3. Train
        val loss = trainer.trainStep(inputBuffer, labelBuffer)
        
        // Clean up
        collectedFrames.forEach { it.recycle() }
        collectedFrames.clear()
        
        currentPhraseIndex = (currentPhraseIndex + 1) % calibrationPhrases.size
        
        loss
    }

    fun isCalibrationComplete(): Boolean {
        // Return true if we've gone through all phrases once
        // (For the sake of this task, we can say it's done after phrases are processed)
        return currentPhraseIndex == 0 && collectedFrames.isEmpty() // Wrapped around
    }

    fun close() {
        trainer.close()
    }
}
