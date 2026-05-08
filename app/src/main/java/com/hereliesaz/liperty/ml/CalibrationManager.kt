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
    private val g2pConverter = G2PConverter()

    // Phrases for calibration (phonetically diverse)
    val calibrationPhrases = listOf(
        "The quick brown fox jumps over the lazy dog",
        "Pack my box with five dozen liquor jugs",
        "How vexingly quick daft zebras jump",
        "Sphinx of black quartz judge my vow"
    )

    private var currentPhraseIndex = 0
    private val collectedFrames = mutableListOf<Bitmap>()
    private var hasStarted = false
    
    // Constants matching vsr_lora_model.tflite (VALLR Production)
    private val INPUT_WIDTH = 224
    private val INPUT_HEIGHT = 224
    private val NUM_FRAMES = 16
    private val NUM_CHANNELS = 3
    private val VOCAB_SIZE = 40 // Matching MLConstants.PHONEME_VOCAB (0-39); label buffer shape: [1, NUM_FRAMES, VOCAB_SIZE]

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
        hasStarted = true
        if (collectedFrames.size < NUM_FRAMES) {
            Log.w("CalibrationManager", "Insufficient frames for training: ${collectedFrames.size}")
            return@withContext -1f
        }

        // 1. Prepare Input Buffer [1, 16, 224, 224, 3]
        val inputBuffer = ByteBuffer.allocateDirect(1 * NUM_FRAMES * INPUT_HEIGHT * INPUT_WIDTH * NUM_CHANNELS * 4)
        inputBuffer.order(ByteOrder.nativeOrder())
        
        for (bitmap in collectedFrames) {
            val pixels = IntArray(INPUT_WIDTH * INPUT_HEIGHT)
            bitmap.getPixels(pixels, 0, INPUT_WIDTH, 0, 0, INPUT_WIDTH, INPUT_HEIGHT)
            for (pixel in pixels) {
                // RGB [0.0 - 1.0]
                inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f)
                inputBuffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)
                inputBuffer.putFloat((pixel and 0xFF) / 255.0f)
            }
        }
        inputBuffer.rewind()

        // 2. Prepare Label Buffer [1, NUM_FRAMES, VOCAB_SIZE]
        val labelBuffer = ByteBuffer.allocateDirect(1 * NUM_FRAMES * VOCAB_SIZE * 4)
        labelBuffer.order(ByteOrder.nativeOrder())
        
        // Vocab matching decoders
        val vocab = MLConstants.PHONEME_VOCAB

        // Use G2PConverter to translate text to phonemes
        val targetPhonemes = g2pConverter.sentenceToPhonemes(label)
        
        // Convert phonemes to their respective vocab indices
        val targetIndices = targetPhonemes.mapNotNull { phoneme ->
            val idx = vocab.indexOf(phoneme)
            if (idx >= 0) idx else null
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
        collectedFrames.forEach { BitmapPool.recycle(it) }
        collectedFrames.clear()
        
        currentPhraseIndex = (currentPhraseIndex + 1) % calibrationPhrases.size
        
        loss
    }

    fun isCalibrationComplete(): Boolean {
        // Return true only after calibration has started and all phrases have been processed
        // (currentPhraseIndex wraps back to 0 after the final phrase)
        return hasStarted && currentPhraseIndex == 0 && collectedFrames.isEmpty()
    }

    fun close() {
        trainer.close()
    }
}
