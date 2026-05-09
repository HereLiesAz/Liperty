package com.hereliesaz.liperty.ml

import java.nio.ByteBuffer

enum class InputLayout {
    /** Batch, Time, Height, Width, Channels — TFLite/Keras convention. */
    NTHWC,
    /** Batch, Channels, Time, Height, Width — PyTorch/HuggingFace convention
     *  for video classifiers like VideoMAE. */
    NCTHW,
    /** Batch, Time, Channels, Height, Width — PyTorch convention for
     *  per-frame image classifiers stacked over time. Used by Auto-AVSR /
     *  VSR-ML's visual encoder, where each frame is `(C, H, W)` and the
     *  whole clip is `(T, C, H, W)`. */
    NTCHW
}

/** Resolved logical dimensions for a 5-D model input. */
data class InputDimensions(
    val numFrames: Int,
    val inputHeight: Int,
    val inputWidth: Int,
    val numChannels: Int,
)

/**
 * Map a 5-D input shape to logical (frames, H, W, channels) per the engine's
 * declared layout. Returns null for shorter shapes so callers can keep their
 * existing defaults (e.g. when an engine isn't yet loaded and returns a stub
 * shape).
 */
fun parseInputShape(shape: IntArray, layout: InputLayout): InputDimensions? {
    if (shape.size < 5) return null
    return when (layout) {
        InputLayout.NTHWC -> InputDimensions(
            numFrames = shape[1], inputHeight = shape[2],
            inputWidth = shape[3], numChannels = shape[4],
        )
        InputLayout.NCTHW -> InputDimensions(
            numChannels = shape[1], numFrames = shape[2],
            inputHeight = shape[3], inputWidth = shape[4],
        )
        InputLayout.NTCHW -> InputDimensions(
            numFrames = shape[1], numChannels = shape[2],
            inputHeight = shape[3], inputWidth = shape[4],
        )
    }
}

interface ModelEngine {
    fun initialize(): Boolean
    fun run(inputBuffer: ByteBuffer, outputBuffer: ByteBuffer)
    // Optional: Overload for multi-input models (e.g., dual-input LipCoordNet)
    fun run(inputBuffers: Array<Any>, outputBuffer: ByteBuffer) {
        // By default, fallback to single input if array has 1 element
        if (inputBuffers.size == 1 && inputBuffers[0] is ByteBuffer) {
            run(inputBuffers[0] as ByteBuffer, outputBuffer)
        } else {
            throw UnsupportedOperationException("Multi-input run() not implemented for this engine.")
        }
    }
    fun getOutputShape(outputIndex: Int): IntArray
    fun getInputShape(inputIndex: Int): IntArray

    /**
     * The layout the engine expects for video input on input 0. Callers must
     * write the input ByteBuffer in this order. Default is NTHWC for back-compat
     * with the existing TFLite path.
     */
    fun getInputLayout(): InputLayout = InputLayout.NTHWC

    fun close()
}
