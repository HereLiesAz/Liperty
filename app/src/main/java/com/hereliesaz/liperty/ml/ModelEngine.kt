package com.hereliesaz.liperty.ml

import java.nio.ByteBuffer

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
    fun close()
}
