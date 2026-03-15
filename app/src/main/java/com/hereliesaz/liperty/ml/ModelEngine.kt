package com.hereliesaz.liperty.ml

import java.nio.ByteBuffer

interface ModelEngine {
    fun initialize(): Boolean
    fun run(inputBuffer: ByteBuffer, outputBuffer: ByteBuffer)
    fun getOutputShape(outputIndex: Int): IntArray
    fun close()
}
