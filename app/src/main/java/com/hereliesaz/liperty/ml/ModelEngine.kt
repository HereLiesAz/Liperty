package com.hereliesaz.liperty.ml

import java.nio.ByteBuffer

interface ModelEngine {
    fun initialize()
    fun run(inputBuffer: ByteBuffer, outputBuffer: ByteBuffer)
    fun getOutputShape(outputIndex: Int): IntArray
    fun close()
}
