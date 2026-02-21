package com.HereLiesAz.Liperty.ml

import java.nio.ByteBuffer

interface ModelEngine {
    fun initialize()
    fun run(inputBuffer: ByteBuffer, outputBuffer: ByteBuffer)
    fun getOutputShape(outputIndex: Int): IntArray
    fun close()
}
