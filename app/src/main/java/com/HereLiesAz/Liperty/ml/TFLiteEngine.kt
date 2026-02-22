package com.HereLiesAz.Liperty.ml

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer

class TFLiteEngine(private val context: Context) : ModelEngine {

    private var interpreter: Interpreter? = null
    // Changed from GpuDelegate? to Delegate? to prevent class loading issues in Robolectric
    // where native libs for GPU delegate might be missing.
    private var gpuDelegate: Delegate? = null
    private val MODEL_NAME = "vsr_model.tflite"

    override fun initialize() {
        if (interpreter != null) return

        try {
            val options = Interpreter.Options()
            try {
                // Try to initialize GPU delegate. This might fail if native libs are missing.
                gpuDelegate = GpuDelegate()
                options.addDelegate(gpuDelegate)
            } catch (e: NoClassDefFoundError) {
                Log.e("TFLiteEngine", "GPU Delegate class not found, falling back to CPU", e)
            } catch (e: UnsatisfiedLinkError) {
                Log.e("TFLiteEngine", "GPU Delegate native lib not found, falling back to CPU", e)
            } catch (e: Exception) {
                Log.e("TFLiteEngine", "GPU Delegate not supported, falling back to CPU", e)
            }

            val modelFile = FileUtil.loadMappedFile(context, MODEL_NAME)
            interpreter = Interpreter(modelFile, options)
            Log.i("TFLiteEngine", "TFLite Model loaded successfully")

        } catch (e: java.io.FileNotFoundException) {
            Log.e("TFLiteEngine", "Model file not found: $MODEL_NAME")
        } catch (e: Exception) {
            Log.e("TFLiteEngine", "Error initializing TFLite", e)
        }
    }

    override fun run(inputBuffer: ByteBuffer, outputBuffer: ByteBuffer) {
        interpreter?.run(inputBuffer, outputBuffer)
    }

    override fun getOutputShape(outputIndex: Int): IntArray {
        return interpreter?.getOutputTensor(outputIndex)?.shape() ?: IntArray(0)
    }

    override fun close() {
        interpreter?.close()
        gpuDelegate?.close()
    }
}
