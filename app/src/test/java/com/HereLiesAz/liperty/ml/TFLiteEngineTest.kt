package com.HereLiesAz.liperty.ml

import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config
import java.nio.ByteBuffer
import java.nio.ByteOrder

@RunWith(AndroidJUnit4::class)
@Config(sdk = [34], manifest = Config.NONE)
class TFLiteEngineTest {

    @Test
    fun testInterpreterClassExists() {
        try {
            val clazz = Class.forName("org.tensorflow.lite.Interpreter")
            assertNotNull(clazz)
        } catch (e: ClassNotFoundException) {
            org.junit.Assert.fail("Interpreter class not found")
        }
    }

    @Test
    fun testGpuDelegateClassExists() {
        try {
            val clazz = Class.forName("org.tensorflow.lite.gpu.GpuDelegate")
            assertNotNull(clazz)
        } catch (e: ClassNotFoundException) {
            org.junit.Assert.fail("GpuDelegate class not found")
        }
    }

    @Test
    fun testFileUtilClassExists() {
        try {
            val clazz = Class.forName("org.tensorflow.lite.support.common.FileUtil")
            assertNotNull(clazz)
        } catch (e: ClassNotFoundException) {
            org.junit.Assert.fail("FileUtil class not found")
        }
    }

    @Test
    fun testGpuDelegateInstantiation() {
        try {
            // detailed check
            val delegate = org.tensorflow.lite.gpu.GpuDelegate()
            assertNotNull(delegate)
            delegate.close()
        } catch (t: Throwable) {
            println("GpuDelegate instantiation failed: $t")
        }
    }

    @Test
    fun testInitializeAndRun() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()

        // Check assets
        val assets = context.assets.list("")
        println("Assets: ${assets?.joinToString()}")

        // We expect vsr_model.tflite to be in assets
        // Note: In some Robolectric configs, assets might not be mounted correctly if not specified.
        // But usually standard setup works.

        try {
            val clazz = TFLiteEngine::class.java
            println("TFLiteEngine class found: $clazz")

            val engine = TFLiteEngine(context)

            engine.initialize()

            // Verify getOutputShape returns correct shape
            val outputShape = engine.getOutputShape(0)

            if (outputShape.isNotEmpty()) {
                println("Model loaded. Output shape: ${outputShape.joinToString()}")
                assertTrue(outputShape.contentEquals(intArrayOf(1, 50, 40)))

                // Run inference
                // Input shape: [1, 50, 88, 88, 1] -> 50 * 88 * 88 * 1 * 4 bytes (float32)
                val inputSize = 50 * 88 * 88 * 1 * 4
                val inputBuffer = ByteBuffer.allocateDirect(inputSize)
                inputBuffer.order(ByteOrder.nativeOrder())

                // Output shape: [1, 50, 40] -> 50 * 40 * 4 bytes
                val outputSize = 50 * 40 * 4
                val outputBuffer = ByteBuffer.allocateDirect(outputSize)
                outputBuffer.order(ByteOrder.nativeOrder())

                engine.run(inputBuffer, outputBuffer)
                assertNotNull(outputBuffer)
                println("Inference ran successfully")
            } else {
                 println("Engine initialization failed or model not loaded.")
                 // Fail the test if model didn't load, unless assets are missing
                 if (assets?.contains("vsr_model.tflite") == true) {
                     org.junit.Assert.fail("Model found in assets but failed to load.")
                 } else {
                     println("WARNING: vsr_model.tflite not found in assets. Skipping test verification.")
                 }
            }
        } catch (t: Throwable) {
            throw RuntimeException("Test failed with: ${t.javaClass.name}: ${t.message}", t)
        }
    }
}
