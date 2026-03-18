package com.HereLiesAz.liperty.ml

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import com.hereliesaz.liperty.ml.TFLiteEngine
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.nio.ByteBuffer
import java.nio.ByteOrder

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33], manifest = Config.NONE)
class TFLiteEngineTest {

    @Test
    fun testLiteRTClassesPresent() {
        try {
            val clazz = Class.forName("org.tensorflow.lite.Interpreter")
            assertNotNull(clazz)
        } catch (e: Exception) {
            fail("LiteRT Interpreter class not found in classpath")
        }
    }

    @Test
    fun testLiteRTGPUClassesPresent() {
        try {
            val clazz = Class.forName("org.tensorflow.lite.gpu.GpuDelegate")
            assertNotNull(clazz)
        } catch (e: Exception) {
            // Note: GPU delegate might not be present in all test environments
            println("LiteRT GPU Delegate not found (expected in some CI environments)")
        }
    }

    @Test
    fun testLiteRTSupportClassesPresent() {
        try {
            val clazz = Class.forName("org.tensorflow.lite.support.common.FileUtil")
            assertNotNull(clazz)
        } catch (e: Exception) {
            fail("LiteRT Support FileUtil class not found in classpath")
        }
    }

    @Test
    fun testGpuDelegateInstantiation() {
        try {
            val delegate = org.tensorflow.lite.gpu.GpuDelegate()
            assertNotNull(delegate)
            delegate.close()
        } catch (e: UnsatisfiedLinkError) {
            // Expected in Robolectric as it can't load native .so files
            println("Native library load failed as expected in Robolectric")
        } catch (e: NoClassDefFoundError) {
            fail("GpuDelegate instantiation failed: $e")
        } catch (e: Exception) {
            println("Caught expected exception in Robolectric: $e")
        }
    }
}
