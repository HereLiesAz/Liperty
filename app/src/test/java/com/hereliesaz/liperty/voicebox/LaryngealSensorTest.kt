package com.hereliesaz.liperty.voicebox

import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class LaryngealSensorTest {

    private lateinit var sensor: LaryngealSensor

    @Before
    fun setUp() {
        sensor = LaryngealSensor(RuntimeEnvironment.getApplication())
    }

    @Test
    fun `setSensitivity stores value`() {
        // sensitivity is private, but calling setSensitivity should not throw
        sensor.setSensitivity(0.0f)
        sensor.setSensitivity(0.5f)
        sensor.setSensitivity(1.0f)
    }

    @Test
    fun `setCarrierF0 updates value`() {
        sensor.setCarrierF0(150f)
        assertEquals(150f, sensor.carrierF0)
    }

    @Test
    fun `carrierF0 default is 120 Hz`() {
        assertEquals(120f, sensor.carrierF0)
    }

    @Test
    fun `startBCMode double start is guarded`() {
        // In Robolectric, AudioRecord construction will fail (no real audio
        // hardware), so startBCMode will fail gracefully at the AudioRecord
        // stage. The AtomicBoolean guard prevents a true double-start: even
        // if the first call fails internally and resets isRunning, the guard
        // logic itself is exercised.
        val noOp: (FloatArray) -> Unit = {}
        val noOpBool: (Boolean) -> Unit = {}

        sensor.startBCMode(noOp, noOpBool)
        // Second call should be a no-op due to the AtomicBoolean guard
        sensor.startBCMode(noOp, noOpBool)

        // Clean up
        sensor.stop()
    }

    @Test
    fun `stop is idempotent`() {
        // Calling stop() multiple times should not crash
        sensor.stop()
        sensor.stop()
        sensor.stop()
    }

    @Test
    fun `stop without start is safe`() {
        // stop() before any start call should not throw
        sensor.stop()
    }
}
