package com.HereLiesAz.liperty.ml

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class FaceLandmarkerHelperTest {

    @Test
    fun testCalculateHeadPose() {
        // Identity matrix (no rotation)
        // 4x4 row-major:
        // 1 0 0 0
        // 0 1 0 0
        // 0 0 1 0
        // 0 0 0 1
        val identityMatrix = floatArrayOf(
            1f, 0f, 0f, 0f,
            0f, 1f, 0f, 0f,
            0f, 0f, 1f, 0f,
            0f, 0f, 0f, 1f
        )

        val pose = FaceLandmarkerHelper.calculateHeadPose(identityMatrix)
        assertNotNull(pose)
        assertEquals(0f, pose!!.first, 0.001f) // Roll
        assertEquals(0f, pose.second, 0.001f) // Pitch
        assertEquals(0f, pose.third, 0.001f) // Yaw
    }

    // Additional test for 90 degree rotation around Z (Roll)
    // Rz(90) =
    // 0 -1 0
    // 1  0 0
    // 0  0 1
    // Matrix in 4x4 row-major:
    // 0 -1 0 0
    // 1  0 0 0
    // 0  0 1 0
    // 0  0 0 1
    @Test
    fun testCalculateHeadPoseRoll90() {
        val matrix = floatArrayOf(
            0f, -1f, 0f, 0f,
            1f, 0f, 0f, 0f,
            0f, 0f, 1f, 0f,
            0f, 0f, 0f, 1f
        )

        val pose = FaceLandmarkerHelper.calculateHeadPose(matrix)
        assertNotNull(pose)

        // Pitch = atan2(r32, r33) = atan2(0, 1) = 0
        assertEquals(0f, pose!!.second, 0.001f)

        // Yaw = atan2(-r31, sqrt...) = atan2(0, 1) = 0
        assertEquals(0f, pose.third, 0.001f)

        // Roll = atan2(r21, r11) = atan2(1, 0) = PI/2
        assertEquals(Math.PI.toFloat() / 2, pose.first, 0.001f)
    }
}
