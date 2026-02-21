package com.HereLiesAz.Liperty.camera

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class CameraManagerTest {

    @Test
    fun testCameraManagerInstantiation() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        // Ensure CameraManager can be instantiated without throwing exception
        val cameraManager = CameraManager(context)
    }
}
