package com.hereliesaz.liperty.utils

import android.graphics.Bitmap
import android.graphics.Rect
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video
import kotlin.math.max
import kotlin.math.min

/**
 * Calculates inter-frame movement using Lucas-Kanade Optical Flow
 * to maintain mouth centering and reduce jitter.
 */
class OpticalFlowTracker {
    private var prevFrameGray: Mat? = null
    private var prevPoints: MatOfPoint2f? = null

    // Accumulate the offset over frames to smooth out high-frequency jitter
    private var smoothedDx = 0f
    private var smoothedDy = 0f

    /**
     * Calculates the offset to stabilize the bounding box.
     *
     * @param bitmap The current full camera frame.
     * @param lipBox The current bounding box for the lips.
     * @return The updated Rect with optical flow stabilization applied.
     */
    fun stabilizeBox(bitmap: Bitmap, lipBox: Rect): Rect {
        val currFrameMat = Mat()
        val currFrameGray = Mat()
        val corners = MatOfPoint()

        try {
            Utils.bitmapToMat(bitmap, currFrameMat)

            Imgproc.cvtColor(currFrameMat, currFrameGray, Imgproc.COLOR_RGBA2GRAY)

            var dx = 0f
            var dy = 0f

            if (prevFrameGray != null && prevPoints != null && !prevPoints!!.empty()) {
                val nextPoints = MatOfPoint2f()
                val status = MatOfByte()
                val err = MatOfFloat()

                Video.calcOpticalFlowPyrLK(prevFrameGray, currFrameGray, prevPoints, nextPoints, status, err)

                val prevPtsList = prevPoints!!.toList()
                val nextPtsList = nextPoints.toList()
                val statusList = status.toList()

                var validPoints = 0
                var sumDx = 0f
                var sumDy = 0f

                for (i in statusList.indices) {
                    if (statusList[i].toInt() == 1) {
                        val p1 = prevPtsList[i]
                        val p2 = nextPtsList[i]

                        // Calculate how much the features moved
                        sumDx += (p2.x - p1.x).toFloat()
                        sumDy += (p2.y - p1.y).toFloat()
                        validPoints++
                    }
                }

                if (validPoints > 0) {
                    dx = sumDx / validPoints
                    dy = sumDy / validPoints
                }

                nextPoints.release()
                status.release()
                err.release()
            }

            // Exponential moving average for smoothing the flow vectors
            val alpha = 0.7f
            smoothedDx = alpha * smoothedDx + (1 - alpha) * dx
            smoothedDy = alpha * smoothedDy + (1 - alpha) * dy

            // Find new features to track for the next frame

            // Use a mask to only track features near the lipBox to keep it centered on the mouth's actual movement
            val maskMat = Mat.zeros(currFrameGray.size(), org.opencv.core.CvType.CV_8UC1)

            // Define ROI based on the lip box, expanding it slightly to get good features around the mouth
            val expand = 40
            val expandedBox = Rect(
                max(0, lipBox.left - expand),
                max(0, lipBox.top - expand),
                min(bitmap.width, lipBox.right + expand),
                min(bitmap.height, lipBox.bottom + expand)
            )

            if (expandedBox.width() > 0 && expandedBox.height() > 0) {
                val cvRect = org.opencv.core.Rect(expandedBox.left, expandedBox.top, expandedBox.width(), expandedBox.height())
                val roi = maskMat.submat(cvRect)
                roi.setTo(org.opencv.core.Scalar(255.0))
                roi.release()
            } else {
                maskMat.release()
                return lipBox
            }

            Imgproc.goodFeaturesToTrack(currFrameGray, corners, 50, 0.01, 10.0, maskMat, 3, false, 0.04)

            if (!corners.empty()) {
                val corners2f = MatOfPoint2f(*corners.toArray())
                if (prevPoints != null) {
                    prevPoints!!.release()
                }
                prevPoints = corners2f
            }

            maskMat.release()

            if (prevFrameGray != null) {
                prevFrameGray!!.release()
            }
            prevFrameGray = currFrameGray

            // Apply the smoothed offset to the original lipBox
            // If the face moves right (dx > 0), we want the box to follow it right, so we add dx.
            val newLeft = (lipBox.left + smoothedDx).toInt()
            val newTop = (lipBox.top + smoothedDy).toInt()
            val newRight = (lipBox.right + smoothedDx).toInt()
            val newBottom = (lipBox.bottom + smoothedDy).toInt()

            return Rect(newLeft, newTop, newRight, newBottom)
        } finally {
            currFrameMat.release()
            // currFrameGray is either owned by prevFrameGray or needs release on exception path;
            // only release it here if it hasn't been handed off to prevFrameGray.
            if (currFrameGray !== prevFrameGray) {
                currFrameGray.release()
            }
            corners.release()
            // maskMat is released inside the try block after use.
        }
    }

    fun reset() {
        if (prevFrameGray != null) {
            prevFrameGray!!.release()
            prevFrameGray = null
        }
        if (prevPoints != null) {
            prevPoints!!.release()
            prevPoints = null
        }
        smoothedDx = 0f
        smoothedDy = 0f
    }
}
