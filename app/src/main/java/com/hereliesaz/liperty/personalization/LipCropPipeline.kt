package com.hereliesaz.liperty.personalization

import android.graphics.Bitmap
import android.graphics.Rect
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.hereliesaz.liperty.ml.FaceLandmarkerHelper
import com.hereliesaz.liperty.utils.ImageUtils
import kotlin.math.max

/**
 * Offline lip-ROI cropper that reproduces the EXACT preprocessing the live
 * camera + [com.hereliesaz.liperty.ml.VSRInference] pipeline applies, so the
 * personalization training data we persist matches the inference distribution.
 * A mismatch here silently degrades the personalized model.
 *
 * Steps (mirroring MainActivity.processFrame + VSRInference byte-write):
 *   1. lip bounding box ([FaceLandmarkerHelper.extractLipBoundingBox])
 *   2. expand to a square of side `0.55 * face_width` centered on the lip-box
 *      center (fallback `2.5 * longest lip dimension` if face box is null) —
 *      identical to MainActivity's `expandedLipBox`
 *   3. rotate to level the lips ([FaceLandmarkerHelper.calculateLipRotation])
 *   4. [ImageUtils.alignAndCropMouth] to `cropSize x cropSize`
 *   5. Rec.601 luma grayscale + `Normalize(mean, std)` — identical to
 *      VSRInference's grayscale NCTHW/NTCHW byte-write
 *
 * Output: a flat row-major `FloatArray` of length `cropSize*cropSize`, exactly
 * what [PairedTrainingStore] persists and the encoder consumes.
 *
 * The constants (88 / 0.421 / 0.165 / luma coefficients) MUST stay in sync with
 * `MainActivity.AUTOAVSR_*` and `VSRInference`. A future refactor should hoist a
 * single shared implementation that both the live and offline paths call.
 */
object LipCropPipeline {

    /**
     * Crops + normalizes the mouth ROI from [bitmap] given a pre-computed
     * [result] from [helper]. Returns null if no lips/face were found or the
     * derived crop region is degenerate.
     *
     * @param mirror horizontally flip the crop (front-camera selfie video);
     *   imported third-party video is typically not mirrored, matching LRS3.
     */
    fun cropMouthRoi(
        bitmap: Bitmap,
        result: FaceLandmarkerResult,
        helper: FaceLandmarkerHelper,
        cropSize: Int,
        pixelMean: Float,
        pixelStd: Float,
        mirror: Boolean = false,
    ): FloatArray? {
        val lipBox = helper.extractLipBoundingBox(result, bitmap.width, bitmap.height) ?: return null
        if (lipBox.width() <= 0 || lipBox.height() <= 0) return null

        val faceBox = helper.extractFaceBoundingBox(result, bitmap.width, bitmap.height, marginPct = 0f)
        val rotation = FaceLandmarkerHelper.calculateLipRotation(result)

        val cx = lipBox.centerX()
        val cy = lipBox.centerY()
        val side = if (faceBox != null) {
            (faceBox.width() * 0.55f).toInt()
        } else {
            (max(lipBox.width(), lipBox.height()) * 2.5f).toInt()
        }
        if (side <= 0) return null
        val half = side / 2
        val expanded = Rect(
            (cx - half).coerceAtLeast(0),
            (cy - half).coerceAtLeast(0),
            (cx + half).coerceAtMost(bitmap.width),
            (cy + half).coerceAtMost(bitmap.height),
        )
        if (expanded.width() <= 0 || expanded.height() <= 0) return null

        // Fresh dest per call: the offline import path is not latency-critical
        // and a shared pool would risk cross-thread reuse.
        val dest = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888)
        val cropped = ImageUtils.alignAndCropMouth(bitmap, expanded, rotation, cropSize, cropSize, dest)
        val finalBmp = if (mirror) ImageUtils.mirrorBitmap(cropped) else cropped

        val pixels = IntArray(cropSize * cropSize)
        finalBmp.getPixels(pixels, 0, cropSize, 0, 0, cropSize, cropSize)
        if (finalBmp !== dest) finalBmp.recycle()
        dest.recycle()

        val out = FloatArray(cropSize * cropSize)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF
            // Rec.601 luma — identical integer approximation to VSRInference.
            val gray = ((r * 76 + g * 150 + b * 30) ushr 8).coerceIn(0, 255)
            out[i] = (gray / 255f - pixelMean) / pixelStd
        }
        return out
    }
}
