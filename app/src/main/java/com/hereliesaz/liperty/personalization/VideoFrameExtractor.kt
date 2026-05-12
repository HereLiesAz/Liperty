package com.hereliesaz.liperty.personalization

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.util.Log

/**
 * Extracts video frames from a content URI at specific time windows,
 * returning them as bitmaps for downstream lip-ROI cropping. Used by
 * the voice cloning import path to harvest synchronized video for
 * per-user lipreading personalization.
 *
 * Implementation note: uses [MediaMetadataRetriever.getFrameAtTime]
 * which is per-frame slow but works on all Android versions and on any
 * media format the platform decoder supports. For ~30 minutes of
 * imported video at 25 fps target (~45 000 frames), this is on the
 * order of single-digit minutes of decode time — acceptable for an
 * opt-in, one-time setup flow. If we ever need real-time bulk
 * extraction we'd switch to MediaCodec + SurfaceTexture.
 *
 * Returned bitmaps are at the source resolution; downstream callers
 * (the import pipeline's lip-ROI cropper, mirroring the runtime
 * camera pipeline) align + crop them to 88×88 grayscale before
 * persistence in [PairedTrainingStore].
 */
class VideoFrameExtractor(private val context: Context) {

    /**
     * Extract one bitmap per `1000/fps` ms over the window
     * `[startMs, startMs+durationMs)`. Stops early on retriever failure.
     *
     * @return frames in temporal order, or empty list if the URI isn't a
     *   video or all frame fetches failed.
     */
    fun extractFrames(uri: Uri, startMs: Long, durationMs: Long, fps: Int = 25): List<Bitmap> {
        if (durationMs <= 0L || fps <= 0) return emptyList()
        val retriever = MediaMetadataRetriever()
        return try {
            retriever.setDataSource(context, uri)
            val hasVideo = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_HAS_VIDEO)
            if (hasVideo != "yes") {
                Log.i(TAG, "URI $uri has no video track")
                return emptyList()
            }
            val stepUs = (1_000_000L / fps)        // microseconds between frames
            val startUs = startMs * 1000L
            val endUs = startUs + durationMs * 1000L
            val out = ArrayList<Bitmap>(((durationMs * fps) / 1000).toInt().coerceAtLeast(1))
            var t = startUs
            while (t < endUs) {
                val bmp = retriever.getFrameAtTime(t, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                if (bmp != null) out.add(bmp)
                t += stepUs
            }
            out
        } catch (e: Exception) {
            Log.w(TAG, "frame extraction failed for $uri: ${e.message}")
            emptyList()
        } finally {
            try { retriever.release() } catch (_: Exception) {}
        }
    }

    /** True if the URI points to a media resource with a video track. */
    fun hasVideoTrack(uri: Uri): Boolean {
        val retriever = MediaMetadataRetriever()
        return try {
            retriever.setDataSource(context, uri)
            retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_HAS_VIDEO) == "yes"
        } catch (e: Exception) {
            false
        } finally {
            try { retriever.release() } catch (_: Exception) {}
        }
    }

    companion object {
        private const val TAG = "VideoFrameExtractor"
    }
}
