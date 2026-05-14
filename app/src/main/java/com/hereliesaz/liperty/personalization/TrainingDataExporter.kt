package com.hereliesaz.liperty.personalization

import android.content.ContentValues
import android.content.Context
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import java.io.File
import java.io.FileInputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * Bundles the user's [PairedTrainingStore] directory into a zip and
 * writes it to the device's Downloads folder so the user can pull it
 * off the phone (cloud-sync, USB, sharesheet) and feed it to the
 * offline LoRA training notebook
 * (`tools/train_syncvsr_lora.ipynb`).
 *
 * Why a zip and not raw triples: the on-device store has hundreds of
 * small files (3 per record); a single zip is one transfer, preserves
 * directory structure, and travels through every share / cloud / USB
 * path cleanly. The LoRA notebook unpacks it automatically.
 *
 * Format: the zip's root is whatever [PairedTrainingStore.rootDir]'s
 * basename is (typically "viseme_calibration/"), and inside are flat
 * <id>.audio / .video / .json triples. The notebook walks the tree
 * looking for .json files and pairs them with siblings of the same
 * stem, so any nesting works.
 *
 * Writes via MediaStore on API 29+ (no WRITE_EXTERNAL_STORAGE needed),
 * falls back to Environment.DIRECTORY_DOWNLOADS on older API levels.
 */
object TrainingDataExporter {

    private const val TAG = "TrainingDataExporter"
    private const val MIME = "application/zip"

    /**
     * Bundle [sourceDir] into a zip placed under the user's Downloads
     * folder. Returns a human-readable indicator of where the file
     * landed (file path on legacy API, MediaStore URI string on API
     * 29+), or null on failure.
     */
    fun export(context: Context, sourceDir: File): String? {
        if (!sourceDir.exists() || !sourceDir.isDirectory) {
            Log.w(TAG, "No source dir at ${sourceDir.absolutePath}")
            return null
        }
        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        val zipName = "liperty_calibration_$stamp.zip"

        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                exportViaMediaStore(context, sourceDir, zipName)
            } else {
                exportViaLegacyDownloads(sourceDir, zipName)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Export failed", t)
            null
        }
    }

    private fun exportViaMediaStore(context: Context, sourceDir: File, zipName: String): String? {
        val cv = ContentValues().apply {
            put(MediaStore.Downloads.DISPLAY_NAME, zipName)
            put(MediaStore.Downloads.MIME_TYPE, MIME)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Downloads.IS_PENDING, 1)
                put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
            }
        }
        val resolver = context.contentResolver
        val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, cv) ?: return null
        try {
            resolver.openOutputStream(uri)?.use { os ->
                ZipOutputStream(os.buffered()).use { zip ->
                    writeDirToZip(sourceDir, zip, prefix = sourceDir.name)
                }
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                cv.clear()
                cv.put(MediaStore.Downloads.IS_PENDING, 0)
                resolver.update(uri, cv, null, null)
            }
            Log.i(TAG, "Wrote $zipName via MediaStore: $uri")
            return "Downloads/$zipName"
        } catch (t: Throwable) {
            resolver.delete(uri, null, null)
            throw t
        }
    }

    private fun exportViaLegacyDownloads(sourceDir: File, zipName: String): String? {
        val downloads = Environment.getExternalStoragePublicDirectory(
            Environment.DIRECTORY_DOWNLOADS,
        )
        if (!downloads.exists()) downloads.mkdirs()
        val dest = File(downloads, zipName)
        ZipOutputStream(dest.outputStream().buffered()).use { zip ->
            writeDirToZip(sourceDir, zip, prefix = sourceDir.name)
        }
        Log.i(TAG, "Wrote ${dest.absolutePath}")
        return dest.absolutePath
    }

    private fun writeDirToZip(dir: File, zip: ZipOutputStream, prefix: String) {
        val files = dir.listFiles().orEmpty()
        for (f in files) {
            if (f.isDirectory) {
                writeDirToZip(f, zip, prefix = "$prefix/${f.name}")
                continue
            }
            val entry = ZipEntry("$prefix/${f.name}")
            entry.size = f.length()
            zip.putNextEntry(entry)
            FileInputStream(f).use { it.copyTo(zip) }
            zip.closeEntry()
        }
    }
}
