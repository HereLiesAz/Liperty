package com.hereliesaz.liperty.ml

import android.util.Log

/**
 * KenLM-backed [LanguageModelScorer]. Wraps a native KenLM `Model`
 * (`libkenlm.so` loaded via JNI) and exposes a Kotlin-friendly
 * `score(words)` API used by [SubwordCtcBeamDecoder] for n-best
 * rescoring.
 *
 * The companion's [tryLoad] is the canonical constructor — it absorbs
 * any `UnsatisfiedLinkError` (no `libkenlm.so` packaged in the APK yet
 * — see `app/src/main/cpp/CMakeLists.txt`) or missing-file errors and
 * returns an instance whose [score] is a no-op (always 0). That way
 * MainActivity can wire this in immediately and the rescoring path is
 * gated only on the LM artifact actually being present + JNI compiled,
 * not on the developer remembering to instantiate a different class.
 *
 * Vocab is uppercase (LibriSpeech convention). [score] uppercases each
 * input word before calling into native.
 *
 * Thread-safety: KenLM models are stateless (state objects are passed
 * in/out per call), so concurrent calls into [score] are safe on the
 * native side. The Kotlin handle, however, is not synchronized — call
 * [close] only when no inference threads are still running.
 */
class KenLmScorer private constructor(
    /** True if native KenLM was loaded successfully AND the model file
     *  was found. When false, [score] returns 0 unconditionally. */
    val isNativeLoaded: Boolean,
    private var nativeHandle: Long = 0L,
) : LanguageModelScorer {

    override fun score(words: List<String>): Float {
        if (!isNativeLoaded || nativeHandle == 0L) return 0f
        if (words.isEmpty()) return 0f
        // LibriSpeech LM vocabulary is uppercase-normalized — uppercase the
        // input here so callers don't need to remember.
        val upper = Array(words.size) { words[it].uppercase() }
        return try {
            nativeScore(nativeHandle, upper)
        } catch (e: Throwable) {
            Log.w(TAG, "nativeScore failed: ${e.message}")
            0f
        }
    }

    override fun close() {
        if (nativeHandle != 0L) {
            try {
                nativeFree(nativeHandle)
            } catch (e: Throwable) {
                Log.w(TAG, "nativeFree failed: ${e.message}")
            }
            nativeHandle = 0L
        }
    }

    private external fun nativeScore(handle: Long, words: Array<String>): Float
    private external fun nativeFree(handle: Long)

    companion object {
        private const val TAG = "KenLmScorer"

        @Volatile private var libraryLoaded: Boolean? = null

        /** True if the JNI bridge (which lives in `libliperty_cv.so` along
         *  with the OpenCV image utils and audio engine) is loadable in
         *  this process. Cached after first attempt so we don't retry on
         *  every constructor call. */
        private fun ensureLibraryLoaded(): Boolean {
            libraryLoaded?.let { return it }
            synchronized(this) {
                libraryLoaded?.let { return it }
                return try {
                    System.loadLibrary("liperty_cv")
                    Log.i(TAG, "libliperty_cv.so loaded (KenLM JNI inside)")
                    libraryLoaded = true
                    true
                } catch (e: UnsatisfiedLinkError) {
                    Log.i(TAG, "libliperty_cv.so unavailable — LM scoring disabled (${e.message})")
                    libraryLoaded = false
                    false
                } catch (e: Throwable) {
                    Log.w(TAG, "libliperty_cv.so load threw: ${e.message}")
                    libraryLoaded = false
                    false
                }
            }
        }

        @JvmStatic private external fun nativeLoad(modelPath: String): Long

        /**
         * Try to instantiate the scorer. Never throws — on any failure
         * (native lib missing, model file missing, native ctor returns 0),
         * returns a scorer whose [score] is a no-op. Inspect
         * [isNativeLoaded] to see whether the real LM is wired up.
         */
        fun tryLoad(modelPath: String): KenLmScorer {
            if (!ensureLibraryLoaded()) return KenLmScorer(isNativeLoaded = false)
            val handle = try {
                nativeLoad(modelPath)
            } catch (e: Throwable) {
                Log.w(TAG, "nativeLoad('$modelPath') failed: ${e.message}")
                0L
            }
            if (handle == 0L) {
                Log.w(TAG, "nativeLoad returned 0 for '$modelPath' (model file missing?)")
                return KenLmScorer(isNativeLoaded = false)
            }
            return KenLmScorer(isNativeLoaded = true, nativeHandle = handle)
        }
    }
}
