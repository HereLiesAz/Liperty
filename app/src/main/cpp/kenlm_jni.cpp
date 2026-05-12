// JNI bridge between Kotlin's KenLmScorer and the native KenLM
// n-gram language model. Used by Liperty's CTC + viseme rescoring
// stack (see docs/LM_RESCORING.md).
//
// CURRENT STATUS: stub. The real KenLM source needs to be vendored
// into app/src/main/cpp/kenlm/ (planned in setup_libs.sh) and built
// via CMakeLists.txt. Until then, nativeLoad returns 0, signaling
// to KenLmScorer.tryLoad() that the native library is "present but
// not functional" — which it gracefully falls back from by setting
// isNativeLoaded=false and treating every score() call as a no-op.
//
// Once KenLM is vendored, the three functions below will be replaced
// with:
//   - nativeLoad: open the .bin file via lm::ngram::Model
//   - nativeScore: walk the words through model.FullScoreForgotState,
//     accumulating log10 probabilities
//   - nativeFree: delete the Model handle

#include <jni.h>
#include <android/log.h>

#define LOG_TAG "kenlm_jni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)

extern "C" JNIEXPORT jlong JNICALL
Java_com_hereliesaz_liperty_ml_KenLmScorer_00024Companion_nativeLoad(
        JNIEnv *env, jobject /* this */, jstring jModelPath) {
    const char *modelPath = env->GetStringUTFChars(jModelPath, nullptr);
    LOGI("KenLM nativeLoad('%s') called — STUB IMPL, returning 0", modelPath);
    env->ReleaseStringUTFChars(jModelPath, modelPath);
    // Returning 0 signals to KenLmScorer.tryLoad that loading failed,
    // and the Kotlin layer falls back to no-op scoring. This is the
    // correct behavior while the KenLM source is not yet vendored.
    return 0L;
}

extern "C" JNIEXPORT jfloat JNICALL
Java_com_hereliesaz_liperty_ml_KenLmScorer_nativeScore(
        JNIEnv *env, jobject /* this */, jlong handle, jobjectArray jWords) {
    // Unreachable in practice: KenLmScorer.score() short-circuits when
    // handle == 0, and tryLoad() never returns an instance with a
    // non-zero handle while this is a stub.
    if (handle == 0L) {
        LOGW("nativeScore called with handle=0; this should be unreachable");
    }
    return 0.0f;
}

extern "C" JNIEXPORT void JNICALL
Java_com_hereliesaz_liperty_ml_KenLmScorer_nativeFree(
        JNIEnv *env, jobject /* this */, jlong handle) {
    if (handle != 0L) {
        LOGI("nativeFree(handle=%lld) called", (long long) handle);
    }
}
