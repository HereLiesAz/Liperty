// JNI bridge between Kotlin's KenLmScorer and the native KenLM n-gram
// language model. Used by Liperty's CTC + viseme rescoring stack
// (see docs/LM_RESCORING.md).
//
// Build modes:
//   - KENLM_AVAILABLE defined: real `lm::ngram::Model` implementation,
//     links against `libkenlm.a` + `libkenlm_util.a` from
//     `kenlm/android-arm64/` (fetched by setup_libs.sh from
//     `HereLiesAz/liperty-lm/android-arm64`).
//   - KENLM_AVAILABLE undefined: stub. `nativeLoad` returns 0;
//     `KenLmScorer.tryLoad()` then reports `isNativeLoaded = false`
//     and the Kotlin layer no-ops every score call. Build still
//     succeeds on a fresh checkout that hasn't run setup_libs.sh.

#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>

#define LOG_TAG "kenlm_jni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#ifdef KENLM_AVAILABLE
#  include "lm/model.hh"
#  include "lm/virtual_interface.hh"
#endif

extern "C" JNIEXPORT jlong JNICALL
Java_com_hereliesaz_liperty_ml_KenLmScorer_00024Companion_nativeLoad(
        JNIEnv *env, jobject /* this */, jstring jModelPath) {
    const char *modelPath = env->GetStringUTFChars(jModelPath, nullptr);

#ifdef KENLM_AVAILABLE
    LOGI("nativeLoad('%s')", modelPath);
    lm::base::Model *model = nullptr;
    try {
        lm::ngram::Config config;
        config.show_progress = false;
        config.messages = nullptr;
        // LoadVirtual auto-detects the model's binary format (probing,
        // trie, quant_trie, quant_array_trie). Our published LM is
        // trie+q8 -- see tools/kaggle_build_kenlm_android.py's build
        // flags. Returns a polymorphic Model* implementing
        // lm::base::Model's virtual interface.
        model = lm::ngram::LoadVirtual(modelPath, config);
        LOGI("  loaded, state_size=%zu", model->StateSize());
    } catch (const std::exception &e) {
        LOGE("  load failed: %s", e.what());
        delete model;
        model = nullptr;
    } catch (...) {
        LOGE("  load failed: unknown exception");
        model = nullptr;
    }
    env->ReleaseStringUTFChars(jModelPath, modelPath);
    return reinterpret_cast<jlong>(model);
#else
    LOGW("nativeLoad('%s') -- KENLM_AVAILABLE not defined; returning 0", modelPath);
    env->ReleaseStringUTFChars(jModelPath, modelPath);
    return 0L;
#endif
}

extern "C" JNIEXPORT jfloat JNICALL
Java_com_hereliesaz_liperty_ml_KenLmScorer_nativeScore(
        JNIEnv *env, jobject /* this */, jlong handle, jobjectArray jWords) {
#ifdef KENLM_AVAILABLE
    if (handle == 0L) {
        LOGW("nativeScore: null handle");
        return 0.0f;
    }
    auto *model = reinterpret_cast<lm::base::Model *>(handle);
    const lm::base::Vocabulary &vocab = model->BaseVocabulary();

    // KenLM state is an opaque struct whose size depends on the model
    // type (trie vs probing) and KENLM_MAX_ORDER. Allocate from
    // StateSize() and ping-pong two buffers across the word loop.
    const std::size_t state_size = model->StateSize();
    std::vector<char> state_buf(state_size);
    std::vector<char> out_buf(state_size);
    model->BeginSentenceWrite(state_buf.data());

    jint n = env->GetArrayLength(jWords);
    float total = 0.0f;
    for (jint i = 0; i < n; ++i) {
        auto jword = reinterpret_cast<jstring>(env->GetObjectArrayElement(jWords, i));
        if (jword == nullptr) continue;
        const char *word = env->GetStringUTFChars(jword, nullptr);

        // vocab.Index returns 0 (== <unk>) for OOV words. Add the unk
        // penalty just like any other token; the LM was trained with
        // its own unk handling.
        lm::WordIndex wid = vocab.Index(word);
        total += model->BaseScore(state_buf.data(), wid, out_buf.data());

        env->ReleaseStringUTFChars(jword, word);
        env->DeleteLocalRef(jword);

        // Swap buffers: out becomes new state, old state buffer is
        // reused as the next out.
        state_buf.swap(out_buf);
    }
    return total;
#else
    LOGW("nativeScore: KENLM_AVAILABLE not defined; returning 0");
    return 0.0f;
#endif
}

extern "C" JNIEXPORT void JNICALL
Java_com_hereliesaz_liperty_ml_KenLmScorer_nativeFree(
        JNIEnv * /* env */, jobject /* this */, jlong handle) {
#ifdef KENLM_AVAILABLE
    if (handle != 0L) {
        delete reinterpret_cast<lm::base::Model *>(handle);
    }
#else
    (void) handle;
#endif
}
