#include <jni.h>
#include <android/log.h>

#define LOG_TAG "VibraPhoneJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#include <opencv2/core.hpp>
#include <vector>
#include <cmath>
#include "AudioEngine.h"
#include "LC3Wrapper.h"

static AudioEngine* gAudioEngine = nullptr;
static LC3Wrapper* gLc3Wrapper = nullptr;

extern "C" JNIEXPORT jint JNICALL
Java_com_hereliesaz_liperty_voicebox_LaryngealSensor_captureBackEmfAudio(
        JNIEnv* env,
        jobject /* this */,
        jshortArray audioBuffer) {
    
    jsize len = env->GetArrayLength(audioBuffer);
    jshort* body = env->GetShortArrayElements(audioBuffer, nullptr);

    // Physics-based simulation of LRA Back-EMF.
    // Models the mass-spring-damper system of the Linear Resonant Actuator.
    // Displacement (x) and velocity (v) are maintained across calls.
    static float x = 0.0f;
    static float v = 0.0f;
    static const float k = 0.15f;    // Spring constant (resonant at ~170Hz)
    static const float c = 0.01f;    // Damping coefficient
    static const float mass = 1.0f;
    static const float dt = 1.0f;    // Time step
    
    // Simulate laryngeal excitation (F0 carrier + jitter)
    static float phase = 0.0f;
    const float f0 = 120.0f; // Fundamental frequency Hz
    const float sampleRate = 16000.0f;

    for (int i = 0; i < len; ++i) {
        // Driving force: modeled as the user's vocal fold vibration
        float force = std::sin(phase) * 0.1f;
        phase += 2.0f * M_PI * f0 / sampleRate;
        if (phase > 2.0f * M_PI) phase -= 2.0f * M_PI;

        // F = ma -> a = F/m - kx/m - cv/m
        float a = (force - k * x - c * v) / mass;
        v += a * dt;
        x += v * dt;

        // Back-EMF is proportional to velocity (e = B * l * v)
        float emf = v * 32768.0f * 0.5f; 
        body[i] = (jshort)std::clamp(emf, -32768.0f, 32767.0f);
    }

    env->ReleaseShortArrayElements(audioBuffer, body, 0);
    return len;
}

extern "C" JNIEXPORT void JNICALL
Java_com_hereliesaz_liperty_dsp_VibraPhoneDSP_nativeSpectralSubtraction(
        JNIEnv* env,
        jobject /* this */,
        jfloatArray inputSignal,
        jfloatArray noiseProfile) {
    
    jsize signalLen = env->GetArrayLength(inputSignal);
    jsize noiseLen = env->GetArrayLength(noiseProfile);
    
    jfloat* signalBody = env->GetFloatArrayElements(inputSignal, nullptr);
    jfloat* noiseBody = env->GetFloatArrayElements(noiseProfile, nullptr);
    
    // 1. Prepare OpenCV Mat for DFT
    // DFT size should be power of 2
    int n = 1;
    while (n < signalLen) n <<= 1;
    
    cv::Mat padded;
    cv::Mat inputMat(1, signalLen, CV_32F, signalBody);
    cv::copyMakeBorder(inputMat, padded, 0, 0, 0, n - signalLen, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);
    
    cv::dft(complexI, complexI);
    
    // 2. Magnitude Subtraction
    cv::split(complexI, planes);
    cv::magnitude(planes[0], planes[1], planes[0]); // planes[0] = magnitude
    
    // Simplified Spectral Subtraction logic
    for (int i = 0; i < n; i++) {
        float noiseMag = (i < noiseLen) ? noiseBody[i] : 0.001f;
        float alpha = 2.0f; // Oversubtraction
        float beta = 0.02f; // Spectral floor
        
        float sub = planes[0].at<float>(i) - (alpha * noiseMag);
        planes[0].at<float>(i) = std::max(sub, beta * planes[0].at<float>(i));
    }
    
    // 3. Reconstruct complex and IDFT
    // We already have smoothed magnitudes in planes[0]. 
    // We need to restore phases. Original phases are in complexI.
    for (int i = 0; i < n; i++) {
        float oldRe = complexI.at<cv::Vec2f>(i)[0];
        float oldIm = complexI.at<cv::Vec2f>(i)[1];
        float oldMag = std::sqrt(oldRe * oldRe + oldIm * oldIm) + 1e-8f;
        
        float scale = planes[0].at<float>(i) / oldMag;
        complexI.at<cv::Vec2f>(i)[0] *= scale;
        complexI.at<cv::Vec2f>(i)[1] *= scale;
    }
    
    cv::idft(complexI, complexI, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    
    // Copy back to inputSignal
    cv::split(complexI, planes);
    for (int i = 0; i < signalLen; i++) {
        signalBody[i] = planes[0].at<float>(i);
    }
    
    env->ReleaseFloatArrayElements(inputSignal, signalBody, 0);
    env->ReleaseFloatArrayElements(noiseProfile, noiseBody, 0);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_hereliesaz_liperty_voicebox_LaryngealSensor_startNativeAudio(
        JNIEnv* /* env */,
        jobject /* this */) {
    if (gAudioEngine == nullptr) {
        gAudioEngine = new AudioEngine();
    }
    return gAudioEngine->start();
}

extern "C" JNIEXPORT void JNICALL
Java_com_hereliesaz_liperty_voicebox_LaryngealSensor_stopNativeAudio(
        JNIEnv* /* env */,
        jobject /* this */) {
    if (gAudioEngine != nullptr) {
        gAudioEngine->stop();
        delete gAudioEngine;
        gAudioEngine = nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_hereliesaz_liperty_voicebox_BluetoothLEAudioManager_initLC3(
        JNIEnv* /* env */,
        jobject /* this */,
        jint sampleRate,
        jint frameDurationUs) {
    if (gLc3Wrapper != nullptr) {
        delete gLc3Wrapper;
    }
    gLc3Wrapper = new LC3Wrapper(sampleRate, frameDurationUs);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_hereliesaz_liperty_voicebox_BluetoothLEAudioManager_nativeEncodeLC3(
        JNIEnv* env,
        jobject /* this */,
        jshortArray pcmData,
        jbyteArray outEncoded) {
    if (!gLc3Wrapper) return 0;

    jshort* pcm = env->GetShortArrayElements(pcmData, nullptr);
    jbyte* encoded = env->GetByteArrayElements(outEncoded, nullptr);

    int result = gLc3Wrapper->encode(pcm, (uint8_t*)encoded);

    env->ReleaseShortArrayElements(pcmData, pcm, JNI_ABORT);
    env->ReleaseByteArrayElements(outEncoded, encoded, 0);
    return result;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_hereliesaz_liperty_voicebox_BluetoothLEAudioManager_nativeDecodeLC3(
        JNIEnv* env,
        jobject /* this */,
        jbyteArray encodedData,
        jshortArray outPcm) {
    if (!gLc3Wrapper) return 0;

    jsize len = env->GetArrayLength(encodedData);
    jbyte* encoded = env->GetByteArrayElements(encodedData, nullptr);
    jshort* pcm = env->GetShortArrayElements(outPcm, nullptr);

    int result = gLc3Wrapper->decode((uint8_t*)encoded, len, pcm);

    env->ReleaseByteArrayElements(encodedData, encoded, JNI_ABORT);
    env->ReleaseShortArrayElements(outPcm, pcm, 0);
    return result;
}

