#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <android/log.h>

#define LOG_TAG "LipertyCV"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" JNIEXPORT void JNICALL
Java_com_HereLiesAz_Liperty_utils_ImageUtils_applyHistogramEqualization(
        JNIEnv* env,
        jobject /* this */,
        jlong matAddr) {

    cv::Mat* pMat = (cv::Mat*)matAddr;

    if (pMat->empty()) {
        LOGE("Input matrix is empty");
        return;
    }

    if (pMat->channels() != 1) {
        LOGE("Histogram equalization requires a single-channel (grayscale) image. Channels: %d", pMat->channels());
        // Optionally convert to grayscale here if needed, but the contract usually expects grayscale
        return;
    }

    try {
        cv::equalizeHist(*pMat, *pMat);
    } catch (const cv::Exception& e) {
        LOGE("OpenCV Exception during histogram equalization: %s", e.what());
    }
}
