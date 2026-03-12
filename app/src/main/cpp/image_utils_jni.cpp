#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <android/log.h>
#include <android/bitmap.h>

#define LOG_TAG "LipertyCV"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" JNIEXPORT void JNICALL
Java_com_hereliesaz_liperty_utils_ImageUtils_applyHistogramEqualizationNative(
        JNIEnv* env,
        jobject /* this */,
        jobject bitmap) {

    AndroidBitmapInfo info;
    void* pixels;
    int ret;

    if ((ret = AndroidBitmap_getInfo(env, bitmap, &info)) < 0) {
        LOGE("AndroidBitmap_getInfo() failed ! error=%d", ret);
        return;
    }

    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        LOGE("Bitmap format is not RGBA_8888 !");
        return;
    }

    if ((ret = AndroidBitmap_lockPixels(env, bitmap, &pixels)) < 0) {
        LOGE("AndroidBitmap_lockPixels() failed ! error=%d", ret);
        return;
    }

    try {
        // Create a cv::Mat that points to the bitmap pixels
        cv::Mat mat(info.height, info.width, CV_8UC4, pixels);

        // Histogram equalization requires grayscale
        cv::Mat gray;
        cv::cvtColor(mat, gray, cv::COLOR_RGBA2GRAY);
        cv::equalizeHist(gray, gray);

        // Convert back to RGBA
        cv::cvtColor(gray, mat, cv::COLOR_GRAY2RGBA);

    } catch (const cv::Exception& e) {
        LOGE("OpenCV Exception: %s", e.what());
    }

    AndroidBitmap_unlockPixels(env, bitmap);
}

extern "C" JNIEXPORT void JNICALL
Java_com_hereliesaz_liperty_utils_ImageUtils_applyNormalizationNative(
        JNIEnv* env,
        jobject /* this */,
        jobject bitmap) {

    AndroidBitmapInfo info;
    void* pixels;
    int ret;

    if ((ret = AndroidBitmap_getInfo(env, bitmap, &info)) < 0) return;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) return;
    if ((ret = AndroidBitmap_lockPixels(env, bitmap, &pixels)) < 0) return;

    try {
        cv::Mat mat(info.height, info.width, CV_8UC4, pixels);
        cv::Mat gray;
        cv::cvtColor(mat, gray, cv::COLOR_RGBA2GRAY);

        // 1. Gaussian Blur (3x3)
        cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);

        // 2. Histogram Equalization
        cv::equalizeHist(gray, gray);

        // Convert back
        cv::cvtColor(gray, mat, cv::COLOR_GRAY2RGBA);

    } catch (const cv::Exception& e) {
        LOGE("OpenCV Exception: %s", e.what());
    }

    AndroidBitmap_unlockPixels(env, bitmap);
}
