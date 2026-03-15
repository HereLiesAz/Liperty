#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <android/log.h>
#include <android/bitmap.h>

#define LOG_TAG "LipertyCV"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_hereliesaz_liperty_utils_ImageUtils_pitchSynchronousSpectralSubtractionNative(
        JNIEnv* env,
        jobject /* this */,
        jfloatArray inputSignal) {

    jsize length = env->GetArrayLength(inputSignal);
    jfloat* inputElements = env->GetFloatArrayElements(inputSignal, nullptr);

    // Prepare data for DFT
    cv::Mat inputMat(1, length, CV_32FC1, inputElements);

    // Pad for optimal DFT size
    int dftSize = cv::getOptimalDFTSize(length);
    cv::Mat padded;
    cv::copyMakeBorder(inputMat, padded, 0, 0, 0, dftSize - length, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    // Create complex array (real, imaginary)
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);

    // Forward DFT
    cv::dft(complexI, complexI);

    // Split into magnitude and phase
    cv::split(complexI, planes);
    cv::Mat mag, phase;
    cv::cartToPolar(planes[0], planes[1], mag, phase);

    // Estimate Noise Profile: This implementation calculates the mean magnitude
    // of the current frame and uses a fraction of that as the noiseFloor.
    // It acts as a dynamic spectral filter based on the current frame's content
    // rather than stationary background noise subtraction.
    cv::Scalar meanMag = cv::mean(mag);
    float noiseFloor = meanMag[0] * 0.1f; // Thresholding factor

    float alpha = 2.0f; // Oversubtraction factor
    float spectralFloor = 0.01f;

    // Apply Generalized Spectral Subtraction
    for (int i = 0; i < mag.cols; i++) {
        float m = mag.at<float>(0, i);
        float sub = m - (alpha * noiseFloor);
        mag.at<float>(0, i) = std::max(sub, spectralFloor * m);
    }

    // Reconstruct complex spectrum
    cv::polarToCart(mag, phase, planes[0], planes[1]);
    cv::merge(planes, 2, complexI);

    // Inverse DFT
    cv::Mat inverseTransform;
    cv::idft(complexI, inverseTransform, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

    // Copy back to JNI array
    jfloatArray outputArray = env->NewFloatArray(length);
    env->SetFloatArrayRegion(outputArray, 0, length, (jfloat*)inverseTransform.data);

    env->ReleaseFloatArrayElements(inputSignal, inputElements, JNI_ABORT);

    return outputArray;
}

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
