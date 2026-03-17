#include <jni.h>
#include <aaudio/AAudio.h>
#include <android/log.h>
#include <atomic>
#include <vector>
#include <string.h>
#include <algorithm>
#include <mutex>

#define LOG_TAG "NativeAudioPlayer"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

class LockFreeRingBuffer {
private:
    std::vector<float> buffer;
    size_t capacity;
    std::atomic<size_t> writeIndex{0};
    std::atomic<size_t> readIndex{0};

public:
    LockFreeRingBuffer(size_t size) : capacity(size) {
        buffer.resize(size, 0.0f);
    }

    size_t availableRead() const {
        return writeIndex.load(std::memory_order_acquire) - readIndex.load(std::memory_order_relaxed);
    }

    size_t availableWrite() const {
        return capacity - availableRead();
    }

    bool write(const float* data, size_t count) {
        size_t currentWrite = writeIndex.load(std::memory_order_relaxed);
        size_t currentRead = readIndex.load(std::memory_order_acquire);
        size_t available = capacity - (currentWrite - currentRead);

        if (count > available) return false; // Buffer full

        size_t index1 = currentWrite % capacity;
        size_t count1 = std::min(count, capacity - index1);
        size_t count2 = count - count1;

        memcpy(buffer.data() + index1, data, count1 * sizeof(float));
        if (count2 > 0) {
            memcpy(buffer.data(), data + count1, count2 * sizeof(float));
        }

        writeIndex.store(currentWrite + count, std::memory_order_release);
        return true;
    }

    size_t read(float* data, size_t count) {
        size_t avail = availableRead();
        if (avail == 0) return 0; // Buffer empty

        size_t toRead = std::min(count, avail);
        size_t currentRead = readIndex.load(std::memory_order_relaxed);
        size_t index1 = currentRead % capacity;
        size_t count1 = std::min(toRead, capacity - index1);
        size_t count2 = toRead - count1;

        memcpy(data, buffer.data() + index1, count1 * sizeof(float));
        if (count2 > 0) {
            memcpy(data + count1, buffer.data(), count2 * sizeof(float));
        }

        readIndex.store(currentRead + toRead, std::memory_order_release);
        return toRead;
    }

    void clear() {
        writeIndex.store(0, std::memory_order_release);
        readIndex.store(0, std::memory_order_release);
    }
};

// Global instance (simplified for JNI context)
std::mutex globalAudioMutex;
LockFreeRingBuffer* audioRingBuffer = nullptr;
AAudioStream *audioStream = nullptr;

static aaudio_data_callback_result_t dataCallback(
        AAudioStream *stream,
        void *userData,
        void *audioData,
        int32_t numFrames) {

    float *floatData = static_cast<float *>(audioData);
    std::unique_lock<std::mutex> lock(globalAudioMutex, std::try_to_lock);

    if (lock.owns_lock() && audioRingBuffer != nullptr) {
        size_t readCount = audioRingBuffer->read(floatData, numFrames);
        if (readCount < numFrames) {
            // Underflow: pad with zeros
            memset(floatData + readCount, 0, (numFrames - readCount) * sizeof(float));
        }
    } else {
        memset(floatData, 0, numFrames * sizeof(float));
    }

    return AAUDIO_CALLBACK_RESULT_CONTINUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_hereliesaz_liperty_voicebox_NativeAudioPlayer_startPlaybackNative(
        JNIEnv* env,
        jobject /* this */) {

    std::lock_guard<std::mutex> lock(globalAudioMutex);
    if (audioStream != nullptr) {
        return JNI_TRUE;
    }

    // Create a large enough buffer (e.g. 1 second at 16kHz)
    audioRingBuffer = new LockFreeRingBuffer(16000 * 2);

    AAudioStreamBuilder *builder;
    AAudio_createStreamBuilder(&builder);
    AAudioStreamBuilder_setPerformanceMode(builder, AAUDIO_PERFORMANCE_MODE_LOW_LATENCY);
    AAudioStreamBuilder_setSharingMode(builder, AAUDIO_SHARING_MODE_EXCLUSIVE);
    AAudioStreamBuilder_setFormat(builder, AAUDIO_FORMAT_PCM_FLOAT);
    AAudioStreamBuilder_setChannelCount(builder, 1);
    AAudioStreamBuilder_setSampleRate(builder, 16000);
    AAudioStreamBuilder_setDataCallback(builder, dataCallback, nullptr);

    aaudio_result_t result = AAudioStreamBuilder_openStream(builder, &audioStream);
    AAudioStreamBuilder_delete(builder);

    if (result != AAUDIO_OK) {
        LOGE("Error opening AAudio stream: %s", AAudio_convertResultToText(result));
        delete audioRingBuffer;
        audioRingBuffer = nullptr;
        return JNI_FALSE;
    }

    result = AAudioStream_requestStart(audioStream);
    if (result != AAUDIO_OK) {
        LOGE("Error starting AAudio stream: %s", AAudio_convertResultToText(result));
        AAudioStream_close(audioStream);
        audioStream = nullptr;
        delete audioRingBuffer;
        audioRingBuffer = nullptr;
        return JNI_FALSE;
    }

    LOGI("AAudio stream started successfully.");
    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_hereliesaz_liperty_voicebox_NativeAudioPlayer_writeAudioDataNative(
        JNIEnv* env,
        jobject /* this */,
        jfloatArray pcmData) {

    jsize length = env->GetArrayLength(pcmData);
    jfloat* elements = env->GetFloatArrayElements(pcmData, nullptr);

    std::lock_guard<std::mutex> lock(globalAudioMutex);
    if (audioRingBuffer != nullptr) {
        if (!audioRingBuffer->write(elements, length)) {
            LOGE("Audio ring buffer overflow, dropping %d samples", length);
        }
    }

    env->ReleaseFloatArrayElements(pcmData, elements, JNI_ABORT);
}

extern "C" JNIEXPORT void JNICALL
Java_com_hereliesaz_liperty_voicebox_NativeAudioPlayer_stopPlaybackNative(
        JNIEnv* env,
        jobject /* this */) {
    std::lock_guard<std::mutex> lock(globalAudioMutex);
    if (audioStream != nullptr) {
        AAudioStream_requestStop(audioStream);
        AAudioStream_close(audioStream);
        audioStream = nullptr;

        if (audioRingBuffer != nullptr) {
            delete audioRingBuffer;
            audioRingBuffer = nullptr;
        }
        LOGI("AAudio stream stopped.");
    }
}
