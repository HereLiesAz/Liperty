#include "AudioEngine.h"
#include <android/log.h>
#include <cstring>

#define TAG "AudioEngine"

static aaudio_data_callback_result_t dataCallback(
        AAudioStream *stream,
        void *userData,
        void *audioData,
        int32_t numFrames) {
    return ((AudioEngine *)userData)->onAudioReady(stream, userData, audioData, numFrames);
}

static void errorCallback(AAudioStream *stream,
                   void *userData,
                   aaudio_result_t error) {
    if (error == AAUDIO_ERROR_DISCONNECTED) {
        // Handle stream restart on separate thread
    }
}

AudioEngine::AudioEngine() : stream_(nullptr) {}

AudioEngine::~AudioEngine() {
    stop();
}

bool AudioEngine::start() {
    AAudioStreamBuilder *builder;
    aaudio_result_t result = AAudio_createStreamBuilder(&builder);
    if (result != AAUDIO_OK) return false;

    AAudioStreamBuilder_setDirection(builder, AAUDIO_DIRECTION_OUTPUT);
    AAudioStreamBuilder_setPerformanceMode(builder, AAUDIO_PERFORMANCE_MODE_LOW_LATENCY);
    AAudioStreamBuilder_setFormat(builder, AAUDIO_FORMAT_PCM_FLOAT);
    AAudioStreamBuilder_setChannelCount(builder, 1);
    AAudioStreamBuilder_setDataCallback(builder, dataCallback, this);
    AAudioStreamBuilder_setErrorCallback(builder, errorCallback, this);

    result = AAudioStreamBuilder_openStream(builder, &stream_);
    AAudioStreamBuilder_delete(builder);

    if (result != AAUDIO_OK) return false;

    result = AAudioStream_requestStart(stream_);
    return (result == AAUDIO_OK);
}

void AudioEngine::stop() {
    std::lock_guard<std::mutex> lock(streamMutex_);
    if (stream_ != nullptr) {
        AAudioStream_requestStop(stream_);
        AAudioStream_close(stream_);
        stream_ = nullptr;
    }
}

aaudio_data_callback_result_t AudioEngine::onAudioReady(
        AAudioStream *stream,
        void *userData,
        void *audioData,
        int32_t numFrames) {
    
    // Fill with silence if no data
    std::memset(audioData, 0, numFrames * sizeof(float));
    
    // In a real implementation, we would pull from a jitter buffer
    // or run the DSP pipeline here if input was also AAudio.
    
    return AAUDIO_CALLBACK_RESULT_CONTINUE;
}
