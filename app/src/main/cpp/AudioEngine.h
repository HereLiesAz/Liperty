#ifndef LIPERTY_AUDIO_ENGINE_H
#define LIPERTY_AUDIO_ENGINE_H

#include <aaudio/AAudio.h>
#include <vector>
#include <mutex>

class AudioEngine {
public:
    AudioEngine();
    ~AudioEngine();

    bool start();
    void stop();

    // Callback for AAudio
    aaudio_data_callback_result_t onAudioReady(
            AAudioStream *stream,
            void *userData,
            void *audioData,
            int32_t numFrames);

private:
    AAudioStream *stream_;
    std::mutex streamMutex_;
    
    // For processing
    std::vector<float> inputBuffer_;
    std::vector<float> outputBuffer_;
};

#endif // LIPERTY_AUDIO_ENGINE_H
