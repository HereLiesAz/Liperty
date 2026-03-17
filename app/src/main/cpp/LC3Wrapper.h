#ifndef LIPERTY_LC3_WRAPPER_H
#define LIPERTY_LC3_WRAPPER_H

#include <vector>
#include <cstdint>

/**
 * LC3Wrapper: Native interface for the Low Complexity Communication Codec (LC3).
 * Used for ultra-low latency (10-15ms) high-fidelity audio transmission in LE Audio.
 */
class LC3Wrapper {
public:
    LC3Wrapper(int sampleRate, int frameDurationUs);
    ~LC3Wrapper();

    /**
     * Encodes PCM 16-bit samples into an LC3 frame.
     * @param pcm Input buffer of samples.
     * @param outEncoded Output buffer for the compressed frame.
     * @return Number of bytes written to outEncoded.
     */
    int encode(const int16_t* pcm, uint8_t* outEncoded);

    /**
     * Decodes an LC3 frame into PCM 16-bit samples.
     * @param encoded Input buffer of the compressed frame.
     * @param encodedSize Size of the input buffer.
     * @param outPcm Output buffer for the decompressed samples.
     * @return Number of samples written to outPcm.
     */
    int decode(const uint8_t* encoded, int encodedSize, int16_t* outPcm);

private:
    void* encoder_ = nullptr;
    void* decoder_ = nullptr;
    int sampleRate_;
    int frameDurationUs_;
    int frameSamples_;
};

#endif // LIPERTY_LC3_WRAPPER_H
