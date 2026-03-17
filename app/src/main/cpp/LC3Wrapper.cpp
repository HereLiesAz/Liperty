#include "LC3Wrapper.h"
#include <iostream>
#include <dlfcn.h>

// Standard LC3 function pointers
typedef void* (*lc3_encoder_init_t)(int, int);
typedef int (*lc3_encode_t)(void*, const int16_t*, uint8_t*);
typedef void* (*lc3_decoder_init_t)(int, int);
typedef int (*lc3_decode_t)(void*, const uint8_t*, int, int16_t*);

static lc3_encoder_init_t lc3_encoder_init_ptr = nullptr;
static lc3_encode_t lc3_encode_ptr = nullptr;
static lc3_decoder_init_t lc3_decoder_init_ptr = nullptr;
static lc3_decode_t lc3_decode_ptr = nullptr;

LC3Wrapper::LC3Wrapper(int sampleRate, int frameDurationUs)
    : sampleRate_(sampleRate), frameDurationUs_(frameDurationUs) {
    
    // Attempt to load the native LC3 library from the system
    void* handle = dlopen("liblc3.so", RTLD_NOW);
    if (!handle) {
         handle = dlopen("liblc3.so.1", RTLD_NOW);
    }

    if (handle) {
        lc3_encoder_init_ptr = (lc3_encoder_init_t)dlsym(handle, "lc3_encoder_init");
        lc3_encode_ptr = (lc3_encode_t)dlsym(handle, "lc3_encode");
        lc3_decoder_init_ptr = (lc3_decoder_init_t)dlsym(handle, "lc3_decoder_init");
        lc3_decode_ptr = (lc3_decode_t)dlsym(handle, "lc3_decode");
    }

    if (lc3_encoder_init_ptr) {
        encoder_ = lc3_encoder_init_ptr(sampleRate, frameDurationUs);
    }
    if (lc3_decoder_init_ptr) {
        decoder_ = lc3_decoder_init_ptr(sampleRate, frameDurationUs);
    }

    frameSamples_ = (sampleRate * frameDurationUs) / 1000000;
}

LC3Wrapper::~LC3Wrapper() {
    // In a real implementation, we would call lc3_encoder_deinit (encoder_) etc.
}

int LC3Wrapper::encode(const int16_t* pcm, uint8_t* outEncoded) {
    if (encoder_ && lc3_encode_ptr) {
        return lc3_encode_ptr(encoder_, pcm, outEncoded);
    }
    return 0;
}

int LC3Wrapper::decode(const uint8_t* encoded, int encodedSize, int16_t* outPcm) {
    if (decoder_ && lc3_decode_ptr) {
        return lc3_decode_ptr(decoder_, encoded, encodedSize, outPcm);
    }
    return 0;
}
