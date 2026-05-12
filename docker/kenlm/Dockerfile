# KenLM toolchain image — builds ARPA -> binary KenLM files for shipping
# alongside Liperty. Single-purpose; the image only exists so the host
# doesn't need a C++ toolchain installed to convert an LM.
#
# Build:
#   docker build -t liperty-kenlm -f docker/kenlm/Dockerfile .
#
# Run (convert an ARPA file mounted in /work to a binary):
#   docker run --rm -v "$(pwd)/lm:/work" liperty-kenlm \
#       build_binary /work/4-gram.pruned.3e-7.arpa /work/4-gram.bin
#
# This image is intentionally small (~200 MB) — KenLM is a few thousand
# lines of C++ with minimal deps. Once it produces the binary we don't
# need to rebuild unless KenLM changes.

FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libboost-program-options-dev \
        libboost-thread-dev \
        libboost-system-dev \
        libboost-test-dev \
        libeigen3-dev \
        zlib1g-dev \
        libbz2-dev \
        liblzma-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 https://github.com/kpu/kenlm.git \
    && cd kenlm \
    && mkdir build \
    && cd build \
    && cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF \
        -DKENLM_MAX_ORDER=6 \
    && make -j$(nproc) lmplz build_binary query

# Symlink the tools into PATH so the entrypoint can use bare names.
RUN ln -s /src/kenlm/build/bin/lmplz       /usr/local/bin/lmplz \
    && ln -s /src/kenlm/build/bin/build_binary /usr/local/bin/build_binary \
    && ln -s /src/kenlm/build/bin/query    /usr/local/bin/query

WORKDIR /work
