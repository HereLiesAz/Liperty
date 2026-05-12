"""Cross-compile KenLM for Android (arm64-v8a) on Kaggle.

Produces a static library + headers that Liperty's NDK build can link
against to enable on-device n-gram language-model scoring. This is the
"libkenlm.so / libkenlm.a" output the Phase A3b/c plan needs (see
docs/LM_RESCORING.md).

USAGE (paste into a Kaggle notebook cell, CPU-only is fine, ~10 minutes):
  1. Add HF_TOKEN as a notebook secret (Add-ons -> Secrets).
  2. Paste the contents of this file into one cell.
  3. Run. Outputs upload to HereLiesAz/liperty-lm under android-arm64/.

Output (on HF):
  - android-arm64/libkenlm.a       static lib for arm64-v8a
  - android-arm64/libkenlm_util.a  KenLM's util sub-library
  - android-arm64/include/lm/*.hh  C++ headers for linking
  - android-arm64/include/util/*.hh

Local integration (in Liperty repo, after this runs):
  1. setup_libs.sh pulls these from HF into app/src/main/cpp/kenlm/.
  2. app/src/main/cpp/CMakeLists.txt adds them to liperty_cv via
     add_library(kenlm STATIC IMPORTED) + target_link_libraries(...).
  3. kenlm_jni.cpp's stub gets replaced with real lm::ngram::Model calls.
"""

import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# 0. HF auth
# ---------------------------------------------------------------------------
if "HF_TOKEN" not in os.environ:
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        raise RuntimeError(
            "HF_TOKEN not found. Add it as a notebook secret first."
        ) from e

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "huggingface_hub>=0.27,<1.0"])
from huggingface_hub import login, upload_folder, create_repo
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)


# ---------------------------------------------------------------------------
# 1. Install Android NDK (one-time per Kaggle session)
# ---------------------------------------------------------------------------
NDK_VERSION = "r26d"
NDK_ZIP = f"android-ndk-{NDK_VERSION}-linux.zip"
NDK_URL = f"https://dl.google.com/android/repository/{NDK_ZIP}"
WORK = "/kaggle/working/kenlm-android"
os.makedirs(WORK, exist_ok=True)
NDK = f"{WORK}/android-ndk-{NDK_VERSION}"

if not os.path.exists(NDK):
    print(f"Downloading Android NDK {NDK_VERSION} ...")
    subprocess.check_call(["wget", "-q", "-O", f"{WORK}/{NDK_ZIP}", NDK_URL])
    print(f"  {os.path.getsize(f'{WORK}/{NDK_ZIP}') / 1e6:.0f} MB downloaded")
    print("Extracting NDK ...")
    subprocess.check_call(["unzip", "-q", f"{WORK}/{NDK_ZIP}", "-d", WORK])
    os.remove(f"{WORK}/{NDK_ZIP}")
print(f"NDK ready at: {NDK}")


# ---------------------------------------------------------------------------
# 2. Clone KenLM source
# ---------------------------------------------------------------------------
KENLM_SRC = f"{WORK}/kenlm-src"
if not os.path.exists(KENLM_SRC):
    print("Cloning KenLM ...")
    subprocess.check_call([
        "git", "clone", "--depth", "1",
        "https://github.com/kpu/kenlm.git", KENLM_SRC,
    ])
print(f"KenLM source: {KENLM_SRC}")


# ---------------------------------------------------------------------------
# 3. Configure & build for arm64-v8a
# ---------------------------------------------------------------------------
BUILD = f"{WORK}/build-arm64"
os.makedirs(BUILD, exist_ok=True)

# CMake flags rationale:
# - CMAKE_TOOLCHAIN_FILE: standard Android NDK CMake integration.
# - ANDROID_ABI=arm64-v8a: 64-bit ARM; covers all reasonably modern Android phones.
# - ANDROID_PLATFORM=android-26: matches Liperty's minSdk in app/build.gradle.kts.
# - BUILD_TESTING=OFF: skips the gtest dep + saves build time.
# - KENLM_MAX_ORDER=6: handles up to 6-gram models; our shipped LibriSpeech LM
#   is 3-gram but room for future personal LMs.
# - ENABLE_PYTHON=OFF: skip the Python bindings (no Python on Android target).
# - FORCE_STATIC=ON: link statically; we want .a files to combine into liperty_cv.so.
# - Boost: pass -DKENLM_USE_BOOST=OFF if available, else hope its default off-path
#   triggers. Recent kenlm has reduced Boost coupling; inference code doesn't need it.
# - WITH_THREADS=OFF: KenLM threading is only used by build tools, not inference.
print("\nConfiguring with CMake ...")
cmake_cmd = [
    "cmake",
    f"-S{KENLM_SRC}",
    f"-B{BUILD}",
    f"-DCMAKE_TOOLCHAIN_FILE={NDK}/build/cmake/android.toolchain.cmake",
    "-DANDROID_ABI=arm64-v8a",
    "-DANDROID_PLATFORM=android-26",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_TESTING=OFF",
    "-DKENLM_MAX_ORDER=6",
    "-DENABLE_PYTHON=OFF",
    "-DFORCE_STATIC=ON",
    # The default Boost detection may try to find host Boost — skip it.
    "-DBoost_NO_BOOST_CMAKE=ON",
    "-DBoost_NO_SYSTEM_PATHS=ON",
]
print("  cmake:", " ".join(cmake_cmd))
subprocess.check_call(cmake_cmd)

print("\nBuilding (kenlm + kenlm_util only — skip tools) ...")
# Targets: kenlm (the main inference library) + kenlm_util (KenLM's util
# subdir built as a static lib). The lmplz / build_binary / query targets
# require Boost program_options + filesystem and aren't useful on-device.
# We're just shipping the inference path.
subprocess.check_call([
    "cmake", "--build", BUILD,
    "--target", "kenlm", "kenlm_util",
    "--config", "Release",
    "--parallel", str(os.cpu_count() or 4),
])


# ---------------------------------------------------------------------------
# 4. Collect outputs
# ---------------------------------------------------------------------------
import shutil
OUT = f"{WORK}/android-arm64"
INC = f"{OUT}/include"
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{INC}/lm", exist_ok=True)
os.makedirs(f"{INC}/util", exist_ok=True)

# Find the built static libs (typically under build/lib/ or build/lm/, build/util/)
print(f"\nSearching {BUILD} for .a files ...")
found_libs = []
for root, _, files in os.walk(BUILD):
    for f in files:
        if f.endswith(".a") and ("kenlm" in f or "lm" == os.path.basename(root) or "util" == os.path.basename(root)):
            src = os.path.join(root, f)
            dst = os.path.join(OUT, f)
            shutil.copy(src, dst)
            found_libs.append(dst)
            print(f"  {f}  ({os.path.getsize(dst) / 1e6:.1f} MB)")

if not any("libkenlm" in p for p in found_libs):
    raise RuntimeError(
        "libkenlm.a not found in build output. Check the cmake build above "
        "for errors. KenLM's CMakeLists may have changed target names; "
        "search the build dir manually."
    )

# Copy headers
print("Copying headers ...")
for sub in ("lm", "util"):
    src_dir = os.path.join(KENLM_SRC, sub)
    dst_dir = os.path.join(INC, sub)
    for root, _, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        for f in files:
            if f.endswith(".hh") or f.endswith(".h"):
                d = os.path.join(dst_dir, rel)
                os.makedirs(d, exist_ok=True)
                shutil.copy(os.path.join(root, f), os.path.join(d, f))
print(f"  headers: $(find {INC} -name '*.hh' | wc -l) files")


# ---------------------------------------------------------------------------
# 5. Write README and upload
# ---------------------------------------------------------------------------
readme = f"""# KenLM Android prebuilt — arm64-v8a

Static libraries + headers for cross-compiled KenLM, suitable for
linking into Liperty's `libliperty_cv.so` (see
`app/src/main/cpp/CMakeLists.txt`).

## Contents

| Path | Purpose |
|---|---|
| `android-arm64/libkenlm.a`        | Main inference lib (`lm::ngram::Model` etc.) |
| `android-arm64/libkenlm_util.a`   | KenLM's util sub-library (string handling, file I/O) |
| `android-arm64/include/lm/*.hh`   | Public headers for the inference API |
| `android-arm64/include/util/*.hh` | Util headers |

## Build info

- Android NDK: {NDK_VERSION}
- minSdk: android-26 (matches Liperty)
- KENLM_MAX_ORDER: 6
- Boost: disabled (inference doesn't need it; would only be needed for
  build tools like `lmplz` which aren't on-device anyway)
- Threads: KenLM's threading is build-tools only; inference is single-threaded

## Liperty integration

In `app/src/main/cpp/CMakeLists.txt`:

```cmake
set(KENLM_ANDROID_DIR ${{CMAKE_SOURCE_DIR}}/kenlm/android-arm64)
add_library(kenlm STATIC IMPORTED)
set_target_properties(kenlm PROPERTIES
    IMPORTED_LOCATION ${{KENLM_ANDROID_DIR}}/libkenlm.a)
add_library(kenlm_util STATIC IMPORTED)
set_target_properties(kenlm_util PROPERTIES
    IMPORTED_LOCATION ${{KENLM_ANDROID_DIR}}/libkenlm_util.a)
include_directories(${{KENLM_ANDROID_DIR}}/include)
target_link_libraries(liperty_cv ... kenlm kenlm_util ${{log-lib}})
```

In `app/src/main/cpp/kenlm_jni.cpp`, replace the stub with real
`lm::ngram::Model` calls — see `docs/LM_RESCORING.md` for the API
sketch.

## Source

Built by [`tools/kaggle_build_kenlm_android.py`](https://github.com/HereLiesAz/Liperty/blob/main/tools/kaggle_build_kenlm_android.py).
"""
with open(os.path.join(OUT, "README.md"), "w") as f:
    f.write(readme)

print(f"\nUploading android-arm64/ to HereLiesAz/liperty-lm ...")
create_repo("HereLiesAz/liperty-lm", repo_type="model", private=False, exist_ok=True)
upload_folder(
    folder_path=OUT,
    path_in_repo="android-arm64",
    repo_id="HereLiesAz/liperty-lm",
    repo_type="model",
    commit_message=f"KenLM Android prebuilt (NDK {NDK_VERSION}, arm64-v8a)",
)
print("Uploaded -> https://huggingface.co/HereLiesAz/liperty-lm/tree/main/android-arm64")
print("\nDone. Pull these into Liperty's app/src/main/cpp/kenlm/ via setup_libs.sh.")
