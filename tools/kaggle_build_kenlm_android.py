"""Cross-compile KenLM for Android (arm64-v8a) on Kaggle.

Produces inference-only static libraries + headers suitable for linking
into Liperty's `libliperty_cv.so` to enable on-device n-gram LM scoring
(see docs/LM_RESCORING.md).

The PoC ran successfully on Kaggle on 2026-05-12. Outputs at
HereLiesAz/liperty-lm/tree/main/android-arm64.

USAGE (paste into a Kaggle notebook cell, CPU is fine, ~10 minutes):
  1. HF_TOKEN secret set (Add-ons -> Secrets).
  2. Internet enabled in the right-panel Settings.
  3. Paste and run.

Output (HF):
  - android-arm64/libkenlm.a         (~10 MB)
  - android-arm64/libkenlm_util.a    (~3 MB)
  - android-arm64/include/lm/*.hh    inference headers
  - android-arm64/include/util/*.hh  util headers
  - android-arm64/README.md          consumer documentation

4 patches landed during the build iteration that are now baked in:
  1. KenLM's CMakeLists.txt has `find_package(Boost REQUIRED)` but the
     inference-only kenlm/kenlm_util targets don't link Boost. We patch
     to drop REQUIRED so the configure proceeds.
  2. CMake's FindBoost still fails the version check (Boost not
     installed for Android NDK). We provide a stub include directory
     with a `boost/version.hpp` reporting 1.81.0 + strip the version
     constraint from KenLM's find_package() calls.
  3. `util/stream/` and `lm/common/` source files #include
     `<boost/thread/mutex.hpp>` at compile time -- these are
     training-pipeline code paths we don't need for inference. We
     remove those subdirectories from the kenlm/kenlm_util targets.
  4. KenLM's CLI exe targets (lmplz, build_binary, query,
     kenlm_benchmark, fragment) link Boost::program_options. We
     comment out their AddExes() call so they're not even attempted.

Local integration (next, in Liperty repo):
  - setup_libs.sh pulls android-arm64/ -> app/src/main/cpp/kenlm/.
  - app/src/main/cpp/CMakeLists.txt declares kenlm + kenlm_util as
    IMPORTED static libs, adds include dir, links into liperty_cv.
  - Replace the kenlm_jni.cpp stub with real lm::ngram::Model calls.
"""

import os
import re
import shutil
import subprocess
import sys

# ---------------------------------------------------------------------------
# 0. HF auth + dep
# ---------------------------------------------------------------------------
if "HF_TOKEN" not in os.environ:
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        raise RuntimeError("HF_TOKEN not found in Kaggle secrets") from e

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "huggingface_hub>=0.27,<1.0"])

WORK = "/kaggle/working/kenlm-android"
os.makedirs(WORK, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Android NDK
# ---------------------------------------------------------------------------
NDK_VERSION = "r26d"
NDK_ZIP = f"android-ndk-{NDK_VERSION}-linux.zip"
NDK_URL = f"https://dl.google.com/android/repository/{NDK_ZIP}"
NDK = f"{WORK}/android-ndk-{NDK_VERSION}"

if not os.path.exists(NDK):
    print(f"[+] Downloading Android NDK {NDK_VERSION} ...")
    subprocess.check_call(["wget", "-q", "-O", f"{WORK}/{NDK_ZIP}", NDK_URL])
    subprocess.check_call(["unzip", "-q", f"{WORK}/{NDK_ZIP}", "-d", WORK])
    os.remove(f"{WORK}/{NDK_ZIP}")
print(f"NDK: {NDK}")

# ---------------------------------------------------------------------------
# 2. KenLM source
# ---------------------------------------------------------------------------
KENLM_SRC = f"{WORK}/kenlm-src"
if not os.path.exists(KENLM_SRC):
    subprocess.check_call([
        "git", "clone", "--depth", "1",
        "https://github.com/kpu/kenlm.git", KENLM_SRC,
    ])
print(f"KenLM source: {KENLM_SRC}")


# ---------------------------------------------------------------------------
# 3. Patch CMakeLists.txt: drop Boost REQUIRED + version + Boost-needing
#    subdirectories from the inference-only build.
# ---------------------------------------------------------------------------
# Patch 1+2: drop REQUIRED + version constraint from find_package(Boost ...)
for cml in [
    f"{KENLM_SRC}/CMakeLists.txt",
    f"{KENLM_SRC}/lm/CMakeLists.txt",
    f"{KENLM_SRC}/lm/filter/CMakeLists.txt",
    f"{KENLM_SRC}/lm/builder/CMakeLists.txt",
    f"{KENLM_SRC}/util/CMakeLists.txt",
]:
    if not os.path.exists(cml):
        continue
    with open(cml, "r") as f:
        c = f.read()
    c2 = re.sub(
        r"find_package\(Boost\s+[0-9.]+\s+REQUIRED",
        "find_package(Boost",
        c,
    )
    c2 = re.sub(
        r"find_package\(Boost\s+REQUIRED",
        "find_package(Boost",
        c2,
    )
    c2 = re.sub(
        r"find_package\(Boost\s+[0-9.]+",
        "find_package(Boost",
        c2,
    )
    if c2 != c:
        with open(cml, "w") as f:
            f.write(c2)

# Patch 3: drop util/stream/ from kenlm_util (it #includes boost/thread)
util_cml = f"{KENLM_SRC}/util/CMakeLists.txt"
with open(util_cml, "r") as f:
    c = f.read()
c = c.replace(
    "add_subdirectory(stream)\n",
    "# add_subdirectory(stream)  # excluded for inference-only build\n",
)
c = c.replace("${KENLM_UTIL_STREAM_SOURCE}", "")
with open(util_cml, "w") as f:
    f.write(c)

# Drop lm/common/ from kenlm (and the builder/filter/interpolate subdirs
# + CLI exe targets that depend on them).
lm_cml = f"{KENLM_SRC}/lm/CMakeLists.txt"
with open(lm_cml, "r") as f:
    c = f.read()
c = c.replace(
    "add_subdirectory(common)\n",
    "# add_subdirectory(common)  # excluded for inference-only build\n",
)
c = c.replace("${KENLM_LM_COMMON_SOURCE}", "")
c = c.replace("add_subdirectory(builder)\n", "# add_subdirectory(builder)\n")
c = c.replace("add_subdirectory(filter)\n", "# add_subdirectory(filter)\n")
c = c.replace("add_subdirectory(interpolate)\n", "# add_subdirectory(interpolate)\n")
c = c.replace(
    "AddExes(EXES ${EXE_LIST}\n        LIBRARIES ${LM_LIBS})",
    "# AddExes ... (CLI tools require Boost::program_options; omitted)",
)
with open(lm_cml, "w") as f:
    f.write(c)
print("[+] patched KenLM CMakeLists for inference-only build")


# ---------------------------------------------------------------------------
# 4. Fake Boost include directory (CMake refuses NOTFOUND as a path)
# ---------------------------------------------------------------------------
FAKE_BOOST = "/kaggle/working/fake-boost-include"
os.makedirs(f"{FAKE_BOOST}/boost", exist_ok=True)
with open(f"{FAKE_BOOST}/boost/version.hpp", "w") as f:
    # Report Boost 1.81 to satisfy any residual FindBoost version probing.
    f.write("#define BOOST_VERSION 108100\n")
    f.write('#define BOOST_LIB_VERSION "1_81"\n')


# ---------------------------------------------------------------------------
# 5. Configure + build
# ---------------------------------------------------------------------------
BUILD = f"{WORK}/build-arm64"
if os.path.exists(BUILD):
    shutil.rmtree(BUILD)
os.makedirs(BUILD, exist_ok=True)

print("[+] CMake configure ...")
subprocess.check_call([
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
    f"-DBoost_INCLUDE_DIR={FAKE_BOOST}",
    "-DBoost_LIBRARIES=",
])

print("[+] Building kenlm + kenlm_util ...")
subprocess.check_call([
    "cmake", "--build", BUILD,
    "--target", "kenlm", "kenlm_util",
    "--config", "Release",
    "--parallel", str(os.cpu_count() or 4),
])


# ---------------------------------------------------------------------------
# 6. Collect outputs + headers
# ---------------------------------------------------------------------------
OUT = f"{WORK}/android-arm64"
os.makedirs(f"{OUT}/include/lm", exist_ok=True)
os.makedirs(f"{OUT}/include/util", exist_ok=True)

for fname in ("libkenlm.a", "libkenlm_util.a"):
    shutil.copy(f"{BUILD}/lib/{fname}", f"{OUT}/{fname}")
    print(f"  {fname}: {os.path.getsize(f'{OUT}/{fname}') / 1e6:.2f} MB")

hdr_count = 0
for sub in ("lm", "util"):
    src_dir = f"{KENLM_SRC}/{sub}"
    dst_dir = f"{OUT}/include/{sub}"
    for root, _, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        for f in files:
            if f.endswith(".hh") or f.endswith(".h"):
                d = os.path.join(dst_dir, rel) if rel != "." else dst_dir
                os.makedirs(d, exist_ok=True)
                shutil.copy(os.path.join(root, f), os.path.join(d, f))
                hdr_count += 1
print(f"  {hdr_count} headers")


# ---------------------------------------------------------------------------
# 7. README + upload
# ---------------------------------------------------------------------------
readme = (
    "# KenLM Android prebuilt -- arm64-v8a\n\n"
    "Static libraries + headers cross-compiled for Android, suitable for\n"
    "linking into Liperty's `libliperty_cv.so` via\n"
    "`app/src/main/cpp/CMakeLists.txt`.\n\n"
    "## Contents\n\n"
    "| Path | Purpose |\n"
    "|---|---|\n"
    "| `libkenlm.a`        | `lm::ngram::Model` and inference code |\n"
    "| `libkenlm_util.a`   | KenLM's util sub-library |\n"
    "| `include/lm/*.hh`   | Public inference API headers |\n"
    "| `include/util/*.hh` | Util headers |\n\n"
    "## Build configuration\n\n"
    f"- Android NDK {NDK_VERSION} (Clang 17)\n"
    "- ABI: arm64-v8a, minSdk: android-26\n"
    "- KENLM_MAX_ORDER: 6\n"
    "- Inference-only build (excluded: `util/stream/`, `lm/common/`,\n"
    "  `lm/builder/`, `lm/filter/`, `lm/interpolate/`, all CLI exes)\n"
    "- Boost: not required (the inference targets don't link it)\n\n"
    "Built by `tools/kaggle_build_kenlm_android.py` in the Liperty repo.\n"
)
with open(f"{OUT}/README.md", "w") as f:
    f.write(readme)

print("[+] Uploading to HereLiesAz/liperty-lm/android-arm64 ...")
from huggingface_hub import login, create_repo, upload_folder
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
create_repo("HereLiesAz/liperty-lm", repo_type="model", private=False, exist_ok=True)
upload_folder(
    folder_path=OUT,
    path_in_repo="android-arm64",
    repo_id="HereLiesAz/liperty-lm",
    repo_type="model",
    commit_message=f"KenLM Android arm64-v8a prebuilt (NDK {NDK_VERSION}, inference-only)",
)
print("Uploaded -> https://huggingface.co/HereLiesAz/liperty-lm/tree/main/android-arm64")
