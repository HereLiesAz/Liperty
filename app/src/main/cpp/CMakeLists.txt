# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.
cmake_minimum_required(VERSION 3.22.1)

# Declares and names the project.
project("liperty_cv")

# Set the OpenCV directory manually to where setup_libs.sh installs it
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/libs/opencv/sdk/native/jni")

# If we are building in Debug but only Release prebuilts exist, map them.
set(CMAKE_MAP_IMPORTED_CONFIG_DEBUG Release)

# Find the OpenCV library
find_package(OpenCV REQUIRED)

# OpenCV 4.10+ static libraries for arm64-v8a depend on KleidiCV for HAL
# acceleration. The OpenCV CMake config defines these as imported targets
# but might not link them correctly in all build configurations (e.g. Debug
# builds when only Release prebuilts exist).
if(ANDROID_ABI STREQUAL "arm64-v8a")
    message(STATUS "Linking KleidiCV and 3rdparty dependencies for arm64-v8a static OpenCV")
    list(APPEND OpenCV_LIBS kleidicv_hal kleidicv kleidicv_thread tegra_hal tbb libcpufeatures ittnotify)
endif()

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.
add_library( # Sets the name of the library.
             liperty_cv

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             image_utils_jni.cpp
             audio_player.cpp
             vibraphone_jni.cpp
             AudioEngine.cpp
             LC3Wrapper.cpp
             # KenLM JNI bridge -- guarded by KENLM_AVAILABLE below.
             # Falls back to a stub when arm64-v8a prebuilts are absent.
             kenlm_jni.cpp )

# 16 KB page-size support for Android 15+. The dynamic linker on devices
# with 16 KiB pages rejects shared libraries whose PT_LOAD segments are
# only aligned to 4 KiB. The NDK toolchain's default is 4 KiB for
# backwards compat, so we have to opt in explicitly. Safe on 4 KiB
# devices too -- the larger alignment just wastes at most 12 KiB of
# virtual address space per segment, which doesn't matter.
target_link_options(liperty_cv PRIVATE
    "-Wl,-z,max-page-size=16384"
    "-Wl,-z,common-page-size=16384")

# Searches for prebuilt libraries.
find_library(log-lib log)
find_library(jnigraphics-lib jnigraphics)

# KenLM Android prebuilts (cross-compiled by
# tools/kaggle_build_kenlm_android.py, pulled by setup_libs.sh).
# When the .a files are present, the JNI bridge compiles against the
# real lm::ngram::Model API. When absent (CI / fresh checkout before
# setup_libs.sh runs), kenlm_jni.cpp falls back to its stub which
# reports load failure and the Kotlin layer no-ops cleanly.
set(KENLM_NDK_DIR ${CMAKE_SOURCE_DIR}/kenlm/android-arm64)
# Prebuilts are cross-compiled for arm64-v8a only (see
# tools/kaggle_build_kenlm_android.py). armeabi-v7a / x86_64 builds
# fall through to the stub -- KenLmScorer.isNativeLoaded then reports
# false on those devices and the Kotlin layer no-ops. The Pixel 5 and
# all modern Android devices are arm64, so this only matters for
# legacy 32-bit hardware.
if(ANDROID_ABI STREQUAL "arm64-v8a"
   AND EXISTS ${KENLM_NDK_DIR}/libkenlm.a
   AND EXISTS ${KENLM_NDK_DIR}/libkenlm_util.a)
    message(STATUS "KenLM NDK prebuilts found (arm64-v8a): enabling JNI bridge")
    add_library(kenlm STATIC IMPORTED)
    set_target_properties(kenlm PROPERTIES
        IMPORTED_LOCATION ${KENLM_NDK_DIR}/libkenlm.a)
    add_library(kenlm_util STATIC IMPORTED)
    set_target_properties(kenlm_util PROPERTIES
        IMPORTED_LOCATION ${KENLM_NDK_DIR}/libkenlm_util.a)
    target_include_directories(liperty_cv PRIVATE ${KENLM_NDK_DIR}/include)
    target_compile_definitions(liperty_cv PRIVATE KENLM_AVAILABLE=1)
    target_link_libraries(liperty_cv kenlm kenlm_util z)
else()
    message(STATUS "KenLM JNI: stub (ABI=${ANDROID_ABI}, prebuilt dir=${KENLM_NDK_DIR})")
endif()

# Specifies libraries CMake should link to your target library.
target_link_libraries( # Specifies the target library.
                       liperty_cv

                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib}
                       ${jnigraphics-lib}
                       aaudio
                       ${OpenCV_LIBS} )
