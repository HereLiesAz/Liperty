# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.
cmake_minimum_required(VERSION 3.22.1)

# Declares and names the project.
project("liperty_cv")

# Set the OpenCV directory manually to where setup_libs.sh installs it
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/libs/opencv/sdk/native/jni")

# Find the OpenCV library
find_package(OpenCV REQUIRED)

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
             LC3Wrapper.cpp )

# Searches for prebuilt libraries.
find_library(log-lib log)
find_library(jnigraphics-lib jnigraphics)

# Specifies libraries CMake should link to your target library.
target_link_libraries( # Specifies the target library.
                       liperty_cv

                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib}
                       ${jnigraphics-lib}
                       aaudio
                       ${OpenCV_LIBS} )
