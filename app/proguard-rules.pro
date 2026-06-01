# Add project-specific ProGuard rules here.

# Keep MediaPipe and LiteRT classes
-keep class com.google.mediapipe.** { *; }
-keep class com.google.ai.edge.litert.** { *; }
-keep class com.google.ai.edge.** { *; }

# Keep ONNX Runtime.
# CRITICAL: the Java classes are in package `ai.onnxruntime` — `com.microsoft.onnxruntime`
# is only the Maven group and matches NO classes. Many ai.onnxruntime members (e.g.
# NodeInfo's (String, ValueInfo) constructor) are invoked ONLY from native JNI
# (libonnxruntime4j_jni.so), which R8's static analysis can't see, so without this
# keep R8 strips them and the native lib crashes with NoSuchMethodError on the first
# OrtSession.getInputInfo() call (force-close on model init in minified release builds).
-keep class ai.onnxruntime.** { *; }
-keepclassmembers class ai.onnxruntime.** { *; }
-keep class com.microsoft.onnxruntime.** { *; }

# Keep OpenCV classes
-keep class org.opencv.** { *; }

# Suppress warnings from missing dependencies
-dontwarn com.google.mediapipe.**
-dontwarn com.google.ai.edge.**
-dontwarn org.tensorflow.lite.**
-dontwarn ai.onnxruntime.**
-dontwarn com.microsoft.onnxruntime.**
-dontwarn org.opencv.**

# Keep everything in our app package
-keep class com.hereliesaz.liperty.** { *; }

# WorkManager and Room fixes for R8
-keep class androidx.work.impl.WorkDatabase_Impl { *; }
-keep class androidx.work.impl.WorkDatabase { *; }
-keep class * extends androidx.room.RoomDatabase { *; }
-keep class androidx.room.MultiInstanceInvalidationService { *; }

# Support generic JNI
-keepclasseswithmembernames class * {
    native <methods>;
}
