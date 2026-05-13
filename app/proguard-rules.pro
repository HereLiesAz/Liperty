# Add project-specific ProGuard rules here.

# Keep MediaPipe and LiteRT classes
-keep class com.google.mediapipe.** { *; }
-keep class com.google.ai.edge.litert.** { *; }
-keep class com.google.ai.edge.** { *; }

# Keep ONNX Runtime
-keep class com.microsoft.onnxruntime.** { *; }

# Keep OpenCV classes
-keep class org.opencv.** { *; }

# Suppress warnings from missing dependencies
-dontwarn com.google.mediapipe.**
-dontwarn com.google.ai.edge.**
-dontwarn org.tensorflow.lite.**
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
