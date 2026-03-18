plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.hereliesaz.liperty"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.hereliesaz.liperty"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary = true
        }

        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
                arguments += "-DANDROID_STL=c++_shared"
            }
        }

        ndk {
            // Filter relevant ABIs
            abiFilters.addAll(listOf("armeabi-v7a", "arm64-v8a", "x86_64"))
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlin {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        }
    }
    sourceSets {
        getByName("test") {
            assets.srcDirs("src/main/assets")
        }
    }
    buildFeatures {
        viewBinding = false
        compose = true
    }
    testOptions {
        unitTests {
            isIncludeAndroidResources = true
        }
    }
    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
}

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.google.material)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.lifecycle.runtime.ktx)

    // Compose
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    implementation(libs.androidx.material.icons.extended)
    implementation(libs.androidx.navigation.compose)
    implementation(libs.androidx.graphics.core)
    implementation(libs.onnxruntime.android)
    debugImplementation(libs.androidx.ui.tooling)

    // AzNavRail
    implementation(libs.aznavrail)

    // CameraX
    implementation(libs.androidx.camera.core)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.video)
    implementation(libs.androidx.camera.view)
    implementation(libs.androidx.camera.extensions)

    // MediaPipe (Face Mesh)
    implementation(libs.mediapipe.tasks.vision)

    // LiteRT (formerly TensorFlow Lite)
    implementation(libs.tflite)
    implementation(libs.tfliteGpu) {
        exclude(group = "com.google.ai.edge.litert", module = "litert-api")
    }
    implementation(libs.tfliteSupport) {
        exclude(group = "com.google.ai.edge.litert", module = "litert-api")
    }
    implementation(libs.tfliteSelectTfOps)
    testImplementation(libs.tflite)
    testImplementation(libs.tfliteGpu)
    testImplementation(libs.tfliteSupport)
    testImplementation(libs.tfliteSelectTfOps)

    // OpenCV - Only include if the project is available (handled in settings.gradle.kts)
    if (findProject(":opencv") != null) {
        implementation(project(":opencv"))
    } else {
        // Fallback: This allows the project to sync even if setup_libs.sh hasn't run yet,
        // though runtime functionality relying on OpenCV Java classes will obviously fail.
        // NDK linking happens via CMakeLists.txt separately.
        println("WARNING: OpenCV module not found. Skipping dependency.")
    }

    // Coroutines
    implementation(libs.kotlinx.coroutines.android)

    testImplementation(libs.junit)
    testImplementation(libs.androidx.test.core)
    testImplementation(libs.androidx.junit)
    testImplementation(libs.robolectric)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.tfliteSelectTfOps)
}

tasks.withType<Test> {
    testLogging {
        events("passed", "skipped", "failed")
        showStandardStreams = true
    }
}
