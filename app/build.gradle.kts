plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

val versionProps = providers.fileContents(layout.projectDirectory.file("../version.properties"))
    .asText.get().lines().associate {
        val parts = it.split("=", limit = 2)
        if (parts.size == 2) parts[0].trim() to parts[1].trim() else "" to ""
    }
val vMajor = versionProps["versionMajor"]?.toIntOrNull() ?: 0
val vMinor = versionProps["versionMinor"]?.toIntOrNull() ?: 1
val vPatch = versionProps["versionPatch"]?.toIntOrNull() ?: 0
val vBuild = (project.findProperty("versionBuild") as? String)?.toIntOrNull() ?: 0

android {
    namespace = "com.hereliesaz.liperty"
    compileSdk = 37

    defaultConfig {
        applicationId = "com.hereliesaz.liperty"
        minSdk = 26
        targetSdk = 37
        versionCode = vMajor * 10000 + vMinor * 100 + vPatch
        versionName = "$vMajor.$vMinor.$vPatch.$vBuild"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary = true
        }

        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
                arguments += "-DANDROID_STL=c++_shared"
                // 16 KB page-size support (required for Android 15+ on
                // arm64 devices with 16 KB pages). The linker must align
                // PT_LOAD segments to 16 KiB instead of the historical
                // 4 KiB, otherwise the dynamic loader refuses libliperty_cv.so
                // on such devices. See:
                // https://developer.android.com/guide/practices/page-sizes
                arguments += "-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON"
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
            // Use the debug signing config for release builds to unblock
            // local performance testing and 'bundleRelease' runs without
            // requiring a production keystore.
            signingConfig = signingConfigs.getByName("debug")
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            // arm64-only for release — covers 95%+ of modern devices.
            // Debug keeps all 3 ABIs for emulator testing.
            ndk {
                abiFilters.clear()
                abiFilters.add("arm64-v8a")
            }
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
        buildConfig = true
    }
    testOptions {
        unitTests {
            isIncludeAndroidResources = true
        }
    }
    // Robolectric mmaps the unit-test APK via a 32-bit ZipFileRO; when the
    // combined assets folder exceeds ~2 GB (post-stage-4 export the SyncVSR
    // + AV-HuBERT + Auto-AVSR ONNX files alone total ~3 GB), parseApkLite
    // throws NegativeArraySizeException and every Robolectric test class
    // fails before its body runs. Strip the giant binary blobs from the
    // unit-test APK only — unit tests never load real model files (they
    // pass fake EncoderSession / DecoderSession etc.); they only need the
    // small text assets (vocab files, viseme_index.json, ...).
    //
    // Hook is on the packaging task itself so we don't need to know which
    // upstream merge task owns the staging directory in AGP 9 — we just
    // walk the task's input file tree and delete matching files.
    // `generateDebugUnitTestAssets` is the merge task in AGP 9 (was
    // `mergeDebugUnitTestAssets` in AGP 7/8); the package task's input
    // points at its output directory either way.
    val unitTestPruneRegex = Regex("""\.(onnx|tflite)$|^librispeech_3gram\.bin$|_landmarker\.task$""")
    val buildDirPath = layout.buildDirectory.get().asFile.absolutePath
    listOf("Debug", "Release").forEach { variantSuffix ->
        tasks.matching { it.name == "package${variantSuffix}UnitTestForUnitTest" }
            .configureEach {
                doFirst {
                    inputs.files.forEach { f ->
                        if (!f.exists()) return@forEach
                        // Hard safety: only delete from build/ intermediates,
                        // never from src/. A misconfigured task input pointing
                        // at src/main/assets would otherwise nuke production
                        // model files on the developer's disk.
                        if (!f.absolutePath.startsWith(buildDirPath)) return@forEach
                        f.walkTopDown().forEach { child ->
                            if (child.isFile && unitTestPruneRegex.containsMatchIn(child.name)) {
                                logger.lifecycle("Stripping ${child.name} from unit-test APK to keep it under 2 GB")
                                child.delete()
                            }
                        }
                    }
                }
            }
    }
    // ── Production asset pruning ─────────────────────────────────────
    // AGP's packaging.resources.excludes does NOT apply to Android assets
    // (only Java classpath resources). We must prune large ML binaries via
    // a task hook on mergeAssets — same pattern as the unit-test hook above.
    // These files are downloaded at runtime by ModelDownloadManager.
    val productionPruneRegex = Regex(
        """\.(onnx|tflite)$|^librispeech_3gram\.bin$|_landmarker\.task$"""
    )
    listOf("Debug", "Release").forEach { variant ->
        tasks.matching { it.name == "merge${variant}Assets" }
            .configureEach {
                doLast {
                    outputs.files.forEach { f ->
                        if (!f.exists()) return@forEach
                        if (!f.absolutePath.startsWith(buildDirPath)) return@forEach
                        f.walkTopDown().forEach { child ->
                            if (child.isFile && productionPruneRegex.containsMatchIn(child.name)) {
                                logger.lifecycle("Pruning ${child.name} from $variant assets (runtime download)")
                                child.delete()
                            }
                        }
                    }
                }
            }
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
            // NOTE: The excludes below are belt-and-suspenders documentation.
            // They do NOT actually work for Android assets in AGP 9 — the
            // productionPruneRegex task hook above is what actually strips them.
            excludes += "**/syncvsr_lrs3_visual_ctc_fp16.onnx"
            excludes += "**/autoavsr_lrs3_visual_ctc.onnx"
            excludes += "**/syncvsr_lrs3_visual_ctc.onnx"
            excludes += "**/syncvsr_lrs3_encoder.onnx"
            excludes += "**/syncvsr_lrs3_decoder.onnx"
            excludes += "**/avhubert_base_vox_433h_visual_encoder.onnx"
            excludes += "**/avhubert_base_vox_433h_decoder.onnx"
            excludes += "**/vallr_model.onnx"
            excludes += "**/vsr_model.tflite"
            excludes += "**/vsr_lora_model.tflite"
            excludes += "**/ssr_model.tflite"
            excludes += "**/tramba_model.tflite"
            excludes += "**/voice_converter.tflite"
            excludes += "**/librispeech_3gram.bin"
            excludes += "**/pocket_tts_base.onnx"
            excludes += "**/pocket_tts_se_extractor.onnx"
            excludes += "**/pocket_tts_tone_converter.onnx"
            excludes += "**/*_landmarker.task"
        }
        // Required for 16 KB page-size devices: native libs must be
        // stored uncompressed in the APK so the dynamic loader can
        // mmap them directly. useLegacyPackaging=false makes the .so
        // entries STORED (not DEFLATED) with the correct alignment.
        //
        // Status of every .so in the APK (verified via llvm-readelf -l):
        //   libliperty_cv.so                    16 KB-aligned (linker flag below)
        //   libLiteRt.so                        16 KB-aligned (LiteRT 2.1.4 ships it)
        //   libonnxruntime4j_jni.so             16 KB-aligned (ORT 1.22 ships it)
        //   libc++_shared.so                    16 KB-aligned (NDK)
        //   libgraphics-core.so                 16 KB-aligned
        //   libopencv_java4.so                   4 KB-aligned (OpenCV 4.10.0 -- upgrade
        //                                       to 4.11+ for 16 KB Android 15 devices)
        //   libmediapipe_tasks_vision_jni.so     4 KB-aligned (MediaPipe Tasks 0.20230731;
        //                                       upgrade to 0.10.21+ for 16 KB)
        jniLibs {
            useLegacyPackaging = false
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    // Redirect C++ build directory outside Google Drive to avoid ninja
    // "build.ninja still dirty" errors caused by cloud sync file locking.
    @Suppress("UnstableApiUsage")
    externalNativeBuild.cmake.buildStagingDirectory =
        file(System.getProperty("java.io.tmpdir") + "/liperty-cxx-build")
}

configurations.all {
    resolutionStrategy.eachDependency {
        if (requested.group == "org.jetbrains.kotlin" && requested.name == "compose-group-mapping") {
            useVersion("2.3.21")
        }
    }
}

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.google.material)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.work.runtime)

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
    // Note: litert-select-tf-ops removed from production — the production
    // backend is ONNX (SyncVSR), and legacy TFLite models requiring flex ops
    // are not shipped in the APK. Saves ~192 MB of native libs.
    implementation(libs.tflite)
    implementation(libs.tfliteGpu) {
        exclude(group = "com.google.ai.edge.litert", module = "litert-api")
    }
    implementation(libs.tfliteSupport) {
        exclude(group = "com.google.ai.edge.litert", module = "litert-api")
    }
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
    // For Dispatchers.setMain + StandardTestDispatcher + advanceUntilIdle,
    // used by VisemeCalibrationViewModelTest (and any future ViewModel
    // test that needs deterministic coroutine scheduling).
    testImplementation(libs.kotlinx.coroutines.test)
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
