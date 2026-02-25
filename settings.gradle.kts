pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
plugins {
    id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        maven { url = uri("https://jitpack.io") }
    }
}

rootProject.name = "Liperty"
include(":app")

// Safely include OpenCV module if the SDK has been downloaded
val opencvSdkPath = file("app/src/main/cpp/libs/opencv/sdk")
if (opencvSdkPath.exists()) {
    include(":opencv")
    project(":opencv").projectDir = opencvSdkPath
} else {
    println("WARNING: OpenCV SDK not found at ${opencvSdkPath}. Run setup_libs.sh to install.")
}
