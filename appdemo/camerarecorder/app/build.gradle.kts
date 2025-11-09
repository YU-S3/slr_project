// Android应用构建脚本配置文件
plugins {
    // 应用Android应用程序插件，用于构建Android应用
    alias(libs.plugins.android.application)
}

android {
    // 应用程序的命名空间，用于避免类名冲突
    namespace = "com.example.camerarecorder"
    // 编译SDK版本，指定使用API级别36进行编译
    compileSdk = 36

    // 默认配置块，定义应用的基本属性
    defaultConfig {
        // 应用程序ID，唯一标识应用
        applicationId = "com.example.camerarecorder"
        // 最低支持的Android版本，API级别24（Android 7.0）
        minSdk = 24
        // 目标SDK版本，API级别36
        targetSdk = 36
        // 版本号，用于应用更新
        versionCode = 1
        // 版本名称，用户可见的版本标识
        versionName = "1.0"

        // 测试框架的运行器配置
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    // 构建类型配置块
    buildTypes {
        // 发布版本配置
        release {
            // 是否启用代码混淆，false表示不启用
            isMinifyEnabled = false
            // 混淆规则文件配置
            proguardFiles(
                // 默认的混淆规则文件
                getDefaultProguardFile("proguard-android-optimize.txt"),
                // 项目特定的混淆规则文件
                "proguard-rules.pro"
            )
        }
    }
    
    // Java编译选项配置
    compileOptions {
        // 源代码兼容性版本设置为Java 11
        sourceCompatibility = JavaVersion.VERSION_11
        // 目标编译版本设置为Java 11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

// 依赖项配置块
dependencies {
    // AppCompat库，提供向后兼容的Android组件
    implementation("androidx.appcompat:appcompat:1.6.1")
    // ConstraintLayout约束布局库，用于构建灵活的UI布局
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    // Android核心库，提供兼容性支持
    implementation("androidx.core:core:1.12.0")

    // 通过libs.versions.toml文件引用的AppCompat库
    implementation(libs.appcompat)
    // 通过libs.versions.toml文件引用的Material Design组件库
    implementation(libs.material)
    // JUnit测试框架依赖
    testImplementation(libs.junit)
    // Android扩展JUnit测试库
    androidTestImplementation(libs.ext.junit)
    // Espresso UI测试框架
    androidTestImplementation(libs.espresso.core)
    // Google Material Design组件库（已注释）
    //implementation("com.google.android.material:material:1.9.0")
}
