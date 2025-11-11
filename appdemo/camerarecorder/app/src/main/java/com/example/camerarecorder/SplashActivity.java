package com.example.camerarecorder;


import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;

import androidx.appcompat.app.AppCompatActivity;

public class SplashActivity extends AppCompatActivity {

    // 开屏页显示时间（毫秒）
    private static final int SPLASH_DELAY = 2000;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_splash); // 设置布局文件

        // 使用Handler延迟跳转到主界面
        new Handler(Looper.getMainLooper()).postDelayed(new Runnable() {
            @Override
            public void run() {
                // 创建跳转到主Activity的Intent
                Intent mainIntent = new Intent(SplashActivity.this, MainActivity.class);
                startActivity(mainIntent);
                finish(); // 结束当前SplashActivity，避免返回时再次看到开屏页
            }
        }, SPLASH_DELAY);
    }

    @Override
    protected void onPause() {
        super.onPause();
        // 可选：在页面暂停时移除所有回调，防止内存泄漏
        // 但finish()通常已经处理了这个问题
    }
}