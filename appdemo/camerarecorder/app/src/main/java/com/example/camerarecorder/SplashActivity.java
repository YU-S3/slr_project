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
        setContentView(R.layout.activity_splash);

        // 使用Handler延迟跳转到选择界面
        new Handler(Looper.getMainLooper()).postDelayed(new Runnable() {
            @Override
            public void run() {
                // 创建跳转到选择界面的Intent
                Intent selectionIntent = new Intent(SplashActivity.this, SelectionActivity.class);
                startActivity(selectionIntent);
                finish();
            }
        }, SPLASH_DELAY);
    }
}
