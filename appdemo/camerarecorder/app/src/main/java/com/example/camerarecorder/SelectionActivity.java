// app/src/main/java/com/example/camerarecorder/SelectionActivity.java
package com.example.camerarecorder;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import androidx.appcompat.app.AppCompatActivity;

public class SelectionActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_selection);

        Button btnOption1 = findViewById(R.id.btn_option1);
        Button btnOption2 = findViewById(R.id.btn_option2);

        btnOption1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // 添加点击时的视觉反馈
                v.setPressed(true);
                // 延迟一小段时间再执行跳转，让用户看到点击效果
                v.postDelayed(new Runnable() {
                    @Override
                    public void run() {
                        // 点击第一个选项跳转到 MainActivity
                        Intent intent = new Intent(SelectionActivity.this, MainActivity.class);
                        startActivity(intent);
                        finish();
                    }
                }, 100);
            }
        });

        btnOption2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // 添加点击时的视觉反馈
                v.setPressed(true);
                // 延迟一小段时间再执行操作
                v.postDelayed(new Runnable() {
                    @Override
                    public void run() {
                        // 这里可以添加第二个选项的功能
                        // 示例：跳转到其他Activity或执行其他操作
                        v.setPressed(false);
                    }
                }, 100);
            }
        });
    }
}
