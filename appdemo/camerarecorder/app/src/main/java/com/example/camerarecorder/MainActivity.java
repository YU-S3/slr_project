package com.example.camerarecorder;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.os.Handler;
import android.view.TextureView;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;



// 主Activity类，负责管理摄像头录像功能的界面和交互逻辑
public class MainActivity extends AppCompatActivity {

    // 定义请求摄像头权限的请求码
    private static final int REQUEST_CAMERA_PERMISSION = 100;

    // UI组件：纹理视图用于显示相机预览
    private TextureView textureView;

    // UI组件：录制按钮和切换摄像头按钮
    private ImageButton btnRecord, btnSwitch;

    // UI组件：状态文本显示当前操作状态
    private TextView tvStatus;

    // 相机助手类实例，封装了具体的相机操作逻辑
    private CameraHelper cameraHelper;

    // 记录当前是否正在录制的状态标志
    private boolean isRecording = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main); // 设置布局文件

        setTitle("摄像头录像器"); // 设置标题栏文字
        initViews(); // 初始化界面控件
        checkPermissions(); // 检查并申请必要权限
    }

    // 初始化界面上的所有控件，并设置点击事件监听器
    private void initViews() {
        textureView = findViewById(R.id.textureView);
        btnRecord = findViewById(R.id.btnRecord);
        btnSwitch = findViewById(R.id.btnSwitch);
        tvStatus = findViewById(R.id.tvStatus);

        // 设置录制按钮初始状态为未激活（灰色）
        btnRecord.setImageResource(R.drawable.ic_record_inactive_large);

        // 录制按钮点击事件处理
        btnRecord.setOnClickListener(v -> toggleRecording());

        // 切换摄像头按钮点击事件处理
        btnSwitch.setOnClickListener(v -> switchCamera());

        // 初始状态下禁用按钮，直到获取到所需权限后再启用
        btnRecord.setEnabled(false);
        btnSwitch.setEnabled(false);
    }

    // 检查所需的运行时权限
    private void checkPermissions() {
        // 基础权限包括摄像头和录音权限
        String[] basePermissions = {
                Manifest.permission.CAMERA,
                Manifest.permission.RECORD_AUDIO
        };

        // 根据不同Android版本处理存储权限需求
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            // Android 13及以上版本需要特定媒体访问权限
            requestPermissionsWithMediaAccess();
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            // Android 10-12使用应用专属存储，无需额外权限
            requestBasePermissions(basePermissions);
        } else {
            // Android 9及以下需要读取外部存储权限
            String[] legacyPermissions = {
                    Manifest.permission.CAMERA,
                    Manifest.permission.RECORD_AUDIO,
                    Manifest.permission.READ_EXTERNAL_STORAGE
            };
            requestBasePermissions(legacyPermissions);
        }
    }

    // 请求与媒体访问相关的权限（适用于Android 13及以上）
    private void requestPermissionsWithMediaAccess() {
        String[] permissions;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            // Android 14以上需要READ_MEDIA_VIDEO和READ_MEDIA_IMAGES权限
            permissions = new String[]{
                    Manifest.permission.CAMERA,
                    Manifest.permission.RECORD_AUDIO,
                    Manifest.permission.READ_MEDIA_VIDEO,
                    Manifest.permission.READ_MEDIA_IMAGES
            };
        } else {
            // Android 13只需要READ_MEDIA_VIDEO权限
            permissions = new String[]{
                    Manifest.permission.CAMERA,
                    Manifest.permission.RECORD_AUDIO,
                    Manifest.permission.READ_MEDIA_VIDEO
            };
        }
        requestBasePermissions(permissions);
    }

    // 请求基础权限的方法
    private void requestBasePermissions(String[] permissions) {
        boolean allGranted = true;
        // 检查每个权限是否已被授予
        for (String permission : permissions) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                allGranted = false;
                break;
            }
        }

        if (allGranted) {
            initCamera(); // 如果所有权限都已获得，则初始化相机
        } else {
            // 否则请求缺失的权限
            ActivityCompat.requestPermissions(this, permissions, REQUEST_CAMERA_PERMISSION);
        }
    }

    // 处理权限请求结果回调
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            boolean allGranted = true;
            // 检查所有权限是否都被允许
            for (int result : grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    allGranted = false;
                    break;
                }
            }

            if (allGranted) {
                initCamera(); // 权限全部获得后初始化相机
            } else {
                handlePermissionDenied(); // 处理权限被拒绝的情况
            }
        }
    }

    // 当权限被拒绝时执行的操作
    private void handlePermissionDenied() {
        Toast.makeText(this, "需要权限才能使用摄像头和录像功能", Toast.LENGTH_LONG).show();
        tvStatus.setText("权限被拒绝，无法使用摄像头");
        btnRecord.setEnabled(false);
        btnSwitch.setEnabled(false);

        // 显示引导用户前往设置开启权限的对话框
        showPermissionGuide();
    }

    // 展示一个提示对话框指导用户如何手动授权
    private void showPermissionGuide() {
        new AlertDialog.Builder(this)
                .setTitle("需要权限")
                .setMessage("应用需要摄像头、麦克风和存储权限才能正常录像。请在设置中授予权限。")
                .setPositiveButton("去设置", (dialog, which) -> openAppSettings())
                .setNegativeButton("取消", null)
                .show();
    }

    // 打开本应用程序的系统设置页面
    private void openAppSettings() {
        Intent intent = new Intent(android.provider.Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
        intent.setData(android.net.Uri.parse("package:" + getPackageName()));
        startActivity(intent);
    }


    // 初始化相机相关资源
    private void initCamera() {
        cameraHelper = new CameraHelper(this, textureView);

        // 设置录制状态变化的监听器
        cameraHelper.setRecordingListener(new CameraHelper.RecordingListener() {
            @Override
            public void onRecordingStarted() {
                runOnUiThread(() -> {
                    tvStatus.setText("识别中");
                    // 设置录制按钮为激活状态（红色）
                    btnRecord.setImageResource(R.drawable.ic_record_active_large);
                    btnSwitch.setEnabled(false);
                    isRecording = true;
                });
            }

            @Override
            public void onRecordingStopped(String filePath) {
                runOnUiThread(() -> {
                    tvStatus.setText("未识别");
                    // 设置录制按钮为未激活状态（灰色）
                    btnRecord.setImageResource(R.drawable.ic_record_inactive_large);
                    btnSwitch.setEnabled(true);
                    isRecording = false;
                    Toast.makeText(MainActivity.this, "视频已保存", Toast.LENGTH_SHORT).show();
                });
            }

            @Override
            public void onError(String error) {
                runOnUiThread(() -> {
                    tvStatus.setText("错误: " + error);
                    // 设置录制按钮为未激活状态（灰色）
                    btnRecord.setImageResource(R.drawable.ic_record_inactive_large);
                    btnSwitch.setEnabled(true);
                    isRecording = false;

                    // 显示详细的错误对话框
                    if (error.contains("摄像头") || error.contains("录制")) {
                        showCameraErrorDialog(error);
                    } else {
                        Toast.makeText(MainActivity.this, error, Toast.LENGTH_LONG).show();
                    }
                });
            }
        });

        // 启用控制按钮并更新状态信息
        btnRecord.setEnabled(true);
        btnSwitch.setEnabled(true);
        // 设置录制按钮初始状态为未激活（灰色）
        btnRecord.setImageResource(R.drawable.ic_record_inactive_large);
        tvStatus.setText("未识别");
    }

    // 控制录制开始或结束
    private void toggleRecording() {
        if (cameraHelper == null) return;

        if (!isRecording) {
            if (cameraHelper.startRecording()) {
                isRecording = true;
            }
        } else {
            cameraHelper.stopRecording();
            isRecording = false;
        }
    }

    // 切换前后摄像头
    private void switchCamera() {
        if (cameraHelper != null) {
            cameraHelper.switchCamera();
        }
    }

    // 启用沉浸式模式，隐藏状态栏和导航栏
    private void enableImmersiveMode() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
            getWindow().getDecorView().setSystemUiVisibility(
                    View.SYSTEM_UI_FLAG_FULLSCREEN
                            | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                            | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                            | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                            | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                            | View.SYSTEM_UI_FLAG_LAYOUT_STABLE
            );
        }
    }

    // 禁用沉浸式模式，显示状态栏和导航栏
    private void disableImmersiveMode() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
            getWindow().getDecorView().setSystemUiVisibility(
                    View.SYSTEM_UI_FLAG_LAYOUT_STABLE
            );
        }
    }

    // Activity恢复可见时调用，启动后台线程并打开相机
    @Override
    protected void onResume() {
        super.onResume();
        // 启用沉浸式模式以隐藏导航栏
        enableImmersiveMode();
        
        if (cameraHelper != null) {
            cameraHelper.startBackgroundThread();
            if (textureView.isAvailable()) {
                cameraHelper.openCamera();
            } else {
                textureView.setSurfaceTextureListener(cameraHelper.getSurfaceTextureListener());
            }
        }
    }

    // Activity暂停时调用，关闭相机和释放资源
    @Override
    protected void onPause() {
        super.onPause();
        if (cameraHelper != null) {
            if (isRecording) {
                cameraHelper.stopRecording();
                isRecording = false;
            }
            cameraHelper.closeCamera();
            cameraHelper.stopBackgroundThread();
        }
    }

    // Activity销毁时调用，彻底清理资源
    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraHelper != null) {
            cameraHelper.closeCamera();
            cameraHelper.stopBackgroundThread();
        }
    }

    // 提供重试机制，在摄像头出错时让用户尝试重新连接
    private void retryCameraOpen() {
        new AlertDialog.Builder(this)
                .setTitle("摄像头错误")
                .setMessage("摄像头出现错误，是否重试？")
                .setPositiveButton("重试", (dialog, which) -> {
                    if (cameraHelper != null) {
                        // 先完全关闭再重新打开
                        cameraHelper.closeCamera();
                        // 延迟一段时间再重试
                        new Handler().postDelayed(() -> {
                            cameraHelper.openCamera();
                        }, 1000);
                    }
                })
                .setNegativeButton("取消", null)
                .show();
    }

    // 显示关于摄像头错误的具体信息及解决建议
    private void showCameraErrorDialog(String error) {
        runOnUiThread(() -> {
            new AlertDialog.Builder(this)
                    .setTitle("摄像头错误")
                    .setMessage(error + "\n\n建议解决方案：\n1. 关闭其他使用摄像头的应用\n2. 重启手机\n3. 检查摄像头权限")
                    .setPositiveButton("重试", (dialog, which) -> retryCameraOpen())
                    .setNegativeButton("确定", null)
                    .show();
        });
    }
}