package com.example.camerarecorder;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.view.TextureView;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_CAMERA_PERMISSION = 100;
    private TextureView textureView;
    private Button btnRecord, btnSwitch;
    private TextView tvStatus;
    private CameraHelper cameraHelper;
    private boolean isRecording = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        setTitle("摄像头录像器");
        initViews();
        checkPermissions();
    }

    private void initViews() {
        textureView = findViewById(R.id.textureView);
        btnRecord = findViewById(R.id.btnRecord);
        btnSwitch = findViewById(R.id.btnSwitch);
        tvStatus = findViewById(R.id.tvStatus);

        btnRecord.setOnClickListener(v -> toggleRecording());
        btnSwitch.setOnClickListener(v -> switchCamera());

        // 初始禁用按钮，等待权限
        btnRecord.setEnabled(false);
        btnSwitch.setEnabled(false);
    }

    private void checkPermissions() {
        // 基础权限：摄像头和录音
        String[] basePermissions = {
                Manifest.permission.CAMERA,
                Manifest.permission.RECORD_AUDIO
        };

        // 存储权限根据Android版本处理
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            // Android 13+ 使用新的媒体权限
            requestPermissionsWithMediaAccess();
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            // Android 10-12: 使用应用专属存储，不需要额外权限
            requestBasePermissions(basePermissions);
        } else {
            // Android 9及以下：需要READ_EXTERNAL_STORAGE
            String[] legacyPermissions = {
                    Manifest.permission.CAMERA,
                    Manifest.permission.RECORD_AUDIO,
                    Manifest.permission.READ_EXTERNAL_STORAGE
            };
            requestBasePermissions(legacyPermissions);
        }
    }

    private void requestPermissionsWithMediaAccess() {
        String[] permissions;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            // Android 14+: 需要READ_MEDIA_VIDEO和READ_MEDIA_IMAGES
            permissions = new String[]{
                    Manifest.permission.CAMERA,
                    Manifest.permission.RECORD_AUDIO,
                    Manifest.permission.READ_MEDIA_VIDEO,
                    Manifest.permission.READ_MEDIA_IMAGES
            };
        } else {
            // Android 13: 需要READ_MEDIA_VIDEO
            permissions = new String[]{
                    Manifest.permission.CAMERA,
                    Manifest.permission.RECORD_AUDIO,
                    Manifest.permission.READ_MEDIA_VIDEO
            };
        }
        requestBasePermissions(permissions);
    }

    private void requestBasePermissions(String[] permissions) {
        boolean allGranted = true;
        for (String permission : permissions) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                allGranted = false;
                break;
            }
        }

        if (allGranted) {
            initCamera();
        } else {
            ActivityCompat.requestPermissions(this, permissions, REQUEST_CAMERA_PERMISSION);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            boolean allGranted = true;
            for (int result : grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    allGranted = false;
                    break;
                }
            }

            if (allGranted) {
                initCamera();
            } else {
                handlePermissionDenied();
            }
        }
    }

    private void handlePermissionDenied() {
        Toast.makeText(this, "需要权限才能使用摄像头和录像功能", Toast.LENGTH_LONG).show();
        tvStatus.setText("权限被拒绝，无法使用摄像头");
        btnRecord.setEnabled(false);
        btnSwitch.setEnabled(false);

        // 可以提供设置引导
        showPermissionGuide();
    }

    private void showPermissionGuide() {
        new AlertDialog.Builder(this)
                .setTitle("需要权限")
                .setMessage("应用需要摄像头、麦克风和存储权限才能正常录像。请在设置中授予权限。")
                .setPositiveButton("去设置", (dialog, which) -> openAppSettings())
                .setNegativeButton("取消", null)
                .show();
    }

    private void openAppSettings() {
        Intent intent = new Intent(android.provider.Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
        intent.setData(android.net.Uri.parse("package:" + getPackageName()));
        startActivity(intent);
    }


    private void initCamera() {
        cameraHelper = new CameraHelper(this, textureView);
        cameraHelper.setRecordingListener(new CameraHelper.RecordingListener() {
            @Override
            public void onRecordingStarted() {
                runOnUiThread(() -> {
                    tvStatus.setText("录制中...");
                    btnRecord.setText("停止录制");
                    btnSwitch.setEnabled(false);
                    isRecording = true;
                });
            }

            @Override
            public void onRecordingStopped(String filePath) {
                runOnUiThread(() -> {
                    tvStatus.setText("录制完成");
                    btnRecord.setText("开始录制");
                    btnSwitch.setEnabled(true);
                    isRecording = false;
                    Toast.makeText(MainActivity.this, "视频已保存", Toast.LENGTH_SHORT).show();
                });
            }

            @Override
            public void onError(String error) {
                runOnUiThread(() -> {
                    tvStatus.setText("错误: " + error);
                    btnRecord.setText("开始录制");
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

        btnRecord.setEnabled(true);
        btnSwitch.setEnabled(true);
        tvStatus.setText("准备就绪");
    }

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

    private void switchCamera() {
        if (cameraHelper != null) {
            cameraHelper.switchCamera();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (cameraHelper != null) {
            cameraHelper.startBackgroundThread();
            if (textureView.isAvailable()) {
                cameraHelper.openCamera();
            } else {
                textureView.setSurfaceTextureListener(cameraHelper.getSurfaceTextureListener());
            }
        }
    }

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

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraHelper != null) {
            cameraHelper.closeCamera();
            cameraHelper.stopBackgroundThread();
        }
    }

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