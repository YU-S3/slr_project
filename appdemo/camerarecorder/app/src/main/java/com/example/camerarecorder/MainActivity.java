package com.example.camerarecorder;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.util.Size;
import android.view.View;
import android.view.TextureView;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

/**
 * 主 Activity — 手语识别实时翻译界面
 * <p>
 * 改造说明：从"视频录制"改为"实时抽帧 + WebSocket 传输 + 翻译结果展示"
 * <p>
 * 工作流程：
 * 1. 用户点击"开始识别" → 建立 WebSocket 连接 → 开启定时抽帧
 * 2. 每 100ms 从 TextureView 捕获一帧 → 运动检测 → JPEG 压缩 → Base64 → 发送
 * 3. 收到后端返回的翻译结果 → 更新 UI 展示
 * 4. 用户点击"停止识别" → 停止抽帧 → 断开 WebSocket
 */
public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_CAMERA_PERMISSION = 100;

    // UI 组件
    private TextureView textureView;
    private ImageButton btnRecord, btnSwitch, btnMicrophone;
    private TextView tvStatus, tvTranslation;

    // 核心模块
    private CameraHelper cameraHelper;
    private WebSocketClient webSocketClient;
    private FrameCaptureHelper frameCaptureHelper;
    private MotionDetector motionDetector;

    // 状态标志
    private boolean isRecording = false;
    private boolean isMicrophoneOn = false;
    private boolean isConnected = false;
    private int frameIdx = 0;

    // 定时抽帧
    private Handler frameHandler;
    private Runnable frameCaptureRunnable;

    // 调试浮层
    private DebugOverlayHelper debugOverlay;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        setTitle("手语实时翻译");
        initViews();
        checkPermissions();

        // 初始化工具类
        frameCaptureHelper = new FrameCaptureHelper();
        motionDetector = new MotionDetector();
        frameHandler = new Handler(Looper.getMainLooper());
    }

    private void initViews() {
        textureView = findViewById(R.id.textureView);
        btnRecord = findViewById(R.id.btnRecord);
        btnSwitch = findViewById(R.id.btnSwitch);
        btnMicrophone = findViewById(R.id.btnMicrophone);
        tvStatus = findViewById(R.id.tvStatus);
        tvTranslation = findViewById(R.id.tvTranslation);

        // 初始状态
        btnRecord.setImageResource(R.drawable.ic_record_inactive_large);
        tvTranslation.setVisibility(View.GONE);

        // 按钮点击事件
        btnRecord.setOnClickListener(v -> toggleRecognition());
        btnSwitch.setOnClickListener(v -> switchCamera());
        btnMicrophone.setOnClickListener(v -> toggleMicrophone());

        // 初始禁用按钮，等待权限
        btnRecord.setEnabled(false);
        btnSwitch.setEnabled(false);
        btnMicrophone.setEnabled(false);

        // ===== 调试浮层初始化 =====
        TextView tvDebugOverlay = findViewById(R.id.tvDebugOverlay);
        debugOverlay = new DebugOverlayHelper(tvDebugOverlay);

        // 长按状态文字切换调试面板（仅在 DEBUG_MODE 开启时生效）
        if (Config.DEBUG_MODE) {
            tvStatus.setOnLongClickListener(v -> {
                boolean nowShowing = debugOverlay.isShowing();
                if (nowShowing) {
                    debugOverlay.hide();
                    Toast.makeText(this, "调试面板已隐藏", Toast.LENGTH_SHORT).show();
                } else {
                    debugOverlay.show();
                    debugOverlay.start();
                    Toast.makeText(this, "调试面板已显示", Toast.LENGTH_SHORT).show();
                }
                return true;
            });
        }
    }

    // ======================== 权限处理 ========================

    private void checkPermissions() {
        String[] basePermissions = {
                Manifest.permission.CAMERA
        };

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            requestPermissionsWithMediaAccess();
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            requestBasePermissions(basePermissions);
        } else {
            String[] legacyPermissions = {
                    Manifest.permission.CAMERA,
                    Manifest.permission.READ_EXTERNAL_STORAGE
            };
            requestBasePermissions(legacyPermissions);
        }
    }

    private void requestPermissionsWithMediaAccess() {
        String[] permissions;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            permissions = new String[] {
                    Manifest.permission.CAMERA,
                    Manifest.permission.READ_MEDIA_VIDEO,
                    Manifest.permission.READ_MEDIA_IMAGES
            };
        } else {
            permissions = new String[] {
                    Manifest.permission.CAMERA,
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
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
            @NonNull int[] grantResults) {
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
        Toast.makeText(this, "需要摄像头权限才能使用手语翻译功能", Toast.LENGTH_LONG).show();
        tvStatus.setText("权限被拒绝，无法使用摄像头");
        btnRecord.setEnabled(false);
        btnSwitch.setEnabled(false);
        btnMicrophone.setEnabled(false);
        showPermissionGuide();
    }

    private void showPermissionGuide() {
        new AlertDialog.Builder(this)
                .setTitle("需要权限")
                .setMessage("应用需要摄像头权限才能进行手语识别翻译。请在设置中授予权限。")
                .setPositiveButton("去设置", (dialog, which) -> openAppSettings())
                .setNegativeButton("取消", null)
                .show();
    }

    private void openAppSettings() {
        Intent intent = new Intent(android.provider.Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
        intent.setData(android.net.Uri.parse("package:" + getPackageName()));
        startActivity(intent);
    }

    // ======================== 相机初始化 ========================

    private void initCamera() {
        cameraHelper = new CameraHelper(this, textureView);

        cameraHelper.setCameraListener(new CameraHelper.CameraListener() {
            @Override
            public void onCameraOpened() {
                runOnUiThread(() -> {
                    tvStatus.setText("就绪");
                    btnRecord.setEnabled(true);
                    btnSwitch.setEnabled(true);
                    btnMicrophone.setEnabled(true);
                });
            }

            @Override
            public void onCameraClosed() {
                runOnUiThread(() -> {
                    tvStatus.setText("已关闭");
                });
            }

            @Override
            public void onError(String error) {
                runOnUiThread(() -> {
                    tvStatus.setText("错误: " + error);
                    showCameraErrorDialog(error);
                });
            }
        });

        btnRecord.setEnabled(true);
        btnSwitch.setEnabled(true);
        btnMicrophone.setEnabled(true);
        btnRecord.setImageResource(R.drawable.ic_record_inactive_large);
        tvStatus.setText("初始中...");
    }

    // ======================== 核心识别逻辑 ========================

    /**
     * 切换识别状态（开始/停止）
     */
    private void toggleRecognition() {
        if (cameraHelper == null)
            return;

        if (!isRecording) {
            startRecognition();
        } else {
            stopRecognition();
        }
    }

    /**
     * 开始实时识别：
     * 1. 连接 WebSocket
     * 2. 启动定时帧捕获循环
     */
    private void startRecognition() {
        // 重置状态
        frameIdx = 0;
        motionDetector.reset();
        if (debugOverlay != null) {
            debugOverlay.resetStats();
            debugOverlay.start();
        }

        // 初始化 WebSocket 客户端
        webSocketClient = new WebSocketClient(Config.WS_URL, new WebSocketClient.WebSocketCallback() {
            @Override
            public void onConnected() {
                isConnected = true;
                if (debugOverlay != null) {
                    debugOverlay.setConnectionStatus(true);
                }
                runOnUiThread(() -> {
                    tvStatus.setText("已连接");
                    startFrameCapture();
                });
            }

            @Override
            public void onDisconnected(int code, String reason) {
                isConnected = false;
                if (debugOverlay != null) {
                    debugOverlay.setConnectionStatus(false);
                }
                runOnUiThread(() -> {
                    tvStatus.setText("连接断开");
                });
            }

            @Override
            public void onTranslationReceived(String text, float confidence, int[] frameRange) {
                runOnUiThread(() -> {
                    // 更新翻译结果 UI
                    buttonRecognitionText(text, confidence);
                });
            }

            @Override
            public void onStatusReceived(float gpuUtil, float latencyMs) {
                // 可选：在调试模式显示服务器状态
                runOnUiThread(() -> {
                    if (gpuUtil > 0 && latencyMs > 0) {
                        tvStatus.setText("识别中 | " + (int) latencyMs + "ms");
                    }
                });
            }

            @Override
            public void onError(String error) {
                if (debugOverlay != null) {
                    debugOverlay.setLastError(error);
                }
                runOnUiThread(() -> {
                    Toast.makeText(MainActivity.this, error, Toast.LENGTH_SHORT).show();
                });
            }
        });

        // 建立连接
        webSocketClient.connect();

        // 更新 UI
        isRecording = true;
        btnRecord.setImageResource(R.drawable.ic_record_active_large);
        btnSwitch.setEnabled(false);
        btnMicrophone.setEnabled(false);
        tvStatus.setText("连接中...");
    }

    /**
     * 停止实时识别
     */
    private void stopRecognition() {
        // 停止帧捕获
        stopFrameCapture();

        // 停止调试覆盖层
        if (debugOverlay != null) {
            debugOverlay.stop();
        }

        // 断开 WebSocket
        if (webSocketClient != null) {
            webSocketClient.disconnect();
            webSocketClient = null;
        }

        // 隐藏翻译结果
        tvTranslation.setVisibility(View.GONE);
        tvTranslation.setText("");

        // 更新 UI
        isRecording = false;
        isConnected = false;
        btnRecord.setImageResource(R.drawable.ic_record_inactive_large);
        btnSwitch.setEnabled(true);
        btnMicrophone.setEnabled(true);
        tvStatus.setText("就绪");
    }

    /**
     * 启动定时帧捕获循环（10 FPS）
     */
    private void startFrameCapture() {
        frameCaptureRunnable = new Runnable() {
            @Override
            public void run() {
                if (!isRecording || !isConnected) {
                    return;
                }

                // 1. 从 TextureView 获取帧，并还原摄像头原始比例（去拉伸）
                Size previewSize = cameraHelper.getPreviewSize();
                int sensorOrientation = cameraHelper.getSensorOrientation();
                int displayRotation = cameraHelper.getDisplayRotation();
                Bitmap frameBmp = frameCaptureHelper.captureFrame(
                        textureView,
                        previewSize.getWidth(), previewSize.getHeight(),
                        sensorOrientation, displayRotation);
                if (frameBmp == null) {
                    // 帧获取失败，跳过本次
                    frameHandler.postDelayed(this, Config.CAPTURE_INTERVAL_MS);
                    return;
                }
                if (debugOverlay != null) {
                    debugOverlay.onFrameCaptured();
                }

                // 2. 运动检测 - 只在有运动时发送
                boolean hasMotion = motionDetector.detectMotion(frameBmp);
                if (!hasMotion) {
                    // 静止帧，丢弃不上传
                    if (debugOverlay != null) {
                        debugOverlay.onFrameSkipped();
                    }
                    frameHandler.postDelayed(this, Config.CAPTURE_INTERVAL_MS);
                    return;
                }

                // 3. 压缩为 JPEG + Base64 编码
                String base64Image = frameCaptureHelper.compressToBase64(frameBmp, Config.JPEG_QUALITY);
                if (base64Image == null) {
                    // 编码失败，跳过
                    frameHandler.postDelayed(this, Config.CAPTURE_INTERVAL_MS);
                    return;
                }

                // 4. 通过 WebSocket 发送
                webSocketClient.sendFrame(frameIdx++, base64Image, System.currentTimeMillis());
                if (debugOverlay != null) {
                    debugOverlay.onFrameSent(base64Image.length() / 1024);
                }

                // 继续下一帧
                frameHandler.postDelayed(this, Config.CAPTURE_INTERVAL_MS);
            }
        };

        // 启动循环
        frameHandler.postDelayed(frameCaptureRunnable, Config.CAPTURE_INTERVAL_MS);
        Log.d("MainActivity", "Frame capture started at " + Config.CAPTURE_INTERVAL_MS + "ms interval");
    }

    /**
     * 停止帧捕获循环
     */
    private void stopFrameCapture() {
        if (frameCaptureRunnable != null) {
            frameHandler.removeCallbacks(frameCaptureRunnable);
            frameCaptureRunnable = null;
        }
        Log.d("MainActivity", "Frame capture stopped");
    }

    /**
     * 更新翻译结果到 UI
     */
    private void buttonRecognitionText(String text, float confidence) {
        if (text == null || text.isEmpty()) {
            return;
        }

        // 置信度过低时显示特殊提示
        String displayText;
        if (confidence < 0.6f || "[不确定]".equals(text)) {
            displayText = "🤔 不确定";
        } else {
            displayText = text;
        }

        tvTranslation.setText(displayText);
        tvTranslation.setVisibility(View.VISIBLE);

        // 状态栏同时显示置信度
        if (confidence > 0) {
            tvStatus.setText("识别中 | " + String.format("%.0f%%", confidence * 100));
        } else {
            tvStatus.setText("识别中");
        }
    }

    // ======================== 其他控件 ========================

    private void switchCamera() {
        if (cameraHelper != null) {
            cameraHelper.switchCamera();
            // 切换摄像头后重置运动检测器
            motionDetector.reset();
        }
    }

    private void toggleMicrophone() {
        isMicrophoneOn = !isMicrophoneOn;
        if (isMicrophoneOn) {
            btnMicrophone.setColorFilter(ContextCompat.getColor(this, android.R.color.holo_blue_light));
            Toast.makeText(this, "语音输出：开", Toast.LENGTH_SHORT).show();
        } else {
            btnMicrophone.setColorFilter(ContextCompat.getColor(this, android.R.color.darker_gray));
            Toast.makeText(this, "语音输出：关", Toast.LENGTH_SHORT).show();
        }
    }

    // ======================== 生命周期管理 ========================

    @Override
    protected void onResume() {
        super.onResume();
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

    @Override
    protected void onPause() {
        super.onPause();
        if (isRecording) {
            stopRecognition();
        }
        if (cameraHelper != null) {
            cameraHelper.closeCamera();
            cameraHelper.stopBackgroundThread();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (isRecording) {
            stopRecognition();
        }
        if (cameraHelper != null) {
            cameraHelper.closeCamera();
            cameraHelper.stopBackgroundThread();
        }
    }

    // ======================== UI 辅助方法 ========================

    private void enableImmersiveMode() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
            getWindow().getDecorView().setSystemUiVisibility(
                    View.SYSTEM_UI_FLAG_FULLSCREEN
                            | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                            | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                            | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                            | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                            | View.SYSTEM_UI_FLAG_LAYOUT_STABLE);
        }
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

    private void retryCameraOpen() {
        new AlertDialog.Builder(this)
                .setTitle("摄像头错误")
                .setMessage("摄像头出现错误，是否重试？")
                .setPositiveButton("重试", (dialog, which) -> {
                    if (cameraHelper != null) {
                        cameraHelper.closeCamera();
                        new Handler(Looper.getMainLooper()).postDelayed(() -> {
                            cameraHelper.openCamera();
                        }, 1000);
                    }
                })
                .setNegativeButton("取消", null)
                .show();
    }
}
