package com.example.camerarecorder;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.speech.tts.TextToSpeech;
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

import java.util.ArrayList;
import java.util.Locale;

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
    private ImageButton btnRecord, btnSwitch, btnMicrophone, btnHistory;
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
    private boolean isSendingFrames = false; // 正在发送缓存的帧
    private int frameIdx = 0;

    // API (1).md 协议状态
    private int frameWindowSize = Config.DEFAULT_FRAME_WINDOW_SIZE;
    private String processingMode = "";

    // 帧缓冲区（本地缓存整段录制，停止后一次性发送）
    private ArrayList<String> frameBuffer;

    // 定时抽帧
    private Handler frameHandler;
    private Runnable frameCaptureRunnable;

    // TTS 语音合成
    private TextToSpeech textToSpeech;

    // 调试浮层
    private DebugOverlayHelper debugOverlay;

    // 翻译记录
    private TranslationHistoryManager historyManager;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        setTitle("手语实时翻译");

        // 初始化翻译记录管理器
        historyManager = new TranslationHistoryManager(this);
        initViews();
        checkPermissions();

        // 初始化工具类
        frameCaptureHelper = new FrameCaptureHelper();
        motionDetector = new MotionDetector();
        frameHandler = new Handler(Looper.getMainLooper());

        // 初始化 TTS（仅在首次需要时真正初始化引擎）
        initTextToSpeech();
    }

    private void initViews() {
        textureView = findViewById(R.id.textureView);
        btnRecord = findViewById(R.id.btnRecord);
        btnSwitch = findViewById(R.id.btnSwitch);
        btnMicrophone = findViewById(R.id.btnMicrophone);
        btnHistory = findViewById(R.id.btnHistory);
        tvStatus = findViewById(R.id.tvStatus);
        tvTranslation = findViewById(R.id.tvTranslation);

        // 初始状态
        btnRecord.setImageResource(R.drawable.ic_record_inactive_large);
        tvTranslation.setVisibility(View.GONE);

        // 按钮点击事件
        btnRecord.setOnClickListener(v -> toggleRecognition());
        btnSwitch.setOnClickListener(v -> switchCamera());
        btnMicrophone.setOnClickListener(v -> toggleMicrophone());
        btnHistory.setOnClickListener(v -> {
            // 打开翻译记录页面
            Intent intent = new Intent(MainActivity.this, HistoryActivity.class);
            startActivity(intent);
        });

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
        isSendingFrames = false;
        frameBuffer = new ArrayList<>();
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
                // 等待 session_started 后才开始帧捕获
                runOnUiThread(() -> {
                    tvStatus.setText("等待会话...");
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
            public void onSessionStarted(String sessionId, int fws, String pm) {
                frameWindowSize = fws;
                processingMode = pm;
                Log.d("MainActivity", "Session started: id=" + sessionId
                        + " window=" + frameWindowSize + " mode=" + processingMode);
                runOnUiThread(() -> {
                    tvStatus.setText("录制中...");
                    // 收到 session_started 后启动帧捕获（本地缓存，不实时发送）
                    startFrameCapture();
                });
            }

            @Override
            public void onRecording(int fidx, int bufferedFrames) {
                runOnUiThread(() -> {
                    if (isSendingFrames) {
                        tvStatus.setText("发送帧 (" + bufferedFrames + "/" + frameBuffer.size() + ")");
                    }
                });
            }

            @Override
            public void onProcessing(String sessionId, int frameIdx, int totalFrames) {
                Log.d("MainActivity", "Processing: session=" + sessionId
                        + " frames=" + frameIdx + "/" + totalFrames);
                runOnUiThread(() -> {
                    tvStatus.setText("后端处理中...");
                    tvTranslation.setText("后端处理中，请稍候");
                    tvTranslation.setVisibility(View.VISIBLE);
                });
            }

            @Override
            public void onTranslationReceived(String sessionId, int frameIdx, String result,
                    org.json.JSONArray ragHits) {
                // 保存翻译记录
                TranslationRecord record = new TranslationRecord(
                        System.currentTimeMillis(),
                        result,
                        frameIdx,
                        sessionId);
                historyManager.addRecord(record);
                Log.d("MainActivity", "Saved translation record: " + result + " (frame=" + frameIdx + ")");

                runOnUiThread(() -> {
                    // 更新翻译结果 UI（只显示 result 文本）
                    buttonRecognitionText(result);
                    // 收到翻译结果后清理会话
                    cleanupAfterTranslation();
                });
            }

            @Override
            public void onError(String error) {
                if (debugOverlay != null) {
                    debugOverlay.setLastError(error);
                }
                runOnUiThread(() -> {
                    Toast.makeText(MainActivity.this, error, Toast.LENGTH_SHORT).show();
                    // 出错时也清理
                    if (isSendingFrames || isRecording) {
                        cleanupAfterTranslation();
                    }
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
     * 停止实时识别（API (1).md 新流程）：
     * 1. 停止帧捕获
     * 2. 将本地缓存的帧一次性发送给后端
     * 3. 最后一帧设 flush=true, is_final=true
     * 4. 等待后端处理并返回翻译结果
     */
    private void stopRecognition() {
        // 停止帧捕获
        stopFrameCapture();

        // 检查是否有缓存的帧需要发送
        if (frameBuffer != null && !frameBuffer.isEmpty()) {
            // 发送所有缓存的帧
            sendBufferedFrames();
        } else {
            // 没有帧数据，直接清理
            cleanupAfterTranslation();
        }
    }

    /**
     * 将本地缓存的帧一次性发送给后端
     */
    private void sendBufferedFrames() {
        if (webSocketClient == null || !webSocketClient.isConnected()) {
            Log.w("MainActivity", "Cannot send frames: not connected");
            cleanupAfterTranslation();
            return;
        }

        isSendingFrames = true;
        int totalFrames = frameBuffer.size();
        Log.d("MainActivity", "Sending " + totalFrames + " buffered frames to server");

        runOnUiThread(() -> {
            tvStatus.setText("发送帧 (0/" + totalFrames + ")");
            btnRecord.setEnabled(false); // 发送期间禁用按钮
        });

        // 逐帧发送（不阻塞主线程）
        new Thread(() -> {
            try {
                for (int i = 0; i < totalFrames; i++) {
                    if (webSocketClient == null || !webSocketClient.isConnected()) {
                        Log.e("MainActivity", "Connection lost while sending frames");
                        runOnUiThread(() -> {
                            Toast.makeText(MainActivity.this, "发送中断：连接已断开", Toast.LENGTH_SHORT).show();
                            cleanupAfterTranslation();
                        });
                        return;
                    }

                    boolean isLast = (i == totalFrames - 1);
                    String base64Image = frameBuffer.get(i);
                    webSocketClient.sendFrame(i, base64Image, isLast, isLast);

                    final int sentCount = i + 1;
                    runOnUiThread(() -> {
                        tvStatus.setText("发送帧 (" + sentCount + "/" + totalFrames + ")");
                    });

                    // 每帧之间稍作延迟，避免拥塞
                    Thread.sleep(50);
                }

                Log.d("MainActivity", "All " + totalFrames + " frames sent successfully");
                runOnUiThread(() -> {
                    tvStatus.setText("等待后端处理...");
                });
            } catch (InterruptedException e) {
                Log.e("MainActivity", "Frame sending interrupted", e);
                runOnUiThread(() -> {
                    cleanupAfterTranslation();
                });
            }
        }).start();
    }

    /**
     * 翻译完成或出错后的清理工作
     */
    private void cleanupAfterTranslation() {
        // 停止调试覆盖层
        if (debugOverlay != null) {
            debugOverlay.stop();
        }

        // 清理缓冲区
        if (frameBuffer != null) {
            frameBuffer.clear();
        }
        isSendingFrames = false;

        // 断开 WebSocket（保持连接直到翻译完成）
        if (webSocketClient != null) {
            webSocketClient.disconnect();
            webSocketClient = null;
        }

        // 更新 UI
        isRecording = false;
        isConnected = false;
        btnRecord.setImageResource(R.drawable.ic_record_inactive_large);
        btnRecord.setEnabled(true);
        btnSwitch.setEnabled(true);
        btnMicrophone.setEnabled(true);
        tvStatus.setText("就绪");
    }

    /**
     * 启动定时帧捕获循环（10 FPS）—— 帧缓存到本地，不实时发送
     * 遵循 API (1).md 新流程：先录制缓存，停止后一次性提交
     */
    private void startFrameCapture() {
        frameCaptureRunnable = new Runnable() {
            @Override
            public void run() {
                if (!isRecording || !isConnected) {
                    return;
                }

                // 缓冲区安全上限检查
                if (frameBuffer != null && frameBuffer.size() >= Config.FRAME_BUFFER_MAX_SIZE) {
                    Log.w("MainActivity", "Frame buffer full, stopping capture");
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

                // 2. 运动检测 - 只在有运动时缓存
                boolean hasMotion = motionDetector.detectMotion(frameBmp);
                if (!hasMotion) {
                    // 静止帧，丢弃不缓存
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

                // 4. 缓存帧到本地缓冲区（停止后一次性发送，遵循 API (1).md 新流程）
                if (frameBuffer != null) {
                    frameBuffer.add(base64Image);
                    frameIdx++;
                    if (debugOverlay != null) {
                        debugOverlay.onFrameSent(base64Image.length() / 1024);
                    }
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
        Log.d("MainActivity", "Frame capture stopped, buffered frames: "
                + (frameBuffer != null ? frameBuffer.size() : 0));
    }

    /**
     * 更新翻译结果到 UI，并在语音输出开启时朗读
     */
    private void buttonRecognitionText(String text) {
        if (text == null || text.isEmpty()) {
            return;
        }

        tvTranslation.setText(text);
        tvTranslation.setVisibility(View.VISIBLE);

        // 语音输出（如果开启）
        if (isMicrophoneOn && textToSpeech != null) {
            speakTranslation(text);
        }

        tvStatus.setText("识别中");
    }

    // ======================== 语音合成 (TTS) ========================

    /**
     * 初始化 TextToSpeech 引擎
     * <p>
     * 使用 Android 内置 TTS 引擎，语种设为中文。
     * 初始化回调中设置语速和音调。
     */
    private void initTextToSpeech() {
        textToSpeech = new TextToSpeech(this, status -> {
            if (status == TextToSpeech.SUCCESS) {
                int langResult = textToSpeech.setLanguage(Locale.CHINESE);
                if (langResult == TextToSpeech.LANG_MISSING_DATA
                        || langResult == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.w("MainActivity", "TTS: 中文语音数据缺失或不支持");
                }
                // 语速 1.0 = 正常，略慢以便手语翻译清晰
                textToSpeech.setSpeechRate(0.9f);
                // 音调 1.0 = 正常
                textToSpeech.setPitch(1.0f);
                Log.d("MainActivity", "TTS initialized: " + Locale.CHINESE);
            } else {
                Log.e("MainActivity", "TTS initialization failed, status=" + status);
            }
        });
    }

    /**
     * 朗读翻译结果文本
     *
     * @param text 要朗读的中文文本
     */
    private void speakTranslation(String text) {
        if (textToSpeech == null) {
            Log.w("MainActivity", "TTS not initialized");
            return;
        }
        // 停止当前正在播放的语音
        if (textToSpeech.isSpeaking()) {
            textToSpeech.stop();
        }
        // 朗读文本（不加入队列，直接抢占播放）
        textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, "tts_" + System.currentTimeMillis());
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
            // 关闭语音时立即停止正在播放的 TTS
            if (textToSpeech != null && textToSpeech.isSpeaking()) {
                textToSpeech.stop();
            }
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
        // 释放 TTS 资源
        if (textToSpeech != null) {
            textToSpeech.stop();
            textToSpeech.shutdown();
            textToSpeech = null;
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
