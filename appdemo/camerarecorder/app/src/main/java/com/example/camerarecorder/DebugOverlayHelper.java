package com.example.camerarecorder;

import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.TextView;

/**
 * 调试浮层管理类
 * <p>
 * 在相机预览画面上叠加显示实时调试信息，帮助开发阶段验证：
 * - 帧采集是否正常（FPS、帧序号）
 * - 运动检测是否生效（跳过/发送比例）
 * - WebSocket 连接状态
 * - Base64 帧大小
 * <p>
 * 使用方式：
 * 1. 长按顶部 tvStatus 文字开关
 * 2. 或修改 Config.DEBUG_SHOWN_BY_DEFAULT = true
 * 3. 上线前设置 Config.DEBUG_MODE = false 彻底移除
 */
public class DebugOverlayHelper {

    private static final String TAG = "DebugOverlay";

    private final TextView debugView;
    private final Handler handler;
    private boolean isShowing = false;

    // 实时统计数据
    private int frameCount = 0;
    private int sentCount = 0;
    private int skippedCount = 0;
    private int lastB64SizeKb = 0;
    private boolean wsConnected = false;
    private String lastError = "";
    private long lastResetTime = System.currentTimeMillis();

    // FPS 计算
    private int fpsFrameCount = 0;
    private long fpsLastTime = System.currentTimeMillis();

    public DebugOverlayHelper(TextView debugView) {
        this.debugView = debugView;
        this.handler = new Handler(Looper.getMainLooper());
        this.isShowing = Config.DEBUG_SHOWN_BY_DEFAULT;
        updateVisibility();
    }

    /**
     * 记录一次帧采集事件（无论是否发送）
     */
    public void onFrameCaptured() {
        frameCount++;
        fpsFrameCount++;
    }

    /**
     * 记录一次成功发送的帧
     *
     * @param b64SizeKb 该帧 Base64 大小（KB）
     */
    public void onFrameSent(int b64SizeKb) {
        sentCount++;
        lastB64SizeKb = b64SizeKb;
    }

    /**
     * 记录一次被运动检测过滤的帧
     */
    public void onFrameSkipped() {
        skippedCount++;
    }

    /**
     * 更新 WebSocket 连接状态
     */
    public void setConnectionStatus(boolean connected) {
        this.wsConnected = connected;
    }

    /**
     * 记录最后一次错误
     */
    public void setLastError(String error) {
        this.lastError = error;
    }

    /**
     * 重置所有统计计数器
     */
    public void resetStats() {
        frameCount = 0;
        sentCount = 0;
        skippedCount = 0;
        lastB64SizeKb = 0;
        fpsFrameCount = 0;
        lastResetTime = System.currentTimeMillis();
        fpsLastTime = System.currentTimeMillis();
        lastError = "";
    }

    /**
     * 切换显示/隐藏
     */
    public void toggle() {
        isShowing = !isShowing;
        updateVisibility();
        Log.d(TAG, "Debug overlay " + (isShowing ? "shown" : "hidden"));
    }

    /**
     * 强制显示
     */
    public void show() {
        isShowing = true;
        updateVisibility();
    }

    /**
     * 强制隐藏
     */
    public void hide() {
        isShowing = false;
        updateVisibility();
    }

    /**
     * 当前是否可见
     */
    public boolean isShowing() {
        return isShowing;
    }

    /**
     * 启动定期刷新（每 Config.DEBUG_REFRESH_INTERVAL_MS 刷新一次显示）
     */
    public void start() {
        if (!Config.DEBUG_MODE)
            return;
        handler.post(updateRunnable);
    }

    /**
     * 停止定期刷新
     */
    public void stop() {
        handler.removeCallbacks(updateRunnable);
    }

    // ======================== 内部实现 ========================

    private void updateVisibility() {
        if (Config.DEBUG_MODE && isShowing) {
            debugView.setVisibility(View.VISIBLE);
        } else {
            debugView.setVisibility(View.GONE);
        }
    }

    private final Runnable updateRunnable = new Runnable() {
        @Override
        public void run() {
            if (!Config.DEBUG_MODE || !isShowing) {
                return;
            }

            // 计算实时 FPS
            long now = System.currentTimeMillis();
            long elapsed = now - fpsLastTime;
            float currentFps = 0;
            if (elapsed > 0) {
                currentFps = (float) fpsFrameCount / (elapsed / 1000f);
            }

            // 计算运行时长
            long runningSeconds = (now - lastResetTime) / 1000;

            // 跳过率
            String skipRate = "N/A";
            if (frameCount > 0) {
                skipRate = String.format("%.1f%%", (float) skippedCount / frameCount * 100);
            }

            // 构建调试文本
            StringBuilder sb = new StringBuilder();
            sb.append("⚙ 调试面板\n");
            sb.append("─────────────────\n");
            sb.append("📡 WS: ").append(wsConnected ? "✅ 已连接" : "❌ 未连接").append("\n");
            sb.append("📸 总帧: ").append(frameCount)
                    .append(" | FPS: ").append(String.format("%.1f", currentFps)).append("\n");
            sb.append("📤 发送: ").append(sentCount)
                    .append(" | ⏭ 跳过: ").append(skippedCount)
                    .append(" (").append(skipRate).append(")\n");
            sb.append("📦 末帧: ").append(lastB64SizeKb).append(" KB\n");
            sb.append("⏱ 运行: ").append(formatDuration(runningSeconds)).append("\n");

            if (!lastError.isEmpty()) {
                sb.append("⚠ ").append(lastError).append("\n");
            }

            debugView.setText(sb.toString());

            // 重置 FPS 计数器
            fpsFrameCount = 0;
            fpsLastTime = now;

            // 继续刷新
            handler.postDelayed(this, Config.DEBUG_REFRESH_INTERVAL_MS);
        }
    };

    /**
     * 将秒数格式化为 HH:MM:SS
     */
    private String formatDuration(long seconds) {
        long h = seconds / 3600;
        long m = (seconds % 3600) / 60;
        long s = seconds % 60;
        return String.format("%02d:%02d:%02d", h, m, s);
    }
}
