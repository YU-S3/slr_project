package com.example.camerarecorder;

import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import org.json.JSONException;
import org.json.JSONObject;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.WebSocket;
import okhttp3.WebSocketListener;

import java.util.concurrent.TimeUnit;

/**
 * WebSocket 客户端管理器
 * <p>
 * 负责与后端建立 WebSocket 长连接，发送帧数据，接收翻译结果。
 * 内置自动重连机制。
 */
public class WebSocketClient {

    private static final String TAG = "WebSocketClient";

    private final String serverUrl;
    private final WebSocketCallback callback;
    private final OkHttpClient httpClient;
    private final Handler mainHandler;

    private WebSocket webSocket;
    private boolean connected = false;
    private int reconnectAttempts = 0;
    private boolean shouldReconnect = true;

    /**
     * WebSocket 回调接口
     */
    public interface WebSocketCallback {
        /** 连接成功建立 */
        void onConnected();

        /** 连接断开 */
        void onDisconnected(int code, String reason);

        /** 收到翻译结果 */
        void onTranslationReceived(String text, float confidence, int[] frameRange);

        /** 收到服务器状态信息 */
        void onStatusReceived(float gpuUtil, float latencyMs);

        /** 发生错误 */
        void onError(String error);
    }

    /**
     * @param serverUrl WebSocket 服务器地址，例如 "ws://192.168.1.100:8000/ws/translate"
     * @param callback  事件回调
     */
    public WebSocketClient(String serverUrl, WebSocketCallback callback) {
        this.serverUrl = serverUrl;
        this.callback = callback;
        this.mainHandler = new Handler(Looper.getMainLooper());

        this.httpClient = new OkHttpClient.Builder()
                .connectTimeout(10, TimeUnit.SECONDS)
                .readTimeout(30, TimeUnit.SECONDS)
                .writeTimeout(10, TimeUnit.SECONDS)
                .pingInterval(15, TimeUnit.SECONDS) // 心跳保活
                .build();
    }

    /**
     * 建立 WebSocket 连接
     */
    public void connect() {
        if (connected) {
            Log.w(TAG, "Already connected");
            return;
        }

        shouldReconnect = true;
        Log.d(TAG, "Connecting to: " + serverUrl);

        Request request = new Request.Builder()
                .url(serverUrl)
                .build();

        httpClient.newWebSocket(request, new WebSocketListener() {
            @Override
            public void onOpen(WebSocket ws, Response response) {
                Log.d(TAG, "WebSocket connected");
                connected = true;
                reconnectAttempts = 0;
                webSocket = ws;

                // 在主线程回调
                mainHandler.post(() -> {
                    if (callback != null) {
                        callback.onConnected();
                    }
                });
            }

            @Override
            public void onMessage(WebSocket ws, String text) {
                Log.d(TAG, "Received message: " + text);
                handleServerMessage(text);
            }

            @Override
            public void onClosing(WebSocket ws, int code, String reason) {
                Log.d(TAG, "WebSocket closing: code=" + code + " reason=" + reason);
                ws.close(1000, null);
            }

            @Override
            public void onClosed(WebSocket ws, int code, String reason) {
                Log.d(TAG, "WebSocket closed: code=" + code + " reason=" + reason);
                connected = false;
                webSocket = null;

                mainHandler.post(() -> {
                    if (callback != null) {
                        callback.onDisconnected(code, reason);
                    }
                    attemptReconnect();
                });
            }

            @Override
            public void onFailure(WebSocket ws, Throwable t, Response response) {
                String errorDetail = (t.getMessage() != null) ? t.getMessage() : t.toString();
                Log.e(TAG, "WebSocket failure: " + errorDetail, t);
                connected = false;
                webSocket = null;

                mainHandler.post(() -> {
                    if (callback != null) {
                        callback.onError("连接失败: " + errorDetail);
                    }
                    attemptReconnect();
                });
            }
        });
    }

    /**
     * 主动断开 WebSocket 连接
     */
    public void disconnect() {
        shouldReconnect = false;
        reconnectAttempts = 0;
        if (webSocket != null) {
            webSocket.close(1000, "Client closing");
            webSocket = null;
        }
        connected = false;
        Log.d(TAG, "WebSocket disconnected by client");
    }

    /**
     * 发送一帧数据到后端
     *
     * @param frameIdx    帧序号（从0递增）
     * @param base64Image JPEG 压缩并 Base64 编码后的图像数据
     * @param timestamp   捕获时间戳（毫秒）
     */
    public void sendFrame(int frameIdx, String base64Image, long timestamp) {
        if (!connected || webSocket == null) {
            Log.w(TAG, "Cannot send frame: not connected");
            return;
        }

        try {
            JSONObject json = new JSONObject();
            json.put("type", "frame");
            json.put("frame_idx", frameIdx);
            json.put("image_b64", base64Image);
            json.put("timestamp", timestamp);

            String message = json.toString();
            boolean sent = webSocket.send(message);

            if (!sent) {
                Log.w(TAG, "Failed to send frame " + frameIdx);
            }
        } catch (JSONException e) {
            Log.e(TAG, "Error creating JSON frame", e);
        }
    }

    /**
     * 发送控制指令到后端（开始/停止识别）
     *
     * @param action "start" 或 "stop"
     */
    public void sendControl(String action) {
        if (!connected || webSocket == null) {
            Log.w(TAG, "Cannot send control: not connected");
            return;
        }

        try {
            JSONObject json = new JSONObject();
            json.put("type", "control");
            json.put("action", action);
            webSocket.send(json.toString());
            Log.d(TAG, "Control sent: " + action);
        } catch (JSONException e) {
            Log.e(TAG, "Error creating control JSON", e);
        }
    }

    /**
     * 检查是否已连接
     */
    public boolean isConnected() {
        return connected;
    }

    // ======================== 私有方法 ========================

    /**
     * 解析并处理服务器下发的 JSON 消息
     */
    private void handleServerMessage(String text) {
        try {
            JSONObject json = new JSONObject(text);
            String type = json.optString("type", "");

            switch (type) {
                case "translation":
                    handleTranslation(json);
                    break;
                case "status":
                    handleStatus(json);
                    break;
                case "error":
                    String errorMsg = json.optString("message", "未知服务器错误");
                    mainHandler.post(() -> {
                        if (callback != null) {
                            callback.onError(errorMsg);
                        }
                    });
                    break;
                default:
                    Log.w(TAG, "Unknown message type: " + type);
            }
        } catch (JSONException e) {
            Log.e(TAG, "Error parsing server message", e);
        }
    }

    /**
     * 处理翻译结果消息
     */
    private void handleTranslation(JSONObject json) throws JSONException {
        String text = json.optString("text", "");
        float confidence = (float) json.optDouble("confidence", 0.0);

        // frame_range 是可选字段
        int[] frameRange = null;
        if (json.has("frame_range")) {
            // 服务端可能返回 JSON 数组
            String rangeStr = json.optString("frame_range", null);
            if (rangeStr != null) {
                try {
                    // 尝试解析为 JSON 数组
                    String clean = rangeStr.replace("[", "").replace("]", "");
                    String[] parts = clean.split(",");
                    if (parts.length == 2) {
                        frameRange = new int[] {
                                Integer.parseInt(parts[0].trim()),
                                Integer.parseInt(parts[1].trim())
                        };
                    }
                } catch (Exception e) {
                    Log.w(TAG, "Failed to parse frame_range", e);
                }
            }
        }

        final String resultText = text;
        final float resultConfidence = confidence;
        final int[] resultRange = frameRange;

        mainHandler.post(() -> {
            if (callback != null) {
                callback.onTranslationReceived(resultText, resultConfidence, resultRange);
            }
        });
    }

    /**
     * 处理服务器状态消息
     */
    private void handleStatus(JSONObject json) {
        float gpuUtil = (float) json.optDouble("gpu_util", -1);
        float latencyMs = (float) json.optDouble("latency_ms", -1);

        final float gpu = gpuUtil;
        final float latency = latencyMs;

        mainHandler.post(() -> {
            if (callback != null) {
                callback.onStatusReceived(gpu, latency);
            }
        });
    }

    /**
     * 尝试自动重连
     */
    private void attemptReconnect() {
        if (!shouldReconnect) {
            Log.d(TAG, "Reconnect disabled, skipping");
            return;
        }

        if (reconnectAttempts >= Config.MAX_RECONNECT_ATTEMPTS) {
            Log.e(TAG, "Max reconnect attempts reached (" + Config.MAX_RECONNECT_ATTEMPTS + ")");
            mainHandler.post(() -> {
                if (callback != null) {
                    callback.onError("无法连接到服务器，已停止重连");
                }
            });
            return;
        }

        reconnectAttempts++;
        int delay = Config.RECONNECT_DELAY_MS;
        Log.d(TAG, "Reconnecting in " + delay + "ms (attempt " + reconnectAttempts + "/" + Config.MAX_RECONNECT_ATTEMPTS
                + ")");

        mainHandler.postDelayed(this::connect, delay);
    }
}
