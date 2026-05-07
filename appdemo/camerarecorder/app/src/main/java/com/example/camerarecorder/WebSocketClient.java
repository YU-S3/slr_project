package com.example.camerarecorder;

import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import org.json.JSONArray;
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
 * 协议遵循 API (1).md 规范：
 * - 连接后服务端返回 session_started（含 processing_mode）
 * - 帧消息含 flush / is_final 字段
 * - 支持 ping（心跳）和 reset（重置会话）
 * - 服务端返回 recording / processing / translation / error
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

    /** 会话信息（从 session_started 获得） */
    private String sessionId = null;
    private int frameWindowSize = Config.DEFAULT_FRAME_WINDOW_SIZE;
    private String processingMode = ""; // "full_recording" 等

    /**
     * WebSocket 回调接口（与 API (1).md 协议对齐）
     */
    public interface WebSocketCallback {
        /** 连接成功建立 */
        void onConnected();

        /** 连接断开 */
        void onDisconnected(int code, String reason);

        /**
         * 收到 session_started
         * 
         * @param sessionId       会话 ID
         * @param frameWindowSize 帧窗口大小（现为信息性字段）
         * @param processingMode  处理模式（如 "full_recording"）
         */
        void onSessionStarted(String sessionId, int frameWindowSize, String processingMode);

        /**
         * 收到录制进度通知（服务端 type: "recording"）
         * 
         * @param frameIdx       当前帧序号
         * @param bufferedFrames 已缓冲帧数
         */
        void onRecording(int frameIdx, int bufferedFrames);

        /**
         * 收到后端处理中通知（服务端 type: "processing"）
         * 
         * @param sessionId   会话 ID
         * @param frameIdx    总帧数
         * @param totalFrames 总帧数
         */
        void onProcessing(String sessionId, int frameIdx, int totalFrames);

        /** 收到翻译结果 */
        void onTranslationReceived(String sessionId, int frameIdx, String result, JSONArray ragHits);

        /** 发生错误 */
        void onError(String error);
    }

    /**
     * @param serverUrl WebSocket 服务器地址，例如 "ws://192.168.2.32:8000/ws"
     * @param callback  事件回调
     */
    public WebSocketClient(String serverUrl, WebSocketCallback callback) {
        this.serverUrl = serverUrl;
        this.callback = callback;
        this.mainHandler = new Handler(Looper.getMainLooper());

        this.httpClient = new OkHttpClient.Builder()
                .connectTimeout(10, TimeUnit.SECONDS)
                .readTimeout(60, TimeUnit.SECONDS) // 整段录制处理可能耗时较长
                .writeTimeout(30, TimeUnit.SECONDS)
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
                sessionId = null;
                processingMode = "";

                // 连接成功，等待服务端下发 session_started
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
        sessionId = null;
        processingMode = "";
        Log.d(TAG, "WebSocket disconnected by client");
    }

    /**
     * 发送一帧数据到后端（遵循 API (1).md 协议）
     *
     * @param frameIdx    帧序号（从0递增）
     * @param base64Image JPEG 压缩并 Base64 编码后的图像数据
     * @param flush       是否触发整段录制内容的最终处理
     * @param isFinal     是否为最后一帧
     */
    public void sendFrame(int frameIdx, String base64Image, boolean flush, boolean isFinal) {
        if (!connected || webSocket == null) {
            Log.w(TAG, "Cannot send frame: not connected");
            return;
        }

        try {
            JSONObject json = new JSONObject();
            json.put("type", "frame");
            json.put("frame_idx", frameIdx);
            json.put("image_b64", base64Image);
            json.put("flush", flush);
            json.put("is_final", isFinal);

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
     * 发送心跳（ping）
     */
    public void sendPing() {
        if (!connected || webSocket == null) {
            Log.w(TAG, "Cannot send ping: not connected");
            return;
        }

        try {
            JSONObject json = new JSONObject();
            json.put("type", "ping");
            webSocket.send(json.toString());
            Log.d(TAG, "Ping sent");
        } catch (JSONException e) {
            Log.e(TAG, "Error creating ping JSON", e);
        }
    }

    /**
     * 发送重置会话指令
     */
    public void sendReset() {
        if (!connected || webSocket == null) {
            Log.w(TAG, "Cannot send reset: not connected");
            return;
        }

        try {
            JSONObject json = new JSONObject();
            json.put("type", "reset");
            webSocket.send(json.toString());
            Log.d(TAG, "Reset sent");
        } catch (JSONException e) {
            Log.e(TAG, "Error creating reset JSON", e);
        }
    }

    /**
     * 获取会话 ID
     */
    public String getSessionId() {
        return sessionId;
    }

    /**
     * 获取帧窗口大小
     */
    public int getFrameWindowSize() {
        return frameWindowSize;
    }

    /**
     * 获取处理模式
     */
    public String getProcessingMode() {
        return processingMode;
    }

    /**
     * 检查是否已连接
     */
    public boolean isConnected() {
        return connected;
    }

    // ======================== 私有方法 ========================

    /**
     * 解析并处理服务器下发的 JSON 消息（遵循 API (1).md 协议）
     */
    private void handleServerMessage(String text) {
        try {
            JSONObject json = new JSONObject(text);
            String type = json.optString("type", "");

            switch (type) {
                case "session_started":
                    handleSessionStarted(json);
                    break;
                case "recording":
                    // 新 API：录制中通知
                    handleRecording(json);
                    break;
                case "processing":
                    // 新 API：后端处理中通知
                    handleProcessing(json);
                    break;
                case "translation":
                    handleTranslation(json);
                    break;
                case "error":
                    String errorMsg = json.optString("message", "未知服务器错误");
                    mainHandler.post(() -> {
                        if (callback != null) {
                            callback.onError(errorMsg);
                        }
                    });
                    break;
                case "buffering":
                    // 兼容旧 API：映射到 recording 处理
                    handleRecording(json);
                    break;
                default:
                    Log.w(TAG, "Unknown message type: " + type);
            }
        } catch (JSONException e) {
            Log.e(TAG, "Error parsing server message", e);
        }
    }

    /**
     * 处理 session_started 消息
     */
    private void handleSessionStarted(JSONObject json) throws JSONException {
        sessionId = json.optString("session_id", null);
        frameWindowSize = json.optInt("frame_window_size", Config.DEFAULT_FRAME_WINDOW_SIZE);
        processingMode = json.optString("processing_mode", "");

        Log.d(TAG, "Session started: id=" + sessionId
                + " window=" + frameWindowSize
                + " mode=" + processingMode);

        final String sid = sessionId;
        final int fws = frameWindowSize;
        final String pm = processingMode;

        mainHandler.post(() -> {
            if (callback != null) {
                callback.onSessionStarted(sid, fws, pm);
            }
        });
    }

    /**
     * 处理 recording 消息（新 API）
     */
    private void handleRecording(JSONObject json) {
        int frameIdx = json.optInt("frame_idx", -1);
        int bufferedFrames = json.optInt("buffered_frames", 0);

        final int fidx = frameIdx;
        final int bf = bufferedFrames;

        mainHandler.post(() -> {
            if (callback != null) {
                callback.onRecording(fidx, bf);
            }
        });
    }

    /**
     * 处理 processing 消息（新 API）
     */
    private void handleProcessing(JSONObject json) {
        String sid = json.optString("session_id", sessionId != null ? sessionId : "");
        int frameIdx = json.optInt("frame_idx", -1);
        int totalFrames = json.optInt("total_frames", 0);

        final String finalSid = sid;
        final int finalFrameIdx = frameIdx;
        final int finalTotalFrames = totalFrames;

        mainHandler.post(() -> {
            if (callback != null) {
                callback.onProcessing(finalSid, finalFrameIdx, finalTotalFrames);
            }
        });
    }

    /**
     * 处理翻译结果消息
     */
    private void handleTranslation(JSONObject json) {
        String sid = json.optString("session_id", sessionId != null ? sessionId : "");
        int frameIdx = json.optInt("frame_idx", -1);
        String result = json.optString("result", "");
        JSONArray ragHits = json.optJSONArray("rag_hits");

        final String finalSid = sid;
        final int finalFrameIdx = frameIdx;
        final String finalResult = result;
        final JSONArray finalRagHits = ragHits;

        mainHandler.post(() -> {
            if (callback != null) {
                callback.onTranslationReceived(finalSid, finalFrameIdx, finalResult, finalRagHits);
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
