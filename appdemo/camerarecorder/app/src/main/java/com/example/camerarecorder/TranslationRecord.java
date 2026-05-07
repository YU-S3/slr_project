package com.example.camerarecorder;

import org.json.JSONException;
import org.json.JSONObject;

/**
 * 翻译记录数据模型
 * 存储单次翻译结果及其上下文信息
 */
public class TranslationRecord {

    private final long timestamp; // 翻译时间（毫秒）
    private final String resultText; // 翻译结果文本
    private final int frameIdx; // 触发翻译的帧序号
    private final String sessionId; // 会话 ID

    public TranslationRecord(long timestamp, String resultText, int frameIdx, String sessionId) {
        this.timestamp = timestamp;
        this.resultText = resultText;
        this.frameIdx = frameIdx;
        this.sessionId = sessionId;
    }

    /** 从 JSON 反序列化 */
    public static TranslationRecord fromJson(JSONObject json) throws JSONException {
        return new TranslationRecord(
                json.getLong("timestamp"),
                json.getString("resultText"),
                json.getInt("frameIdx"),
                json.optString("sessionId", ""));
    }

    /** 序列化为 JSON */
    public JSONObject toJson() {
        try {
            JSONObject json = new JSONObject();
            json.put("timestamp", timestamp);
            json.put("resultText", resultText);
            json.put("frameIdx", frameIdx);
            json.put("sessionId", sessionId);
            return json;
        } catch (JSONException e) {
            return new JSONObject();
        }
    }

    // ======================== Getter ========================

    public long getTimestamp() {
        return timestamp;
    }

    public String getResultText() {
        return resultText;
    }

    public int getFrameIdx() {
        return frameIdx;
    }

    public String getSessionId() {
        return sessionId;
    }

    /**
     * 判断记录是否匹配搜索关键词
     */
    public boolean matchesQuery(String query) {
        if (query == null || query.isEmpty())
            return true;
        String lower = query.toLowerCase();
        return resultText.toLowerCase().contains(lower);
    }
}
