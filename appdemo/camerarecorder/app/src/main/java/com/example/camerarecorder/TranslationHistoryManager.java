package com.example.camerarecorder;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 翻译记录管理器
 * <p>
 * 维护一个内存中的翻译记录列表，并通过 SharedPreferences 持久化。
 * 支持添加记录、查询（按关键词过滤）、清除全部。
 */
public class TranslationHistoryManager {

    private static final String TAG = "TranslationHistory";
    private static final String PREFS_NAME = "translation_history";
    private static final String KEY_RECORDS = "records";
    private static final int MAX_RECORDS = 500; // 最多保留 500 条

    private final SharedPreferences prefs;
    private final List<TranslationRecord> records;
    private boolean loaded = false;

    public TranslationHistoryManager(Context context) {
        this.prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        this.records = new ArrayList<>();
    }

    /**
     * 添加一条翻译记录（自动持久化）
     */
    public void addRecord(TranslationRecord record) {
        lazyLoad();
        records.add(0, record); // 插入到头部，最新的在最前
        // 限制最大条数
        if (records.size() > MAX_RECORDS) {
            records.subList(MAX_RECORDS, records.size()).clear();
        }
        saveToPrefs();
        Log.d(TAG, "Record added: " + record.getResultText() + " (total: " + records.size() + ")");
    }

    /**
     * 获取所有记录（按时间倒序，最新的在前）
     */
    public List<TranslationRecord> getAllRecords() {
        lazyLoad();
        return Collections.unmodifiableList(records);
    }

    /**
     * 搜索翻译记录（按关键词过滤，不区分大小写）
     */
    public List<TranslationRecord> search(String query) {
        lazyLoad();
        if (query == null || query.trim().isEmpty()) {
            return getAllRecords();
        }
        String lowerQuery = query.trim().toLowerCase();
        List<TranslationRecord> result = new ArrayList<>();
        for (TranslationRecord record : records) {
            if (record.getResultText().toLowerCase().contains(lowerQuery)) {
                result.add(record);
            }
        }
        return result;
    }

    /**
     * 清除所有记录
     */
    public void clearAll() {
        records.clear();
        saveToPrefs();
        Log.d(TAG, "All records cleared");
    }

    /**
     * 获取记录总数
     */
    public int getCount() {
        lazyLoad();
        return records.size();
    }

    // ======================== 持久化 ========================

    private void lazyLoad() {
        if (!loaded) {
            loadFromPrefs();
            loaded = true;
        }
    }

    private void loadFromPrefs() {
        records.clear();
        String jsonStr = prefs.getString(KEY_RECORDS, "[]");
        try {
            JSONArray arr = new JSONArray(jsonStr);
            for (int i = 0; i < arr.length(); i++) {
                JSONObject obj = arr.getJSONObject(i);
                records.add(TranslationRecord.fromJson(obj));
            }
            Log.d(TAG, "Loaded " + records.size() + " records from storage");
        } catch (JSONException e) {
            Log.e(TAG, "Failed to load records", e);
        }
    }

    private void saveToPrefs() {
        JSONArray arr = new JSONArray();
        for (TranslationRecord record : records) {
            arr.put(record.toJson());
        }
        prefs.edit().putString(KEY_RECORDS, arr.toString()).apply();
    }
}
