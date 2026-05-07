package com.example.camerarecorder;

import android.content.Context;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

/**
 * 翻译记录查看与搜索 Activity
 * <p>
 * 展示所有历史翻译记录，支持关键词搜索和清除。
 */
public class HistoryActivity extends AppCompatActivity {

    private TranslationHistoryManager historyManager;
    private ListView listView;
    private TextView tvEmpty, tvRecordCount;
    private EditText etSearch;
    private ImageButton btnClearSearch;
    private Button btnClearAll;
    private HistoryAdapter adapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_history);

        historyManager = new TranslationHistoryManager(this);

        initViews();
        loadRecords();
    }

    private void initViews() {
        // 返回按钮
        ImageButton btnBack = findViewById(R.id.btnBack);
        btnBack.setOnClickListener(v -> finish());

        // 记录计数
        tvRecordCount = findViewById(R.id.tvRecordCount);

        // 搜索
        etSearch = findViewById(R.id.etSearch);
        btnClearSearch = findViewById(R.id.btnClearSearch);

        etSearch.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {
            }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                String query = s.toString().trim();
                btnClearSearch.setVisibility(query.isEmpty() ? View.GONE : View.VISIBLE);
                searchRecords(query);
            }

            @Override
            public void afterTextChanged(Editable s) {
            }
        });

        btnClearSearch.setOnClickListener(v -> {
            etSearch.setText("");
            searchRecords("");
        });

        // 清除全部
        btnClearAll = findViewById(R.id.btnClearAll);
        btnClearAll.setOnClickListener(v -> confirmClearAll());

        // 列表
        listView = findViewById(R.id.listViewHistory);
        tvEmpty = findViewById(R.id.tvEmpty);

        adapter = new HistoryAdapter(this, new ArrayList<>());
        listView.setAdapter(adapter);
    }

    private void loadRecords() {
        List<TranslationRecord> all = historyManager.getAllRecords();
        updateUI(all);
    }

    private void searchRecords(String query) {
        List<TranslationRecord> results = historyManager.search(query);
        updateUI(results);
    }

    private void updateUI(List<TranslationRecord> records) {
        adapter.setData(records);
        adapter.notifyDataSetChanged();

        boolean empty = records.isEmpty();
        listView.setVisibility(empty ? View.GONE : View.VISIBLE);
        tvEmpty.setVisibility(empty ? View.VISIBLE : View.GONE);
        tvRecordCount.setText(records.size() + "条");
    }

    private void confirmClearAll() {
        new AlertDialog.Builder(this)
                .setTitle("清除全部记录")
                .setMessage("确定要删除所有翻译记录吗？此操作不可恢复。")
                .setPositiveButton("确定", (dialog, which) -> {
                    historyManager.clearAll();
                    loadRecords();
                    Toast.makeText(this, "已清除全部记录", Toast.LENGTH_SHORT).show();
                })
                .setNegativeButton("取消", null)
                .show();
    }

    // ======================== 列表适配器 ========================

    private static class HistoryAdapter extends BaseAdapter {

        private final LayoutInflater inflater;
        private List<TranslationRecord> records;
        private final SimpleDateFormat timeFormat;

        HistoryAdapter(Context context, List<TranslationRecord> records) {
            this.inflater = LayoutInflater.from(context);
            this.records = records;
            this.timeFormat = new SimpleDateFormat("HH:mm:ss", Locale.getDefault());
        }

        void setData(List<TranslationRecord> records) {
            this.records = records;
        }

        @Override
        public int getCount() {
            return records.size();
        }

        @Override
        public Object getItem(int position) {
            return records.get(position);
        }

        @Override
        public long getItemId(int position) {
            return position;
        }

        @Override
        public View getView(int position, View convertView, ViewGroup parent) {
            if (convertView == null) {
                convertView = inflater.inflate(android.R.layout.simple_list_item_2, parent, false);
            }

            TranslationRecord record = records.get(position);

            TextView text1 = convertView.findViewById(android.R.id.text1);
            TextView text2 = convertView.findViewById(android.R.id.text2);

            // 翻译结果文本（大号）
            text1.setText(record.getResultText());
            text1.setTextColor(0xFFFFFFFF);
            text1.setTextSize(18);

            // 时间 + 帧序号（小号灰色）
            String timeStr = timeFormat.format(new Date(record.getTimestamp()));
            text2.setText("#" + record.getFrameIdx() + "  " + timeStr);
            text2.setTextColor(0xFFAAAAAA);
            text2.setTextSize(12);

            convertView.setPadding(16, 12, 16, 12);
            convertView.setBackgroundColor(0xFF222222);

            return convertView;
        }
    }
}
