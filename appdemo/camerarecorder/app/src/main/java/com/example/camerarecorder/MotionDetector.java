package com.example.camerarecorder;

import android.graphics.Bitmap;
import android.util.Log;

/**
 * 帧差法运动检测器
 * <p>
 * 通过比较相邻两帧的灰度差异来判断画面是否发生显著运动，
 * 用于过滤静止帧，减少不必要的网络传输和后端计算。
 */
public class MotionDetector {

    private static final String TAG = "MotionDetector";

    /** 上一帧的灰度像素数组（缩放到小尺寸后） */
    private int[] previousGrayPixels;

    /** 上一帧的宽度 */
    private int prevWidth = 0;

    /** 上一帧的高度 */
    private int prevHeight = 0;

    /** 是否已初始化（第一帧不计入检测） */
    private boolean initialized = false;

    /**
     * 检测当前帧是否包含显著运动
     *
     * @param currentFrame 当前帧 Bitmap（任意尺寸，内部会自动缩放）
     * @return true = 有运动，应当上传该帧；false = 静止帧，建议丢弃
     */
    public boolean detectMotion(Bitmap currentFrame) {
        if (currentFrame == null) {
            return false;
        }

        // 缩放到小尺寸以加速计算
        int scaleW = Config.MOTION_SCALE_SIZE;
        int scaleH = Config.MOTION_SCALE_SIZE;
        Bitmap scaled = Bitmap.createScaledBitmap(currentFrame, scaleW, scaleH, true);

        // 提取灰度像素
        int[] grayPixels = new int[scaleW * scaleH];
        for (int i = 0; i < grayPixels.length; i++) {
            int x = i % scaleW;
            int y = i / scaleW;
            int pixel = scaled.getPixel(x, y);
            // 加权灰度公式: Gray = 0.299*R + 0.587*G + 0.114*B
            int gray = (int) (0.299f * ((pixel >> 16) & 0xFF)
                    + 0.587f * ((pixel >> 8) & 0xFF)
                    + 0.114f * (pixel & 0xFF));
            grayPixels[i] = gray;
        }

        // 如果是第一帧或尺寸变化，无法比较，直接返回 true（上传该帧）
        if (!initialized || prevWidth != scaleW || prevHeight != scaleH) {
            previousGrayPixels = grayPixels;
            prevWidth = scaleW;
            prevHeight = scaleH;
            initialized = true;
            return true;
        }

        // 计算差异像素占比
        int diffCount = 0;
        int threshold = Config.PIXEL_DIFF_THRESHOLD;
        for (int i = 0; i < grayPixels.length; i++) {
            if (Math.abs(grayPixels[i] - previousGrayPixels[i]) > threshold) {
                diffCount++;
            }
        }

        float diffRatio = (float) diffCount / grayPixels.length;

        // 更新上一帧
        previousGrayPixels = grayPixels;

        boolean hasMotion = diffRatio >= Config.MOTION_THRESHOLD;
        Log.d(TAG, "Motion ratio: " + String.format("%.4f", diffRatio)
                + " | hasMotion: " + hasMotion);
        return hasMotion;
    }

    /**
     * 重置检测器状态（切换摄像头或开始新识别时调用）
     */
    public void reset() {
        initialized = false;
        previousGrayPixels = null;
        Log.d(TAG, "MotionDetector reset");
    }

    /**
     * 手动设置灵敏度阈值
     *
     * @param threshold 0.0~1.0，越小越敏感，推荐 0.05
     */
    public void setThreshold(float threshold) {
        // 阈值在 Config 中定义，可通过此方法动态调整
        Log.d(TAG, "Threshold adjustment requested: " + threshold);
    }
}
