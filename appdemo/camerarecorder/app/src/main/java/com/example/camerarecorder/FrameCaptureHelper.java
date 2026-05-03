package com.example.camerarecorder;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.util.Base64;
import android.util.Log;
import android.view.TextureView;

import java.io.ByteArrayOutputStream;

/**
 * 帧捕获与编码工具类
 * <p>
 * 负责从 TextureView 捕获帧 → 去拉伸（还原摄像头原始比例）→ 缩放 → JPEG 压缩 → Base64 编码
 */
public class FrameCaptureHelper {

    private static final String TAG = "FrameCaptureHelper";

    /**
     * 从 TextureView 捕获当前显示帧（不进行去拉伸处理，保留 TextureView 当前显示）
     * <p>
     * 如果摄像头预览被 {@link android.graphics.Matrix.ScaleToFit#FILL} 拉伸，返回的图像也是拉伸的。
     * 建议使用 {@link #captureFrame(TextureView, int, int, int, int)} 传入摄像头参数进行去拉伸。
     *
     * @param textureView 相机预览的 TextureView
     * @return 当前帧 Bitmap，失败时返回 null
     */
    public Bitmap captureFrame(TextureView textureView) {
        if (textureView == null || !textureView.isAvailable()) {
            Log.w(TAG, "TextureView not available for capture");
            return null;
        }

        // 获取当前显示的 Bitmap
        Bitmap original = textureView.getBitmap();
        if (original == null) {
            Log.w(TAG, "Failed to get bitmap from TextureView");
            return null;
        }

        // 如果需要缩放，等比缩放到 MAX 尺寸内
        return scaleToMaxSize(original, Config.MAX_FRAME_WIDTH, Config.MAX_FRAME_HEIGHT);
    }

    /**
     * 从 TextureView 捕获帧，并根据摄像头参数还原正确比例（去拉伸）
     * <p>
     * {@link com.example.camerarecorder.CameraHelper#configureTransform(int, int)}
     * 使用 {@link android.graphics.Matrix.ScaleToFit#FILL} 将摄像头画面拉伸填满 TextureView，
     * 导致 {@link TextureView#getBitmap()} 返回的图像被拉伸。此方法通过中心裁剪还原为摄像头原始比例。
     *
     * @param textureView       相机预览的 TextureView
     * @param previewWidth      摄像头预览宽度（来自 CameraHelper.getPreviewSize().getWidth()）
     * @param previewHeight     摄像头预览高度（来自
     *                          CameraHelper.getPreviewSize().getHeight()）
     * @param sensorOrientation 摄像头传感器方向（来自 CameraHelper.getSensorOrientation()）
     * @param displayRotation   屏幕旋转系数（0/1/2/3，来自 CameraHelper.getDisplayRotation()）
     * @return 去拉伸后的 Bitmap，失败时返回 null
     */
    public Bitmap captureFrame(TextureView textureView,
            int previewWidth, int previewHeight,
            int sensorOrientation, int displayRotation) {
        if (textureView == null || !textureView.isAvailable()) {
            Log.w(TAG, "TextureView not available for capture");
            return null;
        }

        Bitmap original = textureView.getBitmap();
        if (original == null) {
            Log.w(TAG, "Failed to get bitmap from TextureView");
            return null;
        }

        int viewW = original.getWidth();
        int viewH = original.getHeight();

        // 计算摄像头画面经旋转后的有效尺寸（摄像头坐标系 → 显示坐标系）
        int rotation = (sensorOrientation + displayRotation * 90) % 360;
        boolean isRotated = (rotation == 90 || rotation == 270);
        float camW = isRotated ? previewHeight : previewWidth;
        float camH = isRotated ? previewWidth : previewHeight;

        // 计算在 TextureView 上以 CENTER 方式显示时的有效显示区域
        // （FILL 把摄像头画面拉伸到整个 view，CENTER 则保持比例居中显示）
        float displayScale = Math.min((float) viewW / camW, (float) viewH / camH);
        int dispW = Math.round(camW * displayScale);
        int dispH = Math.round(camH * displayScale);

        // 居中裁剪，还原到摄像头原始宽高比
        int startX = (viewW - dispW) / 2;
        int startY = (viewH - dispH) / 2;

        Log.d(TAG, String.format(
                "De-stretch: view=%dx%d, cam(effective)=%.0fx%.0f, crop=(%d,%d)-%dx%d",
                viewW, viewH, camW, camH, startX, startY, dispW, dispH));

        Bitmap cropped = Bitmap.createBitmap(original, startX, startY, dispW, dispH);

        // 等比缩放到 MAX 尺寸内
        return scaleToMaxSize(cropped, Config.MAX_FRAME_WIDTH, Config.MAX_FRAME_HEIGHT);
    }

    /**
     * 将 Bitmap 等比缩放到不超过指定最大尺寸
     */
    private Bitmap scaleToMaxSize(Bitmap bitmap, int maxWidth, int maxHeight) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        // 如果尺寸在允许范围内，直接返回原图
        if (width <= maxWidth && height <= maxHeight) {
            return bitmap;
        }

        // 计算缩放比例（取较小比例以完全适配）
        float scale = Math.min(
                (float) maxWidth / width,
                (float) maxHeight / height);

        int newWidth = Math.round(width * scale);
        int newHeight = Math.round(height * scale);

        Log.d(TAG, "Scaling frame from " + width + "x" + height
                + " to " + newWidth + "x" + newHeight);

        return Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true);
    }

    /**
     * 将 Bitmap 压缩为 JPEG 并编码为 Base64 字符串
     *
     * @param bitmap  要编码的帧
     * @param quality JPEG 压缩质量（0-100），推荐 85
     * @return Base64 编码的 JPEG 字符串
     */
    public String compressToBase64(Bitmap bitmap, int quality) {
        if (bitmap == null) {
            return null;
        }

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        // 压缩为 JPEG 格式
        bitmap.compress(Bitmap.CompressFormat.JPEG, quality, baos);

        byte[] jpegBytes = baos.toByteArray();
        Log.d(TAG, "JPEG compressed size: " + (jpegBytes.length / 1024) + " KB");

        // Base64 编码（NO_WRAP = 不换行）
        return Base64.encodeToString(jpegBytes, Base64.NO_WRAP);
    }

    /**
     * 一站式方法：从 TextureView 捕获 → 压缩 → Base64
     *
     * @param textureView 相机预览
     * @return Base64 字符串，失败时返回 null
     */
    public String captureAndEncode(TextureView textureView) {
        Bitmap bitmap = captureFrame(textureView);
        if (bitmap == null) {
            return null;
        }
        return compressToBase64(bitmap, Config.JPEG_QUALITY);
    }

    /**
     * 解码 Base64 为 Bitmap（测试用，或用于显示捕获的图像）
     */
    public Bitmap decodeBase64ToBitmap(String base64) {
        if (base64 == null || base64.isEmpty()) {
            return null;
        }
        byte[] decodedBytes = Base64.decode(base64, Base64.NO_WRAP);
        return BitmapFactory.decodeByteArray(decodedBytes, 0, decodedBytes.length);
    }
}
