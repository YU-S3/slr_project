package com.example.camerarecorder;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.annotation.TargetApi;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.TextureView;
import androidx.annotation.NonNull;
import androidx.core.content.ContextCompat;
import android.view.WindowManager;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * 相机助手类
 * <p>
 * 负责管理 Camera2 预览生命周期：打开/关闭相机、创建预览会话、切换摄像头。
 * <p>
 * ⚠ 改造说明：已移除 MediaRecorder 录制功能，仅保留实时预览。
 * 帧采集通过 MainActivity 定时调用 TextureView.getBitmap() 实现。
 */
@TargetApi(21)
public class CameraHelper {

    private static final String TAG = "CameraHelper";

    private final Context context;
    private TextureView textureView;
    private CameraManager cameraManager;
    private CameraDevice cameraDevice;
    private CameraCaptureSession captureSession;
    private Handler backgroundHandler;
    private HandlerThread backgroundThread;
    private String cameraId;
    private boolean isRecording = false;
    private CameraListener cameraListener;
    private Size previewSize;
    private int sensorOrientation;

    /** 相机事件监听器 */
    public interface CameraListener {
        void onCameraOpened();

        void onCameraClosed();

        void onError(String error);
    }

    public CameraHelper(Context context, TextureView textureView) {
        this.context = context;
        this.textureView = textureView;
        this.cameraManager = (CameraManager) context.getSystemService(Context.CAMERA_SERVICE);
        this.cameraId = getBackCameraId();
        Log.d(TAG, "CameraHelper initialized with cameraId: " + cameraId);
    }

    public void setCameraListener(CameraListener listener) {
        this.cameraListener = listener;
    }

    private String getBackCameraId() {
        try {
            String[] cameraIds = cameraManager.getCameraIdList();
            Log.d(TAG, "Available cameras: " + Arrays.toString(cameraIds));
            for (String id : cameraIds) {
                CameraCharacteristics characteristics = cameraManager.getCameraCharacteristics(id);
                Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_BACK) {
                    Log.d(TAG, "Found back camera: " + id);
                    return id;
                }
            }
            if (cameraIds.length > 0) {
                Log.d(TAG, "No back camera found, using first available: " + cameraIds[0]);
                return cameraIds[0];
            }
        } catch (CameraAccessException e) {
            Log.e(TAG, "Error accessing camera characteristics", e);
        }
        Log.e(TAG, "No camera found!");
        return null;
    }

    private Size chooseOptimalSize() {
        try {
            CameraCharacteristics characteristics = cameraManager.getCameraCharacteristics(cameraId);
            StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            if (map == null) {
                Log.e(TAG, "StreamConfigurationMap is null");
                return new Size(1920, 1080);
            }
            Size[] outputSizes = map.getOutputSizes(SurfaceTexture.class);
            if (outputSizes == null || outputSizes.length == 0) {
                Log.e(TAG, "No output sizes available");
                return new Size(1920, 1080);
            }
            int textureWidth = textureView.getWidth();
            int textureHeight = textureView.getHeight();
            if (textureWidth == 0 || textureHeight == 0) {
                textureWidth = 1920;
                textureHeight = 1080;
            }
            sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
            Log.d(TAG, "Sensor orientation: " + sensorOrientation);

            Size optimalSize = chooseOptimalSize(outputSizes, textureWidth, textureHeight);
            Log.d(TAG, "Selected preview size: " + optimalSize.getWidth() + "x" + optimalSize.getHeight());
            return optimalSize;
        } catch (CameraAccessException e) {
            Log.e(TAG, "Error choosing optimal size", e);
            return new Size(1920, 1080);
        }
    }

    private Size chooseOptimalSize(Size[] choices, int width, int height) {
        List<Size> bigEnough = new ArrayList<>();
        List<Size> notBigEnough = new ArrayList<>();
        int w = width;
        int h = height;
        for (Size option : choices) {
            if (option.getWidth() <= 1920 && option.getHeight() <= 1920 &&
                    option.getWidth() >= 640 && option.getHeight() >= 480) {
                float aspectRatio = (float) option.getWidth() / option.getHeight();
                if (sensorOrientation == 90 || sensorOrientation == 270) {
                    if (Math.abs(aspectRatio - 16f / 9f) < 0.2 || Math.abs(aspectRatio - 4f / 3f) < 0.2) {
                        if (option.getHeight() >= h && option.getWidth() >= w) {
                            bigEnough.add(option);
                        } else {
                            notBigEnough.add(option);
                        }
                    }
                } else {
                    if (option.getWidth() >= w && option.getHeight() >= h) {
                        bigEnough.add(option);
                    } else {
                        notBigEnough.add(option);
                    }
                }
            }
        }
        if (bigEnough.size() > 0) {
            return Collections.min(bigEnough, new CompareSizesByArea());
        } else if (notBigEnough.size() > 0) {
            return Collections.max(notBigEnough, new CompareSizesByArea());
        } else {
            Log.e(TAG, "Couldn't find any suitable preview size");
            return choices[0];
        }
    }

    static class CompareSizesByArea implements Comparator<Size> {
        @Override
        public int compare(Size lhs, Size rhs) {
            return Long.signum((long) lhs.getWidth() * lhs.getHeight() -
                    (long) rhs.getWidth() * rhs.getHeight());
        }
    }

    private void configureTransform(int viewWidth, int viewHeight) {
        if (textureView == null || previewSize == null)
            return;
        Matrix matrix = new Matrix();
        RectF viewRect = new RectF(0, 0, viewWidth, viewHeight);
        RectF bufferRect = new RectF(0, 0, previewSize.getWidth(), previewSize.getHeight());
        float centerX = viewRect.centerX();
        float centerY = viewRect.centerY();
        int rotation = getWindowManagerRotation(textureView.getContext());
        int rotate = (sensorOrientation + rotation * 90) % 360;
        boolean isFrontCamera = false;
        try {
            CameraCharacteristics characteristics = cameraManager.getCameraCharacteristics(cameraId);
            Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
            isFrontCamera = (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT);
        } catch (CameraAccessException e) {
            Log.e(TAG, "Error getting camera characteristics", e);
        }
        if (isFrontCamera) {
            rotate = (rotate + 180) % 360;
        }
        matrix.postRotate(rotate, centerX, centerY);
        if (rotate == 90 || rotate == 270) {
            bufferRect = new RectF(0, 0, previewSize.getHeight(), previewSize.getWidth());
            bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY());
            matrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.FILL);
        } else if (rotate == 0 || rotate == 180) {
            bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY());
            matrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.FILL);
        }
        textureView.setTransform(matrix);
    }

    /** 切换前后摄像头 */
    public void switchCamera() {
        try {
            closeCameraSession();
            String[] cameraIds = cameraManager.getCameraIdList();
            if (cameraIds.length > 1) {
                String newCameraId = cameraId.equals(getBackCameraId()) ? getFrontCameraId() : getBackCameraId();
                if (newCameraId != null) {
                    cameraId = newCameraId;
                    Log.d(TAG, "Switching to camera: " + cameraId);
                    backgroundHandler.postDelayed(this::openCamera, 500);
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "Error switching camera", e);
            if (cameraListener != null) {
                cameraListener.onError("切换摄像头失败");
            }
        }
    }

    private String getFrontCameraId() {
        try {
            for (String id : cameraManager.getCameraIdList()) {
                CameraCharacteristics characteristics = cameraManager.getCameraCharacteristics(id);
                Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                    return id;
                }
            }
        } catch (CameraAccessException e) {
            Log.e(TAG, "Error getting front camera", e);
        }
        return null;
    }

    /** 打开相机 */
    public void openCamera() {
        if (cameraId == null) {
            Log.e(TAG, "No camera ID available");
            if (cameraListener != null) {
                cameraListener.onError("没有可用的摄像头");
            }
            return;
        }
        try {
            if (ContextCompat.checkSelfPermission(context,
                    Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                Log.e(TAG, "Camera permission not granted");
                if (cameraListener != null) {
                    cameraListener.onError("没有摄像头权限");
                }
                return;
            }
            Log.d(TAG, "Opening camera: " + cameraId);
            cameraManager.openCamera(cameraId, new CameraDevice.StateCallback() {
                @Override
                public void onOpened(@NonNull CameraDevice camera) {
                    Log.d(TAG, "Camera opened successfully");
                    cameraDevice = camera;
                    createCameraPreview();
                }

                @Override
                public void onDisconnected(@NonNull CameraDevice camera) {
                    Log.w(TAG, "Camera disconnected");
                    closeCameraSession();
                    if (cameraListener != null) {
                        cameraListener.onError("摄像头连接断开");
                    }
                }

                @Override
                public void onError(@NonNull CameraDevice camera, int error) {
                    Log.e(TAG, "Camera error: " + error);
                    closeCameraSession();
                    String errorMsg = getCameraErrorMessage(error);
                    if (cameraListener != null) {
                        cameraListener.onError(errorMsg);
                    }
                }
            }, backgroundHandler);
        } catch (Exception e) {
            Log.e(TAG, "Error opening camera", e);
            if (cameraListener != null) {
                cameraListener.onError("打开摄像头失败: " + e.getMessage());
            }
        }
    }

    private String getCameraErrorMessage(int error) {
        switch (error) {
            case CameraDevice.StateCallback.ERROR_CAMERA_IN_USE:
                return "摄像头被其他应用占用";
            case CameraDevice.StateCallback.ERROR_MAX_CAMERAS_IN_USE:
                return "摄像头数量达到上限";
            case CameraDevice.StateCallback.ERROR_CAMERA_DISABLED:
                return "摄像头被禁用";
            case CameraDevice.StateCallback.ERROR_CAMERA_DEVICE:
                return "摄像头设备故障";
            case CameraDevice.StateCallback.ERROR_CAMERA_SERVICE:
                return "摄像头服务故障";
            default:
                return "摄像头打开失败，错误代码: " + error;
        }
    }

    private void createCameraPreview() {
        if (cameraDevice == null) {
            Log.e(TAG, "Camera device is null, cannot create preview");
            return;
        }
        try {
            SurfaceTexture texture = textureView.getSurfaceTexture();
            if (texture == null) {
                Log.e(TAG, "SurfaceTexture is null");
                return;
            }
            previewSize = chooseOptimalSize();
            texture.setDefaultBufferSize(previewSize.getWidth(), previewSize.getHeight());

            int viewWidth = textureView.getWidth();
            int viewHeight = textureView.getHeight();
            if (viewWidth > 0 && viewHeight > 0) {
                configureTransform(viewWidth, viewHeight);
            }

            Surface surface = new Surface(texture);

            CaptureRequest.Builder previewRequestBuilder = cameraDevice
                    .createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            previewRequestBuilder.addTarget(surface);
            previewRequestBuilder.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO);
            previewRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE,
                    CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);

            cameraDevice.createCaptureSession(Collections.singletonList(surface),
                    new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(@NonNull CameraCaptureSession session) {
                            Log.d(TAG, "Camera preview session configured");
                            if (cameraDevice == null)
                                return;
                            captureSession = session;
                            try {
                                session.setRepeatingRequest(previewRequestBuilder.build(), null, backgroundHandler);
                                Log.d(TAG, "Camera preview started successfully");
                                if (textureView.getWidth() > 0 && textureView.getHeight() > 0) {
                                    configureTransform(textureView.getWidth(), textureView.getHeight());
                                }
                                if (cameraListener != null) {
                                    cameraListener.onCameraOpened();
                                }
                            } catch (CameraAccessException e) {
                                Log.e(TAG, "Error creating camera preview", e);
                                if (cameraListener != null) {
                                    cameraListener.onError("创建摄像头预览失败");
                                }
                            }
                        }

                        @Override
                        public void onConfigureFailed(@NonNull CameraCaptureSession session) {
                            Log.e(TAG, "Camera preview session configuration failed");
                            if (cameraListener != null) {
                                cameraListener.onError("摄像头预览配置失败");
                            }
                        }
                    }, backgroundHandler);
        } catch (CameraAccessException e) {
            Log.e(TAG, "Error creating camera preview", e);
            if (cameraListener != null) {
                cameraListener.onError("创建摄像头预览失败");
            }
        }
    }

    private void closeCameraSession() {
        Log.d(TAG, "Closing camera session");
        isRecording = false;
        if (captureSession != null) {
            captureSession.close();
            captureSession = null;
        }
        if (cameraDevice != null) {
            cameraDevice.close();
            cameraDevice = null;
        }
    }

    public void closeCamera() {
        Log.d(TAG, "Closing camera");
        closeCameraSession();
        if (cameraListener != null) {
            cameraListener.onCameraClosed();
        }
    }

    public void startBackgroundThread() {
        if (backgroundThread == null) {
            backgroundThread = new HandlerThread("CameraBackground");
            backgroundThread.start();
            backgroundHandler = new Handler(backgroundThread.getLooper());
            Log.d(TAG, "Background thread started");
        }
    }

    public void stopBackgroundThread() {
        if (backgroundThread != null) {
            Log.d(TAG, "Stopping background thread");
            closeCameraSession();
            backgroundThread.quitSafely();
            try {
                backgroundThread.join();
                backgroundThread = null;
                backgroundHandler = null;
            } catch (InterruptedException e) {
                Log.e(TAG, "Error stopping background thread", e);
            }
        }
    }

    public TextureView.SurfaceTextureListener getSurfaceTextureListener() {
        return surfaceTextureListener;
    }

    private TextureView.SurfaceTextureListener surfaceTextureListener = new TextureView.SurfaceTextureListener() {
        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
            Log.d(TAG, "SurfaceTexture available: " + width + "x" + height);
            openCamera();
        }

        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {
            configureTransform(width, height);
        }

        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
            return true;
        }

        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture surface) {
            // 纹理更新 - 不需要处理
        }
    };

    public boolean isRecording() {
        return isRecording;
    }

    /** 获取当前摄像头预览尺寸 */
    public Size getPreviewSize() {
        return previewSize;
    }

    /** 获取摄像头传感器旋转角度 */
    public int getSensorOrientation() {
        return sensorOrientation;
    }

    /** 获取当前屏幕旋转值（与 getWindowManagerRotation 相同逻辑） */
    public int getDisplayRotation() {
        return getWindowManagerRotation(context);
    }

    @SuppressWarnings("deprecation")
    private int getWindowManagerRotation(Context context) {
        WindowManager windowManager = (WindowManager) context.getSystemService(Context.WINDOW_SERVICE);
        int rotation;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            rotation = context.getDisplay().getRotation();
        } else {
            rotation = windowManager.getDefaultDisplay().getRotation();
        }
        switch (rotation) {
            case Surface.ROTATION_0:
                return 0;
            case Surface.ROTATION_90:
                return 1;
            case Surface.ROTATION_180:
                return 2;
            case Surface.ROTATION_270:
                return 3;
            default:
                return 0;
        }
    }
}
