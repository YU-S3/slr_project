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
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Size;
import android.util.Rational;
import android.view.Surface;
import android.view.TextureView;
import androidx.annotation.NonNull;
import androidx.core.content.ContextCompat;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.List;

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
    private MediaRecorder mediaRecorder;
    private boolean isRecording = false;
    private RecordingListener recordingListener;
    private File outputVideoFile;
    private Size previewSize;
    private int sensorOrientation;
    public interface RecordingListener {
        void onRecordingStarted();
        void onRecordingStopped(String filePath);
        void onError(String error);
    }

    public CameraHelper(Context context, TextureView textureView) {
        this.context = context;
        this.textureView = textureView;
        this.cameraManager = (CameraManager) context.getSystemService(Context.CAMERA_SERVICE);
        this.cameraId = getBackCameraId();
        Log.d(TAG, "CameraHelper initialized with cameraId: " + cameraId);
    }

    public void setRecordingListener(RecordingListener listener) {
        this.recordingListener = listener;
    }

    private String getBackCameraId() {
        try {
            String[] cameraIds = cameraManager.getCameraIdList();
            Log.d(TAG, "Available cameras: " + Arrays.toString(cameraIds));

            for (String cameraId : cameraIds) {
                CameraCharacteristics characteristics = cameraManager.getCameraCharacteristics(cameraId);
                Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_BACK) {
                    Log.d(TAG, "Found back camera: " + cameraId);
                    return cameraId;
                }
            }

            // 如果没有后置摄像头，使用第一个可用的摄像头
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
                return new Size(1920, 1080); // 默认返回1080p
            }

            // 获取摄像头支持的输出尺寸
            Size[] outputSizes = map.getOutputSizes(SurfaceTexture.class);
            if (outputSizes == null || outputSizes.length == 0) {
                Log.e(TAG, "No output sizes available");
                return new Size(1920, 1080);
            }

            int textureWidth = textureView.getWidth();
            int textureHeight = textureView.getHeight();

            Log.d(TAG, "TextureView size: " + textureWidth + "x" + textureHeight);

            // 如果TextureView还没有测量完成，使用默认尺寸
            if (textureWidth == 0 || textureHeight == 0) {
                textureWidth = 1920;
                textureHeight = 1080;
            }

            // 获取摄像头传感器方向
            sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
            Log.d(TAG, "Sensor orientation: " + sensorOrientation);

            // 是否需要交换宽高（针对前置摄像头）
            boolean swapDimensions = false;
            int displayRotation = 0; // 假设竖屏

            switch (sensorOrientation) {
                case 90:
                case 270:
                    swapDimensions = true;
                    break;
            }

            int displayWidth = textureWidth;
            int displayHeight = textureHeight;

            if (swapDimensions) {
                displayWidth = textureHeight;
                displayHeight = textureWidth;
            }

            Log.d(TAG, "Adjusted display size: " + displayWidth + "x" + displayHeight);

            // 选择最匹配的预览尺寸
            Size optimalSize = chooseOptimalSize(outputSizes, displayWidth, displayHeight);
            Log.d(TAG, "Selected preview size: " + optimalSize.getWidth() + "x" + optimalSize.getHeight());

            return optimalSize;

        } catch (CameraAccessException e) {
            Log.e(TAG, "Error choosing optimal size", e);
            return new Size(1920, 1080);
        }
    }

    private Size chooseOptimalSize(Size[] choices, int width, int height) {
        // 收集所有长宽比大于1的尺寸（横屏模式）
        List<Size> bigEnough = new ArrayList<>();
        List<Size> notBigEnough = new ArrayList<>();

        int w = width;
        int h = height;

        // 对于竖屏应用，我们通常想要竖屏比例的预览
        // 但摄像头传感器通常是横屏的，所以需要调整

        for (Size option : choices) {
            if (option.getWidth() <= 1920 && option.getHeight() <= 1920 &&
                    option.getWidth() >= 640 && option.getHeight() >= 480) {

                float aspectRatio = (float) option.getWidth() / option.getHeight();
                Log.d(TAG, "Available size: " + option.getWidth() + "x" + option.getHeight() +
                        " aspect: " + aspectRatio);

                // 根据传感器方向调整判断逻辑
                if (sensorOrientation == 90 || sensorOrientation == 270) {
                    // 传感器是横屏的，选择接近16:9或4:3的比例
                    if (Math.abs(aspectRatio - 16f/9f) < 0.2 || Math.abs(aspectRatio - 4f/3f) < 0.2) {
                        if (option.getHeight() >= h && option.getWidth() >= w) {
                            bigEnough.add(option);
                        } else {
                            notBigEnough.add(option);
                        }
                    }
                } else {
                    // 传感器是竖屏的（少见）
                    if (option.getWidth() >= w && option.getHeight() >= h) {
                        bigEnough.add(option);
                    } else {
                        notBigEnough.add(option);
                    }
                }
            }
        }

        // 优先选择足够大的尺寸中最小的
        if (bigEnough.size() > 0) {
            return Collections.min(bigEnough, new CompareSizesByArea());
        } else if (notBigEnough.size() > 0) {
            // 如果没有足够大的，选择最大的
            return Collections.max(notBigEnough, new CompareSizesByArea());
        } else {
            // 默认返回第一个可用的尺寸
            Log.e(TAG, "Couldn't find any suitable preview size");
            return choices[0];
        }
    }

    // 调整TextureView的变换矩阵以适应预览比例
    private void configureTransform(int viewWidth, int viewHeight) {
        if (textureView == null || previewSize == null) {
            return;
        }

        Matrix matrix = new Matrix();
        RectF viewRect = new RectF(0, 0, viewWidth, viewHeight);
        RectF bufferRect = new RectF(0, 0, previewSize.getHeight(), previewSize.getWidth());

        float centerX = viewRect.centerX();
        float centerY = viewRect.centerY();

        if (Surface.ROTATION_90 == 0 || Surface.ROTATION_270 == 0) {
            bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY());
            matrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.FILL);
        }

        float scale = Math.max(
                (float) viewHeight / previewSize.getHeight(),
                (float) viewWidth / previewSize.getWidth());

        matrix.postScale(scale, scale, centerX, centerY);

        // 考虑传感器方向
        matrix.postRotate(0, centerX, centerY);

        textureView.setTransform(matrix);
    }

    static class CompareSizesByArea implements Comparator<Size> {
        @Override
        public int compare(Size lhs, Size rhs) {
            return Long.signum((long) lhs.getWidth() * lhs.getHeight() -
                    (long) rhs.getWidth() * rhs.getHeight());
        }
    }

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
            if (recordingListener != null) {
                recordingListener.onError("切换摄像头失败");
            }
        }
    }

    private String getFrontCameraId() {
        try {
            for (String cameraId : cameraManager.getCameraIdList()) {
                CameraCharacteristics characteristics = cameraManager.getCameraCharacteristics(cameraId);
                Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                    return cameraId;
                }
            }
        } catch (CameraAccessException e) {
            Log.e(TAG, "Error getting front camera", e);
        }
        return null;
    }

    public void openCamera() {
        if (cameraId == null) {
            Log.e(TAG, "No camera ID available");
            if (recordingListener != null) {
                recordingListener.onError("没有可用的摄像头");
            }
            return;
        }

        try {
            if (ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA)
                    != PackageManager.PERMISSION_GRANTED) {
                Log.e(TAG, "Camera permission not granted");
                if (recordingListener != null) {
                    recordingListener.onError("没有摄像头权限");
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
                    if (recordingListener != null) {
                        recordingListener.onError("摄像头连接断开");
                    }
                }

                @Override
                public void onError(@NonNull CameraDevice camera, int error) {
                    Log.e(TAG, "Camera error: " + error);
                    closeCameraSession();

                    String errorMsg = getCameraErrorMessage(error);
                    if (recordingListener != null) {
                        recordingListener.onError(errorMsg);
                    }
                }
            }, backgroundHandler);
        } catch (Exception e) {
            Log.e(TAG, "Error opening camera", e);
            if (recordingListener != null) {
                recordingListener.onError("打开摄像头失败: " + e.getMessage());
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

            // 选择最优的预览尺寸
            previewSize = chooseOptimalSize();

            // 设置TextureView的缓冲区大小
            texture.setDefaultBufferSize(previewSize.getWidth(), previewSize.getHeight());

            // 调整TextureView的变换以适应预览比例
            int viewWidth = textureView.getWidth();
            int viewHeight = textureView.getHeight();
            if (viewWidth > 0 && viewHeight > 0) {
                configureTransform(viewWidth, viewHeight);
            }

            Surface surface = new Surface(texture);

            CaptureRequest.Builder previewRequestBuilder =
                    cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            previewRequestBuilder.addTarget(surface);

            // 使用自动模式
            previewRequestBuilder.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO);
            previewRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);

            cameraDevice.createCaptureSession(Collections.singletonList(surface),
                    new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(@NonNull CameraCaptureSession session) {
                            Log.d(TAG, "Camera preview session configured");
                            if (cameraDevice == null) {
                                return;
                            }

                            captureSession = session;
                            try {
                                session.setRepeatingRequest(previewRequestBuilder.build(), null, backgroundHandler);
                                Log.d(TAG, "Camera preview started successfully");
                            } catch (CameraAccessException e) {
                                Log.e(TAG, "Error starting camera preview", e);
                            }
                        }

                        @Override
                        public void onConfigureFailed(@NonNull CameraCaptureSession session) {
                            Log.e(TAG, "Camera preview session configuration failed");
                            if (recordingListener != null) {
                                recordingListener.onError("摄像头预览配置失败");
                            }
                        }
                    }, backgroundHandler);
        } catch (CameraAccessException e) {
            Log.e(TAG, "Error creating camera preview", e);
            if (recordingListener != null) {
                recordingListener.onError("创建摄像头预览失败");
            }
        }
    }


    public boolean startRecording() {
        if (isRecording) {
            Log.w(TAG, "Already recording");
            return false;
        }

        if (cameraDevice == null) {
            Log.e(TAG, "Camera device is null, cannot start recording");
            if (recordingListener != null) {
                recordingListener.onError("摄像头未就绪");
            }
            return false;
        }

        try {
            // 检查权限
            if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO)
                    != PackageManager.PERMISSION_GRANTED) {
                Log.e(TAG, "Audio recording permission not granted");
                if (recordingListener != null) {
                    recordingListener.onError("没有录音权限");
                }
                return false;
            }

            if (!checkStoragePermission()) {
                Log.e(TAG, "Storage permission not granted");
                if (recordingListener != null) {
                    recordingListener.onError("没有存储权限");
                }
                return false;
            }

            // 关闭现有的预览会话
            if (captureSession != null) {
                captureSession.close();
                captureSession = null;
            }

            // 等待会话完全关闭
            Thread.sleep(100);

            // 设置MediaRecorder
            mediaRecorder = new MediaRecorder();
            setupMediaRecorder();

            // 准备录制表面
            SurfaceTexture texture = textureView.getSurfaceTexture();
            if (texture == null) {
                Log.e(TAG, "SurfaceTexture is null");
                if (recordingListener != null) {
                    recordingListener.onError("纹理视图未就绪");
                }
                return false;
            }

            int width = Math.max(textureView.getWidth(), 640);
            int height = Math.max(textureView.getHeight(), 480);
            texture.setDefaultBufferSize(width, height);

            Surface previewSurface = new Surface(texture);
            Surface recorderSurface = mediaRecorder.getSurface();

            List<Surface> surfaces = new ArrayList<>();
            surfaces.add(previewSurface);
            surfaces.add(recorderSurface);

            // 创建录制请求
            CaptureRequest.Builder recordingRequestBuilder =
                    cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_RECORD);
            recordingRequestBuilder.addTarget(previewSurface);
            recordingRequestBuilder.addTarget(recorderSurface);
            recordingRequestBuilder.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO);

            // 创建录制会话
            cameraDevice.createCaptureSession(surfaces,
                    new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(@NonNull CameraCaptureSession session) {
                            Log.d(TAG, "Recording session configured successfully");
                            captureSession = session;
                            try {
                                mediaRecorder.start();
                                session.setRepeatingRequest(recordingRequestBuilder.build(), null, backgroundHandler);
                                isRecording = true;
                                Log.d(TAG, "Recording started successfully");

                                if (recordingListener != null) {
                                    recordingListener.onRecordingStarted();
                                }
                            } catch (Exception e) {
                                Log.e(TAG, "Error starting recording in session", e);
                                isRecording = false;
                                if (recordingListener != null) {
                                    recordingListener.onError("开始录制失败: " + e.getMessage());
                                }
                                // 发生错误时恢复到预览模式
                                createCameraPreview();
                            }
                        }

                        @Override
                        public void onConfigureFailed(@NonNull CameraCaptureSession session) {
                            Log.e(TAG, "Recording session configuration failed");
                            isRecording = false;
                            if (recordingListener != null) {
                                recordingListener.onError("录制会话配置失败");
                            }
                            // 恢复到预览模式
                            createCameraPreview();
                        }
                    }, backgroundHandler);

            return true;

        } catch (Exception e) {
            Log.e(TAG, "Error starting recording", e);
            isRecording = false;
            if (recordingListener != null) {
                recordingListener.onError("录制启动异常: " + e.getMessage());
            }
            // 恢复到预览模式
            createCameraPreview();
            return false;
        }
    }

    private boolean checkStoragePermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                return ContextCompat.checkSelfPermission(context, Manifest.permission.READ_MEDIA_VIDEO) == PackageManager.PERMISSION_GRANTED &&
                        ContextCompat.checkSelfPermission(context, Manifest.permission.READ_MEDIA_IMAGES) == PackageManager.PERMISSION_GRANTED;
            } else {
                return ContextCompat.checkSelfPermission(context, Manifest.permission.READ_MEDIA_VIDEO) == PackageManager.PERMISSION_GRANTED;
            }
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            return true;
        } else {
            return ContextCompat.checkSelfPermission(context, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
        }
    }

    private void setupMediaRecorder() throws IOException {
        // 完全重置MediaRecorder
        if (mediaRecorder != null) {
            try {
                mediaRecorder.reset();
            } catch (Exception e) {
                Log.w(TAG, "Error resetting MediaRecorder, creating new instance");
                mediaRecorder = new MediaRecorder();
            }
        } else {
            mediaRecorder = new MediaRecorder();
        }

        // 使用最兼容的设置
        mediaRecorder.setAudioSource(MediaRecorder.AudioSource.CAMCORDER); // 使用CAMCORDER而不是MIC，兼容性更好
        mediaRecorder.setVideoSource(MediaRecorder.VideoSource.SURFACE);
        mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);

        // 设置编码器 - 使用最兼容的设置
        mediaRecorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264);
        mediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);

        // 使用较低的参数确保兼容性
        mediaRecorder.setVideoEncodingBitRate(2000000); // 2 Mbps
        mediaRecorder.setVideoFrameRate(24); // 24 fps
        mediaRecorder.setVideoSize(1280, 720); // 720p 标准分辨率

        // 设置输出文件
        outputVideoFile = createVideoFile();
        mediaRecorder.setOutputFile(outputVideoFile.getAbsolutePath());

        // 设置方向（可选）
        mediaRecorder.setOrientationHint(90);

        try {
            mediaRecorder.prepare();
            Log.d(TAG, "MediaRecorder prepared successfully for file: " + outputVideoFile.getAbsolutePath());
        } catch (IOException e) {
            Log.e(TAG, "MediaRecorder prepare failed", e);
            throw e;
        } catch (IllegalStateException e) {
            Log.e(TAG, "MediaRecorder illegal state during prepare", e);
            throw new IOException("MediaRecorder状态异常: " + e.getMessage());
        }
    }

    private File createVideoFile() {
        File mediaDir;

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            mediaDir = new File(context.getExternalFilesDir(Environment.DIRECTORY_MOVIES), "CameraRecorder");
        } else {
            if (Environment.MEDIA_MOUNTED.equals(Environment.getExternalStorageState())) {
                mediaDir = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES), "CameraRecorder");
            } else {
                mediaDir = new File(context.getFilesDir(), "CameraRecorder");
            }
        }

        if (!mediaDir.exists() && !mediaDir.mkdirs()) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                mediaDir = context.getExternalFilesDir(Environment.DIRECTORY_MOVIES);
            } else {
                mediaDir = context.getFilesDir();
            }
        }

        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        File videoFile = new File(mediaDir, "VID_" + timeStamp + ".mp4");
        Log.d(TAG, "Video file path: " + videoFile.getAbsolutePath());

        return videoFile;
    }

    public void stopRecording() {
        if (!isRecording) {
            Log.w(TAG, "Not recording, cannot stop");
            return;
        }

        try {
            Log.d(TAG, "Stopping recording");
            isRecording = false;

            if (mediaRecorder != null) {
                try {
                    mediaRecorder.stop();
                } catch (Exception e) {
                    Log.e(TAG, "Error stopping MediaRecorder", e);
                }
                mediaRecorder.reset();
                mediaRecorder.release();
                mediaRecorder = null;
            }

            String filePath = outputVideoFile != null ? outputVideoFile.getAbsolutePath() : "未知路径";
            Log.d(TAG, "Recording stopped, file: " + filePath);

            if (recordingListener != null) {
                recordingListener.onRecordingStopped(filePath);
            }

            // 关闭当前会话并重新创建预览
            if (captureSession != null) {
                captureSession.close();
                captureSession = null;
            }

            // 等待一段时间再重新创建预览
            if (backgroundHandler != null) {
                backgroundHandler.postDelayed(this::createCameraPreview, 200);
            }

        } catch (Exception e) {
            Log.e(TAG, "Error stopping recording", e);
            if (recordingListener != null) {
                recordingListener.onError("停止录制失败: " + e.getMessage());
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

        if (mediaRecorder != null) {
            try {
                mediaRecorder.release();
            } catch (Exception e) {
                Log.e(TAG, "Error releasing MediaRecorder", e);
            }
            mediaRecorder = null;
        }
    }

    public void closeCamera() {
        Log.d(TAG, "Closing camera");
        closeCameraSession();
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
            Log.d(TAG, "SurfaceTexture size changed: " + width + "x" + height);
        }

        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
            Log.d(TAG, "SurfaceTexture destroyed");
            return false;
        }

        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture surface) {
            // 纹理更新
        }
    };

    public boolean isRecording() {
        return isRecording;
    }
}