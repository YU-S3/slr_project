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
import android.view.WindowManager;


import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.List;


// 相机助手类，负责处理相机预览、录制等功能
public class CameraHelper {

    // 日志标签
    private static final String TAG = "CameraHelper";

    // 上下文对象
    private final Context context;

    // 纹理视图，用于显示相机预览
    private TextureView textureView;

    // 相机管理器，用于访问相机设备
    private CameraManager cameraManager;

    // 相机设备对象
    private CameraDevice cameraDevice;

    // 相机捕获会话
    private CameraCaptureSession captureSession;

    // 后台处理程序
    private Handler backgroundHandler;

    // 后台线程
    private HandlerThread backgroundThread;

    // 当前使用的相机ID
    private String cameraId;

    // 媒体录制器
    private MediaRecorder mediaRecorder;

    // 录制状态标志
    private boolean isRecording = false;

    // 录制监听器
    private RecordingListener recordingListener;

    // 输出视频文件
    private File outputVideoFile;

    // 预览尺寸
    private Size previewSize;

    // 传感器方向
    private int sensorOrientation;

    // 录制监听器接口
    public interface RecordingListener {
        void onRecordingStarted();   // 开始录制回调
        void onRecordingStopped(String filePath); // 停止录制回调
        void onError(String error);  // 错误回调
    }

    // 构造函数
    public CameraHelper(Context context, TextureView textureView) {
        this.context = context;
        this.textureView = textureView;
        // 获取系统相机服务管理器
        this.cameraManager = (CameraManager) context.getSystemService(Context.CAMERA_SERVICE);
        // 获取后置摄像头ID
        this.cameraId = getBackCameraId(); // 默认使用后置摄像头
        Log.d(TAG, "CameraHelper initialized with cameraId: " + cameraId);
    }

    // 设置录制监听器
    public void setRecordingListener(RecordingListener listener) {
        this.recordingListener = listener;
    }

    // 获取后置摄像头ID
    private String getBackCameraId() {
        try {
            // 获取所有可用相机ID列表
            String[] cameraIds = cameraManager.getCameraIdList();
            Log.d(TAG, "Available cameras: " + Arrays.toString(cameraIds));

            for (String cameraId : cameraIds) {
                // 获取相机特性信息
                CameraCharacteristics characteristics = cameraManager.getCameraCharacteristics(cameraId);
                // 获取镜头朝向信息
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

    // 选择最佳预览尺寸
// 选择最佳预览尺寸
private Size chooseOptimalSize() {
    try {
        // 获取相机特性信息
        CameraCharacteristics characteristics = cameraManager.getCameraCharacteristics(cameraId);
        // 获取缩放流配置映射表
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

        // 根据屏幕比例选择最合适的预览尺寸
        Size optimalSize = chooseOptimalSize(outputSizes, textureWidth, textureHeight);
        Log.d(TAG, "Selected preview size: " + optimalSize.getWidth() + "x" + optimalSize.getHeight());

        return optimalSize;

    } catch (CameraAccessException e) {
        Log.e(TAG, "Error choosing optimal size", e);
        return new Size(1920, 1080);
    }
}


    // 根据给定的尺寸选择最佳预览尺寸
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
// 调整TextureView的变换矩阵以适应预览比例
// 在 CameraHelper 类中替换 configureTransform 方法
// 替换整个 configureTransform 方法
private void configureTransform(int viewWidth, int viewHeight) {
    if (textureView == null || previewSize == null) {
        return;
    }

    Matrix matrix = new Matrix();
    RectF viewRect = new RectF(0, 0, viewWidth, viewHeight);
    RectF bufferRect = new RectF(0, 0, previewSize.getHeight(), previewSize.getWidth());

    float centerX = viewRect.centerX();
    float centerY = viewRect.centerY();

    // 获取设备当前旋转角度
    int rotation = getWindowManagerRotation(textureView.getContext());

    // 计算需要旋转的角度，改为左旋转90度（减去90度）
    int rotate = (sensorOrientation + rotation * 90 - 90) % 360;
    // 处理负数情况
    if (rotate < 0) {
        rotate += 360;
    }

    // 根据旋转角度调整 bufferRect
    if (rotate == 90 || rotate == 270) {
        bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY());
        matrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.FILL);

        // 计算缩放比例
        float scale = Math.max(
                (float) viewHeight / (float) previewSize.getHeight(),
                (float) viewWidth / (float) previewSize.getWidth());
        matrix.postScale(scale, scale, centerX, centerY);
    }

    // 应用旋转
    matrix.postRotate(rotate, centerX, centerY);

    textureView.setTransform(matrix);
}






    // 按面积比较尺寸大小的比较器
    static class CompareSizesByArea implements Comparator<Size> {
        @Override
        public int compare(Size lhs, Size rhs) {
            return Long.signum((long) lhs.getWidth() * lhs.getHeight() -
                    (long) rhs.getWidth() * rhs.getHeight());
        }
    }

    // 切换摄像头
    public void switchCamera() {
        try {
            // 关闭当前相机session
            closeCameraSession();

            // 获取所有相机ID
            String[] cameraIds = cameraManager.getCameraIdList();
            if (cameraIds.length > 1) {
                // 切换到另一个摄像头
                String newCameraId = cameraId.equals(getBackCameraId()) ? getFrontCameraId() : getBackCameraId();
                if (newCameraId != null) {
                    cameraId = newCameraId;
                    Log.d(TAG, "Switching to camera: " + cameraId);
                    // 延迟打开新相机
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

    // 获取前置摄像头ID
    private String getFrontCameraId() {
        try {
            // 遍历所有相机ID查找前置摄像头
            for (String cameraId : cameraManager.getCameraIdList()) {
                // 获取相机特性信息
                CameraCharacteristics characteristics = cameraManager.getCameraCharacteristics(cameraId);
                // 获取镜头朝向信息
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

    // 打开相机
    public void openCamera() {
        if (cameraId == null) {
            Log.e(TAG, "No camera ID available");
            if (recordingListener != null) {
                recordingListener.onError("没有可用的摄像头");
            }
            return;
        }

        try {
            // 检查相机权限
            if (ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA)
                    != PackageManager.PERMISSION_GRANTED) {
                Log.e(TAG, "Camera permission not granted");
                if (recordingListener != null) {
                    recordingListener.onError("没有摄像头权限");
                }
                return;
            }

            Log.d(TAG, "Opening camera: " + cameraId);
            // 异步打开相机设备
            cameraManager.openCamera(cameraId, new CameraDevice.StateCallback() {
                @Override
                public void onOpened(@NonNull CameraDevice camera) {
                    Log.d(TAG, "Camera opened successfully");
                    cameraDevice = camera;
                    // 创建相机预览
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

    // 获取相机错误信息
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

    // 创建相机预览
    private void createCameraPreview() {
        if (cameraDevice == null) {
            Log.e(TAG, "Camera device is null, cannot create preview");
            return;
        }

        try {
            // 获取SurfaceTexture对象
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


            // 创建Surface对象
            Surface surface = new Surface(texture);

            // 创建预览请求构建器
            CaptureRequest.Builder previewRequestBuilder =
                    cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            // 添加目标Surface
            previewRequestBuilder.addTarget(surface);

            // 使用自动模式
            previewRequestBuilder.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO);
            previewRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);

            // 创建捕获会话
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
                                // 设置重复请求以持续预览
                                session.setRepeatingRequest(previewRequestBuilder.build(), null, backgroundHandler);
                                Log.d(TAG, "Camera preview started successfully");

                                // 配置变换
                                if (textureView.getWidth() > 0 && textureView.getHeight() > 0) {
                                    configureTransform(textureView.getWidth(), textureView.getHeight());
                                }
                            } catch (CameraAccessException e) {
                                Log.e(TAG, "Error creating camera preview", e);
                                if (recordingListener != null) {
                                    recordingListener.onError("创建摄像头预览失败");
                                }
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


    // 开始录制
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
            // 检查录音权限
            if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO)
                    != PackageManager.PERMISSION_GRANTED) {
                Log.e(TAG, "Audio recording permission not granted");
                if (recordingListener != null) {
                    recordingListener.onError("没有录音权限");
                }
                return false;
            }

            // 检查存储权限
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
            // 设置默认缓冲区大小
            texture.setDefaultBufferSize(width, height);

            // 创建预览Surface和录制Surface
            Surface previewSurface = new Surface(texture);
            Surface recorderSurface = mediaRecorder.getSurface();

            List<Surface> surfaces = new ArrayList<>();
            surfaces.add(previewSurface);
            surfaces.add(recorderSurface);

            // 创建录制请求构建器
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
                                // 启动录制
                                mediaRecorder.start();
                                // 设置重复请求以持续录制
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

    // 检查存储权限
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

    // 设置媒体录制器
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
        // 设置音频源为摄像机模式
        mediaRecorder.setAudioSource(MediaRecorder.AudioSource.CAMCORDER); // 使用CAMCORDER而不是MIC，兼容性更好
        // 设置视频源为Surface
        mediaRecorder.setVideoSource(MediaRecorder.VideoSource.SURFACE);
        // 设置输出格式为MPEG-4
        mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);

        // 设置编码器 - 使用最兼容的设置
        // 设置视频编码器为H.264
        mediaRecorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264);
        // 设置音频编码器为AAC
        mediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);

        // 使用较低的参数确保兼容性
        mediaRecorder.setVideoEncodingBitRate(2000000); // 2 Mbps
        mediaRecorder.setVideoFrameRate(24); // 24 fps
        mediaRecorder.setVideoSize(1280, 720); // 720p 标准分辨率

        // 设置输出文件
        outputVideoFile = createVideoFile();
        // 设置输出文件路径
        mediaRecorder.setOutputFile(outputVideoFile.getAbsolutePath());

        // 设置方向（可选）
        mediaRecorder.setOrientationHint(90);


// 获取设备当前方向
int rotation = getWindowManagerRotation(context);
int degrees = rotation * 90;

// 计算正确的方向提示，改为左旋转90度（减去90度）
int orientationHint = (sensorOrientation + degrees - 90) % 360;
// 处理负数情况
if (orientationHint < 0) {
    orientationHint += 360;
}
// 确保是正确的角度（0, 90, 180, 270）
orientationHint = (orientationHint + 360) % 360;

mediaRecorder.setOrientationHint(orientationHint);



        try {
            // 准备录制器
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

    // 创建视频文件
    private File createVideoFile() {
        File mediaDir;

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            // Android 10及以上使用应用专属目录
            mediaDir = new File(context.getExternalFilesDir(Environment.DIRECTORY_MOVIES), "CameraRecorder");
        } else {
            if (Environment.MEDIA_MOUNTED.equals(Environment.getExternalStorageState())) {
                // 外部存储可用时使用公共目录
                mediaDir = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES), "CameraRecorder");
            } else {
                // 否则使用应用内部目录
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

        // 生成时间戳
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        File videoFile = new File(mediaDir, "VID_" + timeStamp + ".mp4");
        Log.d(TAG, "Video file path: " + videoFile.getAbsolutePath());

        return videoFile;
    }

    // 停止录制
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
                    // 停止录制
                    mediaRecorder.stop();
                } catch (Exception e) {
                    Log.e(TAG, "Error stopping MediaRecorder", e);
                }
                // 重置录制器
                mediaRecorder.reset();
                // 释放录制器资源
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

    // 关闭相机会话
    private void closeCameraSession() {
        Log.d(TAG, "Closing camera session");
        isRecording = false;

        if (captureSession != null) {
            // 关闭捕获会话
            captureSession.close();
            captureSession = null;
        }

        if (cameraDevice != null) {
            // 关闭相机设备
            cameraDevice.close();
            cameraDevice = null;
        }

        if (mediaRecorder != null) {
            try {
                // 释放录制器资源
                mediaRecorder.release();
            } catch (Exception e) {
                Log.e(TAG, "Error releasing MediaRecorder", e);
            }
            mediaRecorder = null;
        }
    }

    // 关闭相机
    public void closeCamera() {
        Log.d(TAG, "Closing camera");
        closeCameraSession();
    }

    // 启动后台线程
    public void startBackgroundThread() {
        if (backgroundThread == null) {
            // 创建后台线程
            backgroundThread = new HandlerThread("CameraBackground");
            // 启动线程
            backgroundThread.start();
            // 创建处理程序
            backgroundHandler = new Handler(backgroundThread.getLooper());
            Log.d(TAG, "Background thread started");
        }
    }

    // 停止后台线程
    public void stopBackgroundThread() {
        if (backgroundThread != null) {
            Log.d(TAG, "Stopping background thread");
            closeCameraSession();
            // 安全退出线程
            backgroundThread.quitSafely();
            try {
                // 等待线程结束
                backgroundThread.join();
                backgroundThread = null;
                backgroundHandler = null;
            } catch (InterruptedException e) {
                Log.e(TAG, "Error stopping background thread", e);
            }
        }
    }

    // 获取SurfaceTexture监听器
    public TextureView.SurfaceTextureListener getSurfaceTextureListener() {
        return surfaceTextureListener;
    }

// SurfaceTexture监听器实现
private TextureView.SurfaceTextureListener surfaceTextureListener = new TextureView.SurfaceTextureListener() {
    @Override
    public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
        Log.d(TAG, "SurfaceTexture available: " + width + "x" + height);
        // 当SurfaceTexture可用时打开相机
        openCamera();
    }

    @Override
public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {
    Log.d(TAG, "SurfaceTexture size changed: " + width + "x" + height);
    // 尺寸改变时重新配置变换
    configureTransform(width, height);
}


    @Override
    public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
        Log.d(TAG, "SurfaceTexture destroyed");
        return true;
    }

    @Override
    public void onSurfaceTextureUpdated(SurfaceTexture surface) {
        // 纹理更新
    }
};


    // 检查是否正在录制
    public boolean isRecording() {
        return isRecording;
    }

// 确认该方法存在且正确实现
@SuppressWarnings("deprecation")
private int getWindowManagerRotation(Context context) {
    WindowManager windowManager = (WindowManager) context.getSystemService(Context.WINDOW_SERVICE);
    int rotation;
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
        rotation = context.getDisplay().getRotation();
    } else {
        rotation = windowManager.getDefaultDisplay().getRotation();
    }

    // 将 Surface.ROTATION_* 转换为 0,1,2,3 的数值
    switch (rotation) {
        case Surface.ROTATION_0: return 0;
        case Surface.ROTATION_90: return 1;
        case Surface.ROTATION_180: return 2;
        case Surface.ROTATION_270: return 3;
        default: return 0;
    }
}


}
