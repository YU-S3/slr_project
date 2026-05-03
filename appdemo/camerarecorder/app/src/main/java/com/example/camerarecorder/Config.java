package com.example.camerarecorder;

/**
 * 应用全局配置常量
 * 所有可调参数集中管理，方便调试和部署
 */
public class Config {

    // ======================== 测试模式开关 ========================
    /** 测试模式：true 时使用 TEST_WS_URL 连接本地 Mock 服务器 */
    public static final boolean TEST_MODE = true;

    // ======================== WebSocket 服务器配置 ========================
    /** 后端 WebSocket 地址（部署时根据实际 IP 修改） */
    public static final String PRODUCTION_WS_URL = "ws://192.168.1.100:8000/ws/translate";

    /** 测试模式 WebSocket 地址（模拟器用 10.0.2.2，真机同WiFi用PC的局域网IP） */
    public static final String TEST_WS_URL = "ws://192.168.2.32:8000/ws/translate";

    /** 当前生效的 WebSocket 地址（根据 TEST_MODE 自动切换） */
    public static final String WS_URL = TEST_MODE ? TEST_WS_URL : PRODUCTION_WS_URL;

    /** WebSocket 自动重连延迟（毫秒） */
    public static final int RECONNECT_DELAY_MS = 3000;

    /** 最大自动重连次数 */
    public static final int MAX_RECONNECT_ATTEMPTS = 5;

    // ======================== 帧采集参数 ========================
    /** 抽帧间隔（毫秒），100ms = 10 FPS */
    public static final int CAPTURE_INTERVAL_MS = 100;

    /** JPEG 压缩质量（0-100），推荐 85 */
    public static final int JPEG_QUALITY = 85;

    /** 发送帧的最大宽度（超过此值会等比缩放） */
    public static final int MAX_FRAME_WIDTH = 640;

    /** 发送帧的最大高度（超过此值会等比缩放） */
    public static final int MAX_FRAME_HEIGHT = 480;

    // ======================== 运动检测参数 ========================
    /** 运动检测阈值：差异像素占比超过此值判定为"有运动"（0.0 - 1.0） */
    public static final float MOTION_THRESHOLD = 0.05f;

    /** 帧差法缩放到的小尺寸（用于加速计算） */
    public static final int MOTION_SCALE_SIZE = 32;

    /** 像素差异阈值（灰度值差超过此值视为"不同像素"） */
    public static final int PIXEL_DIFF_THRESHOLD = 30;

    // ======================== 调试模式 ========================
    /** 是否启用调试浮层（false = 彻底不显示，true = 按 DEBUG_SHOWN_BY_DEFAULT 决定初始状态） */
    public static final boolean DEBUG_MODE = true;

    /** 调试浮层初始是否可见（DEBUG_MODE=true 时生效） */
    public static final boolean DEBUG_SHOWN_BY_DEFAULT = false;

    /** 调试信息刷新间隔（毫秒） */
    public static final int DEBUG_REFRESH_INTERVAL_MS = 500;
}
