"""
Android 手势识别前端 - Mock WebSocket 测试服务器
================================================
模拟后端行为，用于在无真实后端时测试 App 的通信链路。

功能：
  1. 接收 Android 发送的帧（Base64 JPEG）
  2. 解码并保存帧图像到 received_frames/ 目录
  3. 实时显示统计信息：帧率、大小、帧索引、时间戳
  4. 发送模版翻译结果返回给 App（模拟后端响应）
  5. 支持保存的帧按顺序播放生成视频（可选）

用法：
  pip install websockets
  python mock_server.py

Android 端 Config.java 修改（二选一）:
  - 模拟器测试: WS_URL = "ws://10.0.2.2:8000/ws/translate"
  - 真机同一WiFi: WS_URL = "ws://<本机IP>:8000/ws/translate"
"""

import asyncio
import json
import time
import os
import base64
import signal
import sys
from datetime import datetime
from collections import deque

try:
    import websockets
except ImportError:
    print("=" * 60)
    print("错误: 缺少 websockets 库，请先安装:")
    print("  pip install websockets")
    print("=" * 60)
    sys.exit(1)

# ======================== 配置 ========================
HOST = "0.0.0.0"  # 监听所有网络接口
PORT = 8000
PATH = "/ws/translate"  # 必须与 Config.java 中的路径一致

SAVE_FRAMES = True  # 是否将接收到的帧保存为图片
SAVE_DIR = "received_frames"

# 启动选项
#   --fresh : 启动时清空 SAVE_DIR 目录（每次测试从干净状态开始）
CLEAR_ON_START = "--fresh" in sys.argv

# 模拟翻译响应配置
MOCK_TRANSLATION_TEXTS = [
    "你好",
    "谢谢",
    "是的",
    "不是",
    "好的",
    "没问题",
    "很高兴认识你",
    "再见",
    "请",
    "对不起",
]

# ======================== 统计信息 ========================
stats = {
    "total_frames": 0,
    "total_bytes": 0,
    "start_time": None,
    "last_frame_time": None,
    "fps_history": deque(maxlen=30),  # 最近30帧用于计算平滑FPS
    "connected": False,
    "client_addr": None,
    "received_texts": [],  # 用于接收非frame消息
}
# ======================== 帧保存 ========================
os.makedirs(SAVE_DIR, exist_ok=True)

# 如果指定了 --fresh，清空 SAVE_DIR 内所有旧文件
if CLEAR_ON_START:
    cleared = 0
    for f in os.listdir(SAVE_DIR):
        fpath = os.path.join(SAVE_DIR, f)
        if os.path.isfile(fpath):
            os.remove(fpath)
            cleared += 1
    print(f"[启动] 已清空 {SAVE_DIR}/ 目录（{cleared} 个旧文件）")
else:
    print(f"[启动] 旧帧保留在 {SAVE_DIR}/ 目录中，追加 --fresh 可启动时清空")



def save_frame(frame_idx, image_data, timestamp):
    """将解码后的 JPEG 数据保存为文件"""
    filename = f"frame_{frame_idx:06d}_ts{timestamp}.jpg"
    filepath = os.path.join(SAVE_DIR, filename)
    try:
        with open(filepath, "wb") as f:
            f.write(image_data)
        return filepath
    except Exception as e:
        return f"[保存失败: {e}]"


# ======================== 清除终端行 ========================
def clear_lines(n=6):
    """向上移动 n 行并清除"""
    for _ in range(n):
        sys.stdout.write("\033[F")  # 光标上移一行
        sys.stdout.write("\033[K")  # 清除当前行


# ======================== 打印统计面板 ========================
def print_stats_panel(initial=False):
    """打印或刷新实时统计面板"""
    now = time.time()
    elapsed = now - stats["start_time"] if stats["start_time"] else 0

    # 计算实时 FPS
    if stats["fps_history"]:
        recent_fps = sum(stats["fps_history"]) / len(stats["fps_history"])
    else:
        recent_fps = 0.0

    # 计算平均 FPS
    avg_fps = stats["total_frames"] / elapsed if elapsed > 0 else 0.0

    # 计算吞吐量
    total_mb = stats["total_bytes"] / (1024 * 1024)
    throughput = total_mb / elapsed * 8 if elapsed > 0 else 0.0  # Mbps

    lines = [
        "",
        "=" * 60,
        f"  🖥️  Mock 翻译服务器 | {HOST}:{PORT}{PATH}",
        f"  {'🟢 已连接' if stats['connected'] else '🔴 等待连接...'} "
        f" 客户端: {stats['client_addr'] or '无'}",
        "-" * 60,
        f"  帧数       : {stats['total_frames']:>6} 帧",
        f"  实时 FPS   : {recent_fps:>6.1f}  (最近30帧平均)",
        f"  平均 FPS   : {avg_fps:>6.1f}  (总运行时间)",
        f"  数据量     : {total_mb:>6.2f} MB  ({throughput:.1f} Mbps)",
        f"  运行时间   : {elapsed:>6.1f} 秒",
        "-" * 60,
        f"  帧保存目录 : {os.path.abspath(SAVE_DIR)}" if SAVE_FRAMES else "",
        "=" * 60,
        "",
    ]

    # 过滤空行
    output = "\n".join(line for line in lines if line)

    if initial:
        print(output)
    else:
        # 向上移动到面板顶部并重绘
        clear_lines(len([l for l in lines if l]))
        print(output)


# ======================== 帧处理 ========================
def process_frame_message(data):
    """处理客户端发送的帧消息"""
    try:
        # 解析 JSON
        message = json.loads(data)

        # 检查消息类型
        msg_type = message.get("type", "")

        if msg_type == "frame":
            # ----- 帧消息处理 -----
            frame_idx = message.get("frame_idx", -1)
            b64_image = message.get("image_b64", "")
            timestamp = message.get("timestamp", 0)
            b64_size = len(b64_image)
            raw_size = int(b64_size * 0.75)  # Base64 → 原始字节估算

            # 更新统计
            stats["total_frames"] += 1
            stats["total_bytes"] += raw_size
            now = time.time()
            if stats["last_frame_time"]:
                dt = now - stats["last_frame_time"]
                if dt > 0:
                    stats["fps_history"].append(1.0 / dt)
            stats["last_frame_time"] = now

            # 解码并保存
            saved_path = ""
            if SAVE_FRAMES:
                try:
                    image_data = base64.b64decode(b64_image)
                    saved_path = save_frame(frame_idx, image_data, timestamp)
                except Exception as e:
                    saved_path = f"[解码失败: {e}]"

            # 更新统计面板
            print_stats_panel(initial=False)

            # 每帧打印简要日志
            ts_str = datetime.fromtimestamp(timestamp / 1000).strftime("%H:%M:%S.%f")[:-3]
            img_info = f" | 保存: {os.path.basename(saved_path)}" if SAVE_FRAMES and saved_path else ""
            print(f"  📸 帧 #{frame_idx:>4d} | {b64_size:>6d}B Base64 | 时间戳: {ts_str}{img_info}")

            return True, raw_size

        elif msg_type == "control":
            # ----- 控制消息处理 -----
            action = message.get("action", "unknown")
            print(f"\n  🎮 收到控制消息: {action}")
            return True, 0

        else:
            print(f"\n  ❓ 未知消息类型: {msg_type}")
            print(f"     内容: {data[:200]}...")
            return True, 0

    except json.JSONDecodeError as e:
        print(f"\n  ❌ JSON 解析错误: {e}")
        print(f"     原始数据前200字符: {data[:200]}")
        return False, 0
    except Exception as e:
        print(f"\n  ❌ 处理异常: {e}")
        return False, 0


# ======================== 生成模拟翻译响应 ========================
def generate_mock_response(frame_idx):
    """生成一个模拟的翻译响应"""
    import random

    # 轮换不同的翻译文本
    text_idx = frame_idx % len(MOCK_TRANSLATION_TEXTS)
    text = MOCK_TRANSLATION_TEXTS[text_idx]

    # 模拟置信度（逐渐提高，模拟模型持续识别）
    confidence = min(0.95, 0.5 + frame_idx * 0.01)

    # 每 5 帧发送一次翻译结果（模拟实际推理频率）
    if frame_idx % 5 == 0:
        return {
            "type": "translation",
            "text": text,
            "confidence": round(confidence, 2),
            "frame_range": [max(0, frame_idx - 4), frame_idx],
        }
    else:
        # 大部分帧只返回 status（模拟心跳）
        return {
            "type": "status",
            "gpu_util": round(random.uniform(30, 80), 1),
            "latency_ms": round(random.uniform(50, 200), 1),
        }


# ======================== WebSocket 处理器 ========================
async def handler(websocket):
    """处理单个 WebSocket 连接"""
    stats["connected"] = True
    stats["client_addr"] = websocket.remote_address
    stats["start_time"] = time.time()
    stats["total_frames"] = 0
    stats["total_bytes"] = 0
    stats["last_frame_time"] = None
    stats["fps_history"].clear()

    print_stats_panel(initial=True)
    print(f"\n  ✅ 客户端已连接: {websocket.remote_address}")
    print(f"  📁 帧将保存到: {os.path.abspath(SAVE_DIR)}")
    print()

    try:
        async for message in websocket:
            # 处理接收到的消息
            success, raw_size = process_frame_message(message)

            # 发送模拟响应
            if success:
                response = generate_mock_response(stats["total_frames"])
                await websocket.send(json.dumps(response))

    except websockets.exceptions.ConnectionClosed as e:
        print(f"\n  ⚠️ 连接断开: {e.code} - {e.reason}")
    except Exception as e:
        print(f"\n  ❌ 连接异常: {e}")
    finally:
        stats["connected"] = False
        stats["client_addr"] = None
        print_stats_panel(initial=False)
        print(f"\n  🔴 客户端已断开，等待新连接...")


# ======================== 启动 ========================
async def main():
    print()
    print("=" * 60)
    print("  🚀 Mock 翻译服务器启动中...")
    print(f"  地址: ws://{HOST}:{PORT}{PATH}")
    print(f"  帧保存: {'启用' if SAVE_FRAMES else '禁用'}")
    print(f"  保存目录: {os.path.abspath(SAVE_DIR)}")
    print("=" * 60)
    print()
    print("  📋 Android 端配置指引:")
    print(f"     - 模拟器测试: ws://10.0.2.2:{PORT}{PATH}")
    print(f"     - 真机WiFi测试: ws://<本机IP>:{PORT}{PATH}")
    print()
    print("  💡 按 Ctrl+C 停止服务器")
    print("=" * 60)
    print()

    stats["start_time"] = time.time()

    async with websockets.serve(handler, HOST, PORT):
        print_stats_panel(initial=True)
        print()
        await asyncio.Future()  # 永久运行


if __name__ == "__main__":
    # 处理 Ctrl+C 优雅退出
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n  👋 服务器已停止")
        total_frames = stats["total_frames"]
        total_mb = stats["total_bytes"] / (1024 * 1024)
        print(f"  本次会话统计: 接收 {total_frames} 帧, {total_mb:.2f} MB")
        sys.exit(0)
