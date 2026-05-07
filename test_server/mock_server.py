"""
Android 手势识别前端 - Mock WebSocket 测试服务器
================================================
遵循 API (1).md 协议规范（整段录制 → 一次性提交 → 后端处理 → 返回翻译）。

核心流程变化（相对于旧 API.md）:
  旧: 实时逐帧推送 → 实时返回中间翻译
  新: 先本地缓存整段录制 → 停止后一次性提交 → 后端集中处理 → 返回最终翻译

功能：
  1. 连接后发送 session_started（含 processing_mode: "full_recording"）
  2. 接收 Android 发送的帧序列（停止后批量提交）
  3. 每帧回复 recording 消息（模拟服务端接收确认）
  4. 最后一帧（flush=true + is_final=true）触发 processing + translation
  5. 支持 ping（心跳）和 reset（重置会话）消息
  6. 解码并保存帧图像到 received_frames/ 目录
  7. 实时显示统计信息

用法：
  pip install websockets
  python mock_server.py [--fresh]

Android 端 Config.java 修改（二选一）:
  - 模拟器测试: WS_URL = "ws://10.0.2.2:8000/ws"
  - 真机同一WiFi: WS_URL = "ws://<本机IP>:8000/ws"
"""

import asyncio
import json
import time
import os
import base64
import signal
import sys
import uuid
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
PATH = "/ws"  # 必须与 API (1).md 和 Config.java 中的路径一致

SAVE_FRAMES = True  # 是否将接收到的帧保存为图片
SAVE_DIR = "received_frames"

# 启动选项
#   --fresh : 启动时清空 SAVE_DIR 目录（每次测试从干净状态开始）
CLEAR_ON_START = "--fresh" in sys.argv

# 模拟翻译响应配置（API (1).md 的 result 字段）
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

# 模拟 rag_hits 配置（固定样本，模拟 RAG 检索结果）
MOCK_RAG_HITS = [
    {
        "rank": 1,
        "sign_gloss": "S000001_P0000_T00",
        "semantic_text": "你好",
        "context": "你好",
        "distance": 0.92,
        "video_path": "/mock/videos/S000001_P0000_T00.mp4",
        "pose_delta_summary": {
            "frame_delta_count": 5,
            "delta_norm_mean": 0.0987,
            "delta_norm_max": 0.1654
        }
    },
    {
        "rank": 2,
        "sign_gloss": "S000002_P0001_T01",
        "semantic_text": "您好",
        "context": "您好",
        "distance": 0.85,
        "video_path": "/mock/videos/S000002_P0001_T01.mp4",
        "pose_delta_summary": {
            "frame_delta_count": 6,
            "delta_norm_mean": 0.1123,
            "delta_norm_max": 0.1987
        }
    }
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
    "session_id": None,
    "frame_window_size": 8,
    "processing_mode": "full_recording",
    "sessions_processed": 0,  # 已处理的完整录制轮次
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
        f"  🖥️  Mock 翻译服务器 (API (1).md 协议) | {HOST}:{PORT}{PATH}",
        f"  {'🟢 已连接' if stats['connected'] else '🔴 等待连接...'} "
        f" 客户端: {stats['client_addr'] or '无'}",
        f"  会话: {stats['session_id'] or '无'}  模式: {stats['processing_mode']}",
        "-" * 60,
        f"  帧数       : {stats['total_frames']:>6} 帧",
        f"  实时 FPS   : {recent_fps:>6.1f}  (最近30帧平均)",
        f"  平均 FPS   : {avg_fps:>6.1f}  (总运行时间)",
        f"  数据量     : {total_mb:>6.2f} MB  ({throughput:.1f} Mbps)",
        f"  运行时间   : {elapsed:>6.1f} 秒",
        f"  已处理轮次 : {stats['sessions_processed']:>6} 次",
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
    """处理客户端发送的帧消息（API (1).md 协议）"""
    try:
        # 解析 JSON
        message = json.loads(data)

        # 检查消息类型
        msg_type = message.get("type", "")

        if msg_type == "frame":
            # ----- 帧消息处理 -----
            frame_idx = message.get("frame_idx", -1)
            b64_image = message.get("image_b64", "")
            flush = message.get("flush", False)
            is_final = message.get("is_final", False)

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
            if SAVE_FRAMES and b64_image:
                try:
                    image_data = base64.b64decode(b64_image)
                    timestamp = int(now * 1000)
                    saved_path = save_frame(frame_idx, image_data, timestamp)
                except Exception as e:
                    saved_path = f"[解码失败: {e}]"

            # 更新统计面板
            print_stats_panel(initial=False)

            # 每帧打印简要日志
            ts_str = datetime.fromtimestamp(now).strftime("%H:%M:%S.%f")[:-3]
            img_info = f" | 保存: {os.path.basename(saved_path)}" if SAVE_FRAMES and saved_path else ""
            flush_info = " | FLUSH" if flush else ""
            final_info = " | FINAL" if is_final else ""
            print(f"  📸 帧 #{frame_idx:>4d} | {b64_size:>6d}B Base64 | {ts_str}{img_info}{flush_info}{final_info}")

            return True, raw_size, frame_idx, flush, is_final

        elif msg_type == "ping":
            # ----- 心跳 -----
            print(f"\n  💓 收到心跳 (ping)")
            return True, 0, -1, False, False

        elif msg_type == "reset":
            # ----- 重置会话 -----
            print(f"\n  🔄 收到重置 (reset)")
            stats["total_frames"] = 0
            stats["total_bytes"] = 0
            stats["last_frame_time"] = None
            stats["fps_history"].clear()
            return True, 0, -1, False, False

        else:
            print(f"\n  ❓ 未知消息类型: {msg_type}")
            print(f"     内容: {data[:200]}...")
            return True, 0, -1, False, False

    except json.JSONDecodeError as e:
        print(f"\n  ❌ JSON 解析错误: {e}")
        print(f"     原始数据前200字符: {data[:200]}")
        return False, 0, -1, False, False
    except Exception as e:
        print(f"\n  ❌ 处理异常: {e}")
        return False, 0, -1, False, False


# ======================== 生成模拟响应（API (1).md 协议） ========================
def generate_recording_response(frame_idx, buffered_count):
    """生成 recording 消息（新 API，替代旧 buffering）"""
    return {
        "type": "recording",
        "session_id": stats["session_id"],
        "frame_idx": frame_idx,
        "buffered_frames": buffered_count,
    }


def generate_processing_response(total_frames):
    """生成 processing 消息（新 API）"""
    return {
        "type": "processing",
        "session_id": stats["session_id"],
        "frame_idx": total_frames,
        "total_frames": total_frames,
    }


def generate_translation_response():
    """生成翻译结果（API (1).md 格式，含 result 和 rag_hits）"""
    import random

    # 基于已处理的轮次选择翻译文本
    text_idx = stats["sessions_processed"] % len(MOCK_TRANSLATION_TEXTS)
    result_text = MOCK_TRANSLATION_TEXTS[text_idx]

    # 构造 rag_hits
    rag_hits = []
    for hit_template in MOCK_RAG_HITS:
        hit = hit_template.copy()
        hit["semantic_text"] = result_text
        hit["context"] = result_text
        hit["distance"] = round(random.uniform(0.80, 0.98), 2)
        rag_hits.append(hit)

    return {
        "type": "translation",
        "session_id": stats["session_id"],
        "frame_idx": stats["total_frames"],
        "result": result_text,
        "rag_hits": rag_hits,
    }


# ======================== WebSocket 处理器 ========================
async def handler(websocket):
    """处理单个 WebSocket 连接（遵循 API (1).md 协议）"""
    stats["connected"] = True
    stats["client_addr"] = websocket.remote_address
    stats["start_time"] = time.time()
    stats["total_frames"] = 0
    stats["total_bytes"] = 0
    stats["last_frame_time"] = None
    stats["fps_history"].clear()

    # 生成会话 ID
    stats["session_id"] = str(uuid.uuid4())
    stats["frame_window_size"] = 8
    stats["processing_mode"] = "full_recording"

    print_stats_panel(initial=True)
    print(f"\n  ✅ 客户端已连接: {websocket.remote_address}")
    print(f"  🆔 会话 ID: {stats['session_id']}")
    print(f"  📁 帧将保存到: {os.path.abspath(SAVE_DIR)}")
    print()

    # ---- 连接后立即发送 session_started（API (1).md 协议） ----
    session_started_msg = {
        "type": "session_started",
        "session_id": stats["session_id"],
        "frame_window_size": stats["frame_window_size"],
        "processing_mode": stats["processing_mode"],
    }
    await websocket.send(json.dumps(session_started_msg))
    print(f"  🚀 已发送 session_started (window={stats['frame_window_size']}, mode={stats['processing_mode']})")
    print()

    frame_count_in_session = 0
    received_flush = False

    try:
        async for message in websocket:
            # 处理接收到的消息
            success, raw_size, frame_idx, flush, is_final = process_frame_message(message)

            if not success:
                continue

            # 根据消息类型发送响应
            msg_data = json.loads(message)
            msg_type = msg_data.get("type", "")

            if msg_type == "frame":
                frame_count_in_session += 1

                # ---- 1. 发送 recording 消息（替代旧 buffering） ----
                recording_msg = generate_recording_response(frame_idx, frame_count_in_session)
                await websocket.send(json.dumps(recording_msg))

                # ---- 2. 检测是否最后一帧（flush + is_final） ----
                if flush and is_final and not received_flush:
                    received_flush = True
                    total_frames = frame_count_in_session
                    print(f"\n  🏁 收到最后帧 #{frame_idx} (flush + final)，共 {total_frames} 帧")

                    # ---- 3. 发送 processing 消息（新 API） ----
                    await asyncio.sleep(0.1)  # 模拟处理开始确认延迟
                    processing_msg = generate_processing_response(total_frames)
                    await websocket.send(json.dumps(processing_msg))
                    print(f"  ⏳ 已发送 processing ({total_frames} 帧)")

                    # ---- 4. 模拟后端处理延迟 ---- 
                    print(f"  🧠 后端处理中...")
                    await asyncio.sleep(1.5)  # 模拟推理时间

                    # ---- 5. 发送最终翻译结果 ----
                    stats["sessions_processed"] += 1
                    translation_msg = generate_translation_response()
                    await websocket.send(json.dumps(translation_msg))
                    result_text = translation_msg["result"]
                    hit_count = len(translation_msg["rag_hits"])
                    print(f"  📝 发送翻译: \"{result_text}\" ({hit_count}个RAG命中)")
                    print()

            elif msg_type == "ping":
                # ---- 心跳 - 无需回复 ----
                pass

            elif msg_type == "reset":
                # ---- 重置会话 ----
                frame_count_in_session = 0
                received_flush = False
                print(f"  🔄 会话已重置")

            # 每帧后短暂让步，避免阻塞事件循环
            await asyncio.sleep(0.01)

    except websockets.exceptions.ConnectionClosed as e:
        print(f"\n  ⚠️ 连接断开: {e.code} - {e.reason}")
    except Exception as e:
        print(f"\n  ❌ 连接异常: {e}")
    finally:
        stats["connected"] = False
        stats["client_addr"] = None
        stats["session_id"] = None
        print_stats_panel(initial=False)
        print(f"\n  🔴 客户端已断开，等待新连接...")


# ======================== 启动 ========================
async def main():
    print()
    print("=" * 60)
    print("  🚀 Mock 翻译服务器启动中...")
    print(f"  地址: ws://{HOST}:{PORT}{PATH}")
    print(f"  协议: API (1).md (session_started / recording / processing / translation)")
    print(f"  工作流: 整段录制 → 一次性提交 → 后端处理 → 返回翻译")
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
        sessions = stats["sessions_processed"]
        print(f"  本次会话统计: 处理 {sessions} 轮录制, 接收 {total_frames} 帧, {total_mb:.2f} MB")
        sys.exit(0)
