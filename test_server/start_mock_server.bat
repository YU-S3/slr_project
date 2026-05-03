@echo off
chcp 65001 >nul
title Mock 翻译服务器 - 手势识别调试工具

echo ============================================================
echo   🚀 启动 Mock 翻译服务器
echo ============================================================
echo.
echo   正在检查 Python 环境...
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   ❌ 错误: 未找到 Python，请先安装 Python 3.8+
    echo.
    pause
    exit /b 1
)

echo   正在检查 websockets 库...
python -c "import websockets" >nul 2>&1
if %errorlevel% neq 0 (
    echo   ⚠️  未安装 websockets 库，正在安装...
    pip install websockets
    echo.
)

echo   启动服务器（--fresh 模式：自动清空上次的旧帧）...
echo.
python mock_server.py --fresh

pause
