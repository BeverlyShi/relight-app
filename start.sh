#!/bin/bash

echo "======================================"
echo "Restarting Relight FastAPI Server..."
echo "======================================"

# 1. 进入脚本所在的当前项目目录
cd "$(dirname "$0")"

# 2. 杀掉旧的 server.py 进程 (防止端口冲突)
echo "Stopping old server processes..."
pkill -f "python server.py"

# 小幅等待，确保进程彻底释放端口
sleep 2

# 3. 后台启动新的 server.py
echo "Starting new server process..."
nohup python server.py > server.log 2>&1 &

# 4. 检查是否启动成功
sleep 2
if pgrep -f "python server.py" > /dev/null; then
    echo "✅ Server successfully started in background!"
    echo "Logs are being written to server.log"
else
    echo "❌ Failed to start server. Please check server.log"
    cat server.log
fi
