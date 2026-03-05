#!/bin/bash

# ==========================================
# AutoDL 快速部署脚本 (rsync)
# 用法: ./deploy.sh
# ==========================================

# 1. 配置你的 AutoDL SSH 连接信息
# 从 AutoDL 控制台复制 "SSH登录指令"，例如: ssh -p 12345 root@region-xxx.autodl.com
# 请将下面两行替换为你的实际端口和地址
PORT="23017"
HOST="root@connect.nmb1.seetacloud.com"

# 2. 配置服务器上的目标路径 (代码要存放在哪里)
# 例如: /root/autodl-tmp/relight-app/ 或者 /root/relight-app/
DEST_DIR="/root/relight-app/"

# 本地项目目录 (脚本所在目录)
LOCAL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/"

echo "🚀 开始同步代码到 AutoDL..."
echo "-> 目标: $HOST:$PORT$DEST_DIR"

# 3. 执行 rsync 同步
# -a: 归档模式，保留权限和属性
# -v: 显示详细输出
# -z: 压缩传输，加快速度
# --exclude: 排除不需要同步的本地文件和文件夹 (比如大文件、Git 记录、缓存等)
# --delete: (可选) 如果你想让云端和本地严格一致(删除云端多余的文件)，可以取消下一行的注释
#           注意：确保目标目录确实是此项目，避免误删！

rsync -avz -e "ssh -p $PORT" \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='.claude/' \
    --exclude='.DS_Store' \
    --exclude='*.pyc' \
    --exclude='Image_Lighting_Tool_updated/' \
    --exclude='result.png' \
    --exclude='*.jpg' \
    --exclude='*.png' \
    --exclude='models/' \
    "$LOCAL_DIR" \
    "$HOST:$DEST_DIR"

if [ $? -eq 0 ]; then
    echo "✅ 同步成功！"
    echo "正在自动重启 AutoDL 上的服务..."
    
    # 4. 自动重启远端服务
    ssh -p $PORT $HOST "cd $DEST_DIR && bash start.sh"
    
    if [ $? -eq 0 ]; then
        echo "🎉 部署与重启全部完成！现在可以直接测试你的 Web API 了。"
    else
        echo "⚠️  代码已同步，但重启服务时可能发生错误，请登录 AutoDL 检查日志。"
    fi
else
    echo "❌ 同步失败，请检查 SSH 连接和网络！"
fi
