#!/bin/sh
# ============================================================================
# 构建脚本：交叉编译（x86 -> aarch64）并部署到 MLU220 设备
# 使用前请修改下方“使用者配置”中的设备 IP / 密码 / 部署目录。
# ============================================================================

# ---- 使用者配置：改为您自己的设备信息 ----
DEVICE_IP="YOUR_DEVICE_IP"        # 设备 IP，例如 192.168.1.111
DEVICE_PASSWORD="YOUR_PASSWORD"   # 设备 root 密码
DEVICE_DIR="~/Class"              # 设备端部署目录（需提前创建）

CRRENT_FILE=$(cd $(dirname $0); pwd)
echo ${CRRENT_FILE}
if [ -d "build" ]; then
    echo "remove build dir"
    rm -rf build
fi
echo "create build dir"
mkdir build
chmod 755 * -R
cd build
# cmake ..
cmake -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ ..
make -j4
cd ../

sshpass -p "$DEVICE_PASSWORD" scp ./bin/infer_demo \
    root@"$DEVICE_IP":"$DEVICE_DIR"
sshpass -p "$DEVICE_PASSWORD" scp ./lib/libSeg.so \
    root@"$DEVICE_IP":"$DEVICE_DIR/lib"
