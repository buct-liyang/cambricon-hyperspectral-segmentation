#!/bin/sh
# 仅增量编译（不删除 build 目录），部署参数与 build.sh 保持一致。

DEVICE_IP="YOUR_DEVICE_IP"
DEVICE_PASSWORD="YOUR_PASSWORD"
DEVICE_DIR="~/Class"

cd build
make -j4
cd ../

sshpass -p "$DEVICE_PASSWORD" scp ./bin/infer_demo \
    root@"$DEVICE_IP":"$DEVICE_DIR"
sshpass -p "$DEVICE_PASSWORD" scp ./lib/libSeg.so \
    root@"$DEVICE_IP":"$DEVICE_DIR/lib"
