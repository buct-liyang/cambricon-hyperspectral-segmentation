# Cambricon Hyperspectral Segmentation Inference Framework (MLU220)

基于寒武纪 MLU220（EasyDK / CNRT）的高光谱图像逐像素材质分割推理工程。

输入 45 个波段的高光谱图像（每个波段一张灰度 PNG），通过滑窗推理 +
重叠概率平均 + ArgMax 输出逐像素的 6 类分割结果。

## 功能特性

- 45 波段高光谱图像分割（6 类材质）
- 256×256 滑窗 + 128 stride，重叠区域概率平均
- 两种前处理模式：
  - 旧版：逐波段 min-max 对比度拉伸 + 自适应亮度偏移（`seg.cpp`）
  - 新版：整图固定归一化 `clip(x - 5, 0, 40) / 40`（`seg_fixed_norm.cpp`，
    推荐，与方案v2训练前处理一致）
- 结果最近邻放大回原尺寸，输出类别标签图与各类别像素统计

## 目录结构

```
├── build.sh                        构建 + 部署脚本（修改设备 IP/密码后使用）
├── only_make.sh                    增量编译脚本
├── CMakeLists.txt                  CMake 构建配置
├── cmake/                          CMake 查找模块（Neuware/Glog/GFlags）
├── 3rdparty/                       第三方依赖目录（不随仓库分发，需按下方
│                                   “第三方依赖获取”自行放置）
├── common/
│   ├── preprocess/                 前处理实现（adaptive / fixed 两种）
│   └── postprocess/                后处理实现
├── samples/
│   ├── seg.cpp          动态库入口（旧版自适应前处理）
│   ├── seg_fixed_norm.cpp 动态库入口（新版固定归一化）
│   └── unittest_seg.cpp 自测程序
├── predict_adaptive.py             Python 参考推理脚本
└── Seg_MLU220_部署文档.md  部署文档
```

## 环境依赖

- 寒武纪 CNToolkit / Neuware（含 CNRT）
- EasyDK 3.x
- OpenCV 3.4+
- glog / gflags
- aarch64 交叉编译器：`aarch64-linux-gnu-g++`
- MLU220 设备端运行时库

### 第三方依赖获取

本仓库**不随源码分发任何第三方库**（许可证与体积原因）。构建前请自行从
官方来源下载，并按目录结构放置到 `3rdparty/` 下：

| 依赖 | 版本 | 许可证 | 获取来源 |
|---|---|---|---|
| Neuware / CNToolkit（CNRT/CNDRV/CNCV/CNCodec 等） | 4.10.2 | 寒武纪商业许可 | 寒武纪官方（联系寒武纪获取 CNToolkit 安装包） |
| EasyDK | 3.0.0 | 寒武纪商业许可 | 寒武纪官方（随 CNToolkit/开发套件提供） |
| OpenCV | 3.4.15 | Apache-2.0 / BSD | https://opencv.org/ |
| FFmpeg | 4.x | LGPL-2.1+（部分 GPL） | https://ffmpeg.org/ |
| x264 | 157 | GPL-2.0 | https://www.videolan.org/developers/x264.html |
| gflags | 2.2.2 | BSD-3-Clause | https://github.com/gflags/gflags |
| glog | 0.4.0 | BSD-3-Clause | https://github.com/google/glog |

目录布局要求：

```text
3rdparty/
├── neuware/            # include/ + lib64/
├── easydk/             # include/ + lib/
├── opencv/             # include/ + lib/ + share/
├── ffmpeg/             # include/ + lib/
├── x264/               # include/ + lib/
├── gflags/             # include/ + lib/
└── glog/               # include/ + lib/
```

## 构建与部署

1. 修改 `build.sh` 顶部的使用者配置：

   ```sh
   DEVICE_IP="YOUR_DEVICE_IP"        # 例如 192.168.1.111
   DEVICE_PASSWORD="YOUR_PASSWORD"
   DEVICE_DIR="~/Class"
   ```

2. 交叉编译并部署：

   ```bash
   ./build.sh
   ```

   产出：
   - `bin/infer_demo`：自测程序
   - `lib/libSeg.so`：分割算法动态库（固定归一化版为
     `lib/libSegFixedNorm.so`）

3. 设备端运行前，修改 `samples/unittest_seg.cpp` 中的路径：

   ```cpp
   std::string model_path  = "./models/material_1109.cambricon"; // 您的离线模型
   std::string lib_path    = "./lib/libSeg.so";       // 动态库
   std::string input_folder = "your_path";                       // 45 张波段 PNG 所在目录
   ```

4. 设备端执行 `./infer_demo`。

## 模型说明

- 训练脚本（PyTorch）与推理前处理需保持一致的归一化参数；
- 新版固定归一化参数 `clip(x-5, 0, 40)/40` 见
  `samples/seg_fixed_norm.cpp` 顶部 `kNORM_LOW` / `kNORM_HIGH`；
- 旧版自适应前处理对应旧训练策略，二者不可混用。

## License

本项目基于寒武纪（Cambricon）EasyDK 示例代码改造，遵循 **Apache License,
Version 2.0** 开源（见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)）。
第三方库版权归其各自所有者所有，按各自许可证使用，本仓库不代为分发。
