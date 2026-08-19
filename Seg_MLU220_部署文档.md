# Segmentation — 寒武纪 MLU220 高光谱分割推理框架部署文档

## 项目概述

在寒武纪 MLU220 加速卡上部署**高光谱图像材质分割**（高光谱分割）推理服务。输入 45 通道高光谱图像（每个波段为灰度 PNG），通过滑窗推理 + 概率平均 + ArgMax 输出逐像素的 6 类材质分割结果。

---

## 整体架构

```
┌─────────────────────────────────────────────────────────┐
│  PyTorch 训练模型 (.pt)                                  │
│        │                                                │
│        ▼ (寒武纪 CNToolkit 离线转换)                     │
│  Cambricon 离线模型 (.cambricon)                         │
├─────────────────────────────────────────────────────────┤
│  源码工程 (x86 交叉编译)                                 │
│  ┌──────────────────────────────────────────────┐       │
│  │ common/preprocess/                            │       │
│  │   ├── seg_preproc.cpp  前处理 │       │
│  │   └── preproc.h                         声明   │       │
│  │ common/postprocess/                           │       │
│  │   ├── seg_postproc.cpp 后处理 │       │
│  │   └── postproc.h                        声明   │       │
│  │ samples/                                      │       │
│  │   ├── seg.cpp        动态库入口 │       │
│  │   └── unittest_seg.cpp 自测程序│       │
│  └──────────────────────────────────────────────┘       │
│        │ CMake + aarch64-linux-gnu-g++ 交叉编译           │
│        ▼                                                │
│  ┌──────────────────┐  ┌────────────────────────┐       │
│  │ libClassi-   │  │ infer_demo              │       │
│  │ fication.so      │  │ (自测可执行程序)          │       │
│  └──────────────────┘  └────────────────────────┘       │
├─────────────────────────────────────────────────────────┤
│  设备端 (MLU220, aarch64)                                │
│  ~/Class/                                           │
│  ├── infer_demo               # 推理入口                 │
│  ├── lib/libSeg.so  # 算法动态库          │
│  ├── models/material_1109.cambricon  # 离线模型          │
│  ├── 45张波段PNG                                   │
│  └── your_path/label_weixing.txt   # 类别标签                 │
└─────────────────────────────────────────────────────────┘
```

---

## 部署步骤

### 第一步：模型转换（PyTorch → Cambricon 离线模型）

将训练好的 PyTorch 模型转换为 MLU220 可执行的 `.cambricon` 离线模型。

- **输入**：`models/material1109.pt`（PyTorch 模型权重）
- **输出**：`models/material_1109.cambricon`（寒武纪离线推理模型）
- **工具**：寒武纪 CNToolkit / CNML 工具链
- **执行位置**：x86 开发主机

此步骤使用寒武纪官方工具完成，具体命令取决于 CNToolkit 版本和模型结构，一般流程为：

```
PyTorch .pt → ONNX → CNML .cambricon
```

---

### 第二步：编写前后处理，封装为动态库

在 `cambricon_hyperspectral_segmentation/` 源码工程中实现并注册前后处理函数。

#### 2.1 前处理

**文件**：`common/preprocess/seg_preproc.cpp`

**逻辑**：

```
输入：CV_8UC45 的 256×256 patch (HWC, 0-255)
  │
  ▼ cv::Mat::convertTo(CV_32F, 1.0/255.0)
归一化为 [0, 1] float32
  │
  ▼ model_input->buffers[0].CopyFrom(data, size)
拷贝到框架 buffer (NHWC 布局)
  │
  ▼
送入 MLU 推理
```

```cpp
bool SegPreProc::operator()(infer_server::ModelIO *model_input,
                                           const infer_server::InferData &infer_data,
                                           const infer_server::ModelInfo *model) {
    const auto& patch = infer_data.GetLref<cv::Mat>();
    cv::Mat normalized_patch;
    patch.convertTo(normalized_patch, CV_32F, 1.0 / 255.0);
    model_input->buffers[0].CopyFrom(normalized_patch.data,
        normalized_patch.total() * normalized_patch.elemSize());
    return true;
}
```

#### 2.2 后处理

**文件**：`common/postprocess/seg_postproc.cpp`

**逻辑**：

```
MLU 输出 buffer (float*)
  │
  ▼ memcpy → std::vector<float>
提取原始概率向量
  │
  ▼ result->Set(patch_probabilities)
传递给 Observer 做后续聚合
```

#### 2.3 动态库入口（Init / Process）

**文件**：`samples/seg.cpp`

暴露两个 `extern "C"` 函数，供动态库加载调用：

##### Init_() — 初始化

```cpp
Session_t Init_(InferServer* infer_server, SessionDesc& desc,
                Observer, model_path, threshold) {
    // 1. 加载离线模型
    desc.model = infer_server->LoadModel(model_path, "subnet0");
    // 2. 配置输入布局 NHWC + FLOAT32
    desc.host_input_layout.dtype = DataType::FLOAT32;
    desc.host_input_layout.order = DimOrder::NHWC;
    // 3. 绑定前后处理函数
    desc.preproc = PreprocessorHost::Create();
    desc.postproc = Postprocessor::Create();
    desc.preproc->SetParams("process_function",
        PreprocessorHost::ProcessFunction(SegPreProc()));
    desc.postproc->SetParams("process_function",
        Postprocessor::ProcessFunction(SegPostProcess(threshold)));
    // 4. 创建 Session
    return infer_server->CreateSession(desc, SegObserver::ms_observer_);
}
```

##### Process_() — 核心推理

```
输入: 45 通道高光谱图像
  │
  ▼ cv::resize(0.65x) → 缩小图像
  │
  ▼ 256×256 滑窗, stride=128 → 分 patch
  │
  ▼ infer_server->Request() → 异步批推理
  │
  ▼ Observer::Response() → 累加各像素 6 类概率
  │
  ▼ WaitTaskDone() → 等待全部 patch 完成
  │
  ▼ 平均概率 → ArgMax → 分割图
  │
  ▼ cv::resize(原尺寸) → 最近邻上采样
  │
输出: CV_8UC1 分割标签图 + 各类别像素数
```

##### Observer（异步结果聚合）

```cpp
class SegObserver : public Observer {
    void Response(Status, PackagePtr output, any user_data) {
        // 1. 获取该 patch 的 6 类概率向量
        auto& patch_probs = output->data[0]->GetLref<std::vector<float>>();
        // 2. 线程安全地累加到 full_probs (概率图) + count_map (覆盖计数)
        std::lock_guard<std::mutex> lock(*mtx);
        for (int h, h++, for int w) {
            for (int c = 0; c < 6; c++)
                full_probs.at<Vec<float,6>>(h,w)[c] += patch_probs[...];
        }
        count_map(roi) += 1;
    }
};
```

---

### 第三步：交叉编译（x86 → aarch64）

#### 3.1 构建脚本

`build.sh`：

```bash
#!/bin/sh
# 交叉编译: 使用 aarch64-linux-gnu-g++ 编译器
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ ..
make -j4
```

#### 3.2 CMakeLists.txt 关键配置

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `CMAKE_CXX_STANDARD` | 11 | C++11 标准 |
| `CMAKE_CXX_COMPILER` | `aarch64-linux-gnu-g++` | 交叉编译器 |
| 编译宏 | `HAVE_CNCV` | 启用 CNCV 加速 |
| 链接库 | `easydk` | 寒武纪推理 SDK |
| 模型输入布局 | `NHWC`, `FLOAT32` | 在 Init_ 中设置 |

```cmake
add_library(Seg SHARED
    samples/seg.cpp ${srcs})
target_link_libraries(Seg PRIVATE easydk)
target_compile_definitions(Seg PRIVATE HAVE_CNCV)

add_executable(infer_demo
    samples/unittest_seg.cpp
    common/cnosd.cpp common/dataframe.cpp
    common/collection.cpp common/infer_object.cpp)
target_link_libraries(infer_demo PRIVATE easydk pthread dl ${OpenCV_LIBS})
```

#### 3.3 编译产出

| 产出 | 说明 |
|------|------|
| `bin/infer_demo` | 分割自测可执行程序 |
| `lib/libSeg.so` | 分割算法动态库 |

---

### 第四步：部署到 MLU220 设备

通过 `sshpass` + `scp` 推送到设备（IP: YOUR_DEVICE_IP）：

```bash
sshpass -p YOUR_PASSWORD scp ./bin/infer_demo              root@YOUR_DEVICE_IP:~/Class
sshpass -p YOUR_PASSWORD scp ./lib/libSeg.so root@YOUR_DEVICE_IP:~/Class/lib
```

#### 设备端目录结构

```
~/Class/
├── infer_demo                       # 自测可执行程序
├── lib/
│   ├── libSeg.so     # 分割算法动态库
│   ├── libcnrt.so*                  # 寒武纪运行时库
│   ├── libcndrv.so*                 # 寒武纪驱动库
│   ├── libcncodec.so*               # 寒武纪编解码库
│   ├── libcncv.so*                  # 寒武纪视觉加速库
│   ├── libeasydk.so*                # EasyDK 推理框架
│   ├── libopencv_*.so*              # OpenCV 库
│   ├── libav*.so*                   # FFmpeg 库
│   ├── libgflags.so*                # gflags 库
│   └── libglog.so*                  # glog 库
├── models/
│   └── material_1109.cambricon      # 离线推理模型
├── your_path/
│   └── label_weixing.txt            # 类别标签文件
└── *.png                            # 45 张波段 PNG + 测试图片
```

---

### 第五步：设备端运行推理

```bash
cd ~/Class
./infer_demo
```

#### 运行流程（unittest_seg.cpp）

```
1. dlopen("lib/libSeg.so")  加载动态库
2. dlsym("Init_")   获取初始化函数
3. dlsym("Process_")获取处理函数
4. 创建 InferServer(device_id=0)
   配置 SessionDesc:
     - strategy = BatchStrategy::STATIC
     - engine_num = 4 (4个引擎并行处理patches)
     - show_perf = true
5. Init_()  → 加载模型 material_1109.cambricon
6. 从文件夹加载 45 张波段 PNG，合并为 45 通道图像
7. Process_()  → 滑窗推理 → 概率聚合 → ArgMax → 分割图
8. 输出 ClassifyResult2.png (彩色分割可视化图)
9. 输出各类别像素统计
```

#### 输入数据格式

- 45 张 256×256 灰度 PNG，每张代表一个光谱波段
- 文件名按数字排序（如 `*-90.png`, `*-92.png`, ..., `*-178.png`）
- 示例文件夹：`your_path/`、`your_path/`

#### 输出数据格式

- `ClassifyResult2.png`：彩色可视化分割图
  - 颜色映射：`0=黑, 1=红, 2=绿, 3=蓝, 4=黄, 5=青`
- 终端输出：各类别像素数量统计

---

## 核心参数

| 参数 | 值 | 定义位置 | 说明 |
|------|-----|----------|------|
| `NUM_CLASSES` | 6 | `seg.cpp:21` | 材质类别数 (ID 0-5) |
| `NUM_BANDS` | 45 | `seg.cpp:22` | 高光谱波段数 (输入通道数) |
| `WINDOW_SIZE` | 256×256 | `seg.cpp:23` | 滑窗尺寸 |
| `STRIDE` | 128 | `seg.cpp:24` | 滑窗步长 |
| 缩放比例 | 0.65 | `seg.cpp:161` | 输入图像缩放系数 |
| `engine_num` | 4 | `unittest_seg.cpp:98` | 并行推理引擎数 |
| 批处理策略 | STATIC | `unittest_seg.cpp:97` | 静态批处理 |
| 模型布局 | NHWC, FLOAT32 | `seg.cpp:116-117` | CPU 侧输入布局 |
| 模型函数名 | "subnet0" | `seg.cpp:111` | 离线模型子网络名 |
| 上采样方法 | INTER_NEAREST | `seg.cpp:233` | 结果缩放回原尺寸 |

---

## 推理性能（在 MLU220 上实测）

数据来源：`Class/output.txt`（一次完整推理统计）

| 指标 | 数值 |
|------|------|
| 总 patch 数 | 9959 次 |
| 单次推理 (Predictor) | 平均 28.3ms，最大 28.7ms，最小 14.5ms |
| 前处理 (PreprocessorHost) | 平均 0.042ms，最大 0.831ms |
| 后处理 (Postprocessor) | 平均 0.045ms，最大 0.176ms |
| 端到端请求延迟 (RequestLatency) | 平均 1354ms（含 4 引擎并行排队等候） |

---

## 类别标签

`your_path/label_weixing.txt`：

| 类别 ID | 说明 |
|---------|------|
| 0 | — |
| 1 | — |
| 2 | — |
| 3 | — |
| 4 | — |
| 5 | — |

*（标签文件仅包含类别 ID 索引，具体材质名称需参照项目业务文档）*

---

## 第三方依赖

| 依赖 | 版本 | 说明 |
|------|------|------|
| CNToolkit (neuware) | 4.10.2 | 寒武纪运行时和驱动 |
| EasyDK | 3.0.0 | 寒武纪推理开发框架 |
| OpenCV | 3.4.15 | 图像读写和处理 |
| FFmpeg | 4.x | 视频编解码（分割模块不直接使用） |
| gflags | 2.2.2 | 命令行参数解析 |
| glog | 0.4.0 | 日志库 |
| x264 | — | H.264 编码器（分割模块不直接使用） |

---

## 关键代码文件索引

| 文件 | 说明 |
|------|------|
| `samples/seg.cpp` | 动态库入口，Init_ / Process_ / Observer |
| `samples/unittest_seg.cpp` | 自测程序，加载 .so 并执行推理 |
| `common/preprocess/seg_preproc.cpp` | 前处理：CV_8UC45 → 归一化 float32 |
| `common/postprocess/seg_postproc.cpp` | 后处理：提取模型输出概率向量 |
| `common/preprocess/preproc.h` | 前处理函数声明 |
| `common/postprocess/postproc.h` | 后处理函数声明 |
| `CMakeLists.txt` | CMake 构建脚本 |
| `build.sh` | 一键构建 + 部署脚本 |
