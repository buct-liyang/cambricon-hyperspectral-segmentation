# C++ 推理前处理改造：方案v2 固定归一化

## 改动背景

旧版 C++ 推理链路对应旧训练（逐裁剪块/整图自适应 min-max + 亮度偏移），
与方案v2（工业版）训练的前处理不一致。方案v2 训练/推理统一使用**固定归一化**：

```
x' = clip(x - 5, 0, 40) / 40
```

（参数依据：6 组统计 p1≈5、99% 像元集中在 5~36；与 `fanganv2/code/normalize.py`
及 `dataset_enhanced.py` 的 `norm_mode="fixed"` 完全一致。）

## 新增/修改文件

| 文件 | 说明 |
|---|---|
| `samples/seg_fixed_norm.cpp` | **新增**：方案v2 动态库入口（Init_/Process_/Observer），整图固定归一化后滑窗 |
| `common/preprocess/seg_fixed_preproc.cpp` | **新增**：固定归一化版预处理器，仅把 float32 patch 拷贝到模型输入 |
| `common/preprocess/preproc.h` | **修改**：新增 `SegFixedPatch` / `SegFixedPreProc` 声明 |
| `CMakeLists.txt` | **修改**：新增 `libSegFixedNorm.so` 构建目标 |

旧文件 `samples/seg.cpp` 与 `seg_preproc.cpp` **保持不动**。

## 引用关系（与旧版对比）

```
旧版：Process_(seg.cpp)
        → ComputeAdaptiveStats() 全图自适应统计
        → SegPatch{uint8 patch, stats}
        → SegPreProc：逐波段 min-max + 亮度偏移 → float32

新版：Process_(seg_fixed_norm.cpp)
        → FixedNormalize() 整图固定归一化 clip(x-5,0,40)/40
        → SegFixedPatch{float32 patch}
        → SegFixedPreProc：仅 HWC 拷贝 → float32
```

后处理（`SegPostProcess`）、Observer 概率聚合、滑窗/概率平均/ArgMax、
最近邻放大回原尺寸等逻辑不变。

## 构建与部署

1. 交叉编译（与旧版相同命令）：
   ```bash
   cd cambricon_hyperspectral_segmentation
   ./build.sh
   ```
2. 产出新库：`lib/libSegFixedNorm.so`（旧库仍同时产出）。
3. 设备端使用：
   - 方式 A：直接部署新库并改 `unittest_seg.cpp` 中的
     `lib_path`（第 77 行附近）为 `./lib/libSegFixedNorm.so`；
   - 方式 B：将新库复制/改名为 `libSeg.so` 覆盖旧库
     （unittest 的 dlopen 路径无需改动）。
4. 使用的模型权重必须来自方案v2 训练（`fanganv2/code/train_enhanced.py`，
   推荐 `best_miou_model.pth` 转换的 `.cambricon`）。

## 注意事项

- 输入 45 个波段的顺序必须与训练一致（按波段文件名数字 1..45 排序）。
- 归一化参数 `kNORM_LOW=5.0f`、`kNORM_HIGH=40.0f` 定义在
  `seg_fixed_norm.cpp` 顶部，如训练集统计更新需同步修改
  训练（normalize.py）与推理（C++）两侧。
- 滑窗归一化在**整幅缩放后图像**上做一次，patch 不重复归一化。
- 编译必须使用工程自带的交叉工具链（`./build.sh` 中的
  `aarch64-linux-gnu-g++`）；Windows 本地 MinGW 缺少线程库（std::mutex），
  无法在本机做全量语法检查。