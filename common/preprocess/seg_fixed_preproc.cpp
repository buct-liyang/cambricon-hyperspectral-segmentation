/*************************************************************************
 * Copyright (C) [2021] by Cambricon, Inc. All rights reserved 
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *************************************************************************/
/**
 * @file seg_fixed_preproc.cpp
 * @brief 方案v2（固定归一化）的前处理实现。
 *
 * 与旧版 seg_preproc.cpp 的区别：
 *   - 旧版：输入 CV_8UC(N) 原始 uint8 patch，在预处理器内做逐波段 min-max
 *           对比度拉伸 + 自适应亮度偏移（与旧训练 dataset_adaptive.py 对应）。
 *   - 新版：整幅图像已在 Process_（seg_fixed_norm.cpp）中完成
 *           固定归一化 clip(x-5, 0, 40)/40（与方案v2训练/推理一致），
 *           这里只把 CV_32FC(N) 的归一化 patch 按 NHWC 布局拷贝到模型输入 buffer，
 *           不再做任何统计或增强。
 */

#include <glog/logging.h>

#include <exception>
#include <mutex>
#include <condition_variable>

#include "cnis/infer_server.h"
#include "cnis/processor.h"
#include <opencv2/opencv.hpp>

#include "preproc.h"

/**
 * @brief 预处理：CV_32FC(N) 归一化 patch -> 模型输入 buffer（NHWC, FLOAT32）
 */
bool SegFixedPreProc::operator()(infer_server::ModelIO* model_input,
                                                const infer_server::InferData& infer_data,
                                                const infer_server::ModelInfo* model) {
  // 1. 从 infer_data 获取固定归一化后的 patch（CV_32FC(N)，NHWC 布局）
  const auto& patch_data = infer_data.GetLref<SegFixedPatch>();

  // 2. 校验模型输入布局
  const auto& model_layout = model->InputLayout(0);
  if (model_layout.order != infer_server::DimOrder::NHWC) {
    LOG(ERROR) << "Model input layout is not NHWC as expected.";
    return false;
  }
  if (model_layout.dtype != infer_server::DataType::FLOAT32) {
    LOG(WARNING) << "Model input dtype is not FLOAT32, but "
                 << static_cast<int>(model_layout.dtype) << ".";
  }

  // 3. 确保 patch 内存连续
  cv::Mat patch;
  if (patch_data.patch.isContinuous()) {
    patch = patch_data.patch;
  } else {
    patch = patch_data.patch.clone();
  }

  // 4. 校验类型：固定归一化版本要求 float32 输入
  const int channels = patch.channels();
  if (patch.type() != CV_32FC(channels)) {
    LOG(ERROR) << "Patch type is not CV_32FC" << channels
               << ", fixed-normalization preproc requires float32 input.";
    return false;
  }

  // 5. 直接拷贝归一化后的数据到模型输入 buffer（NHWC 布局，无需再转换）
  const size_t src_data_size = patch.total() * patch.elemSize();
  try {
    model_input->buffers[0].CopyFrom(patch.data, src_data_size);
  } catch (const std::exception& e) {
    LOG(FATAL) << "CopyFrom failed with exception: " << e.what() << ". "
               << "Source size: " << src_data_size << ", "
               << "Destination buffer capacity: "
               << model_input->buffers[0].MemorySize();
    return false;
  }

  return true;
}