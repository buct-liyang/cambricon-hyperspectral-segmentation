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
#ifndef EDK_SAMPLES_PREPROCESS_PREPROC_H_
#define EDK_SAMPLES_PREPROCESS_PREPROC_H_

#include <glog/logging.h>

#include <memory>
#include <utility>
#include <vector>

#ifdef HAVE_CNCV
#include "cncv.h"
#endif
#include <cnrt.h>

#include "cnis/infer_server.h"
#include "cnis/processor.h"
#include <opencv2/opencv.hpp>

struct PreprocSSD {
  bool operator()(infer_server::ModelIO* model_input, const infer_server::InferData& batch_data,
                  const infer_server::ModelInfo* model);
};  // struct PreprocSSD

struct PreprocSeg {
  bool operator()(infer_server::ModelIO* model_input, const infer_server::InferData& batch_data,
                  const infer_server::ModelInfo* model);
};  // struct PreprocSeg

typedef struct
{
  char* offset;
  size_t dataSize;
  //float datas[45];
  int index;
}ClassfBatch;

/**
 * @brief 整幅图像的亮度/对比度自适应统计量（与 predict_adaptive.py 中 adaptive_enhance 一致）
 */
struct AdaptiveStats {
  bool valid{false};               ///< 统计量是否有效
  float original_mean{0.0f};       ///< 整幅图原始亮度均值 / 255，取值 [0, 1]
  std::vector<float> band_min;     ///< 各波段原始最小值（uint8 值 0-255）
  std::vector<float> band_max;     ///< 各波段原始最大值（uint8 值 0-255）
};

/**
 * @brief 计算整幅图像的自适应统计量，供 SegPreProc 使用
 * @param image 合并后的多波段图像（CV_8UC(N)，NHWC 布局）
 * @param stats 输出统计量；图像类型不支持时置 valid=false
 */
void ComputeAdaptiveStats(const cv::Mat& image, AdaptiveStats* stats);

/**
 * @brief Seg 预处理器输入：patch 图像 + 可选的整幅图统计量
 * @note stats 为空或无效时，预处理器退化为按当前 patch 自身统计（逐 patch 自适应）
 */
struct SegPatch {
  cv::Mat patch;                            ///< CV_8UC(N)，NHWC 布局
  std::shared_ptr<AdaptiveStats> stats;     ///< 调用方计算的全图统计量，可为空
};

/**
 * @brief Seg 预处理器输入（方案v2 固定归一化版）
 * @note 整幅图像已在 samples/seg_fixed_norm.cpp 的 Process_ 中
 *       完成固定归一化 clip(x-5,0,40)/40，patch 为归一化后的 float32 数据，
 *       预处理器不再做逐波段 min-max 或亮度偏移。
 */
struct SegFixedPatch {
  cv::Mat patch;                            ///< CV_32FC(N)，NHWC 布局，值域 [0,1]
};

struct SegPreProc {
  bool operator()(infer_server::ModelIO* model_input, const infer_server::InferData& batch_data,
                  const infer_server::ModelInfo* model);
};  // struct SegPreProc

/**
 * @brief 固定归一化版本的预处理器（方案v2）
 * @note 输入 patch 为已完成整图固定归一化的 float32 数据，仅做 HWC 拷贝到模型输入。
 */
struct SegFixedPreProc {
  bool operator()(infer_server::ModelIO* model_input, const infer_server::InferData& batch_data,
                  const infer_server::ModelInfo* model);
};  // struct SegFixedPreProc

#endif  // EDK_SAMPLES_PREPROCESS_PREPROC_H_
