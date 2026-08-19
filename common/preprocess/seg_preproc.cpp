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
/*************************************************************************
 * Copyright (C) [2021] by Cambricon, Inc. All rights reserved 
 * ... (版权信息保持不变) ...
 *************************************************************************/
// #include <glog/logging.h>
// #include <memory>
// #include <vector>
// #include "cnis/infer_server.h"
// #include "cnis/processor.h"
// #include <opencv2/opencv.hpp>

// #include "preproc.h"

// /**
//  * @brief 预处理，将HWC格式的cv::Mat转换为NCHW格式的planar data
//  */
// bool SegPreProc::operator()(infer_server::ModelIO *model_input,
//                                          const infer_server::InferData &infer_data,
//                                          const infer_server::ModelInfo *model) {
//     // 1. 从 infer_data 获取 cv::Mat 格式的 patch
//     const auto& patch = infer_data.GetLref<cv::Mat>();
    
//     // 2. 检查模型输入格式和维度
//     if (model->InputLayout(0).order != infer_server::DimOrder::NCHW) {
//         LOG(ERROR) << "Model input layout is not NCHW, this preprocessor is incompatible.";
//         return false;
//     }
//     const int input_c = model->InputShape(0)[1];
//     const int input_h = model->InputShape(0)[2];
//     const int input_w = model->InputShape(0)[3];

//     if (patch.channels() != input_c || patch.rows != input_h || patch.cols != input_w) {
//         LOG(ERROR) << "Patch dimension(" << patch.channels() << "c, " << patch.rows << "h, " << patch.cols << "w) "
//                    << "mismatches model input shape(" << input_c << "c, " << input_h << "h, " << input_w << "w)!";
//         return false;
//     }

//     // 3. 执行 HWC -> NCHW 的数据布局转换
//     std::vector<float> input_data_planar(input_c * input_h * input_w);
//     for (int c = 0; c < input_c; ++c) {
//         for (int h = 0; h < input_h; ++h) {
//             for (int w = 0; w < input_w; ++w) {
//                 size_t dst_idx = c * (input_h * input_w) + h * input_w + w;
//                 input_data_planar[dst_idx] = patch.at<cv::Vec<float, 45>>(h, w)[c]; // 假设45通道
//             }
//         }
//     }

//     // 4. 将转换后的数据拷贝到模型输入 buffer
//     model_input->buffers[0].CopyFrom(input_data_planar.data(), input_data_planar.size() * sizeof(float));

//     return true;
// }

#include <glog/logging.h>
#include <memory>
#include "cnis/infer_server.h"
#include "cnis/processor.h"
#include <opencv2/opencv.hpp>
#include "preproc.h"

#include <glog/logging.h>
#include <memory>
#include "cnis/infer_server.h"
#include "cnis/processor.h"
#include <opencv2/opencv.hpp>
#include "preproc.h"

#include <algorithm>

// =========================================================================
// 亮度/对比度自适应预处理（与 predict_adaptive.py 的 adaptive_enhance 一致）
// =========================================================================
namespace {

// 与 predict_adaptive.py 中 adaptive_enhance() 的默认参数保持一致
constexpr float kBrightnessOffset = 0.10f;    // brightness_offset
constexpr float kBrightnessThreshold = 0.50f; // brightness_threshold
constexpr float kMinMaxEps = 1e-6f;

// 对单个 patch 执行自适应增强：
//   1) 逐波段 min-max 对比度拉伸
//   2) 根据原始亮度均值（优先全图，否则当前 patch）动态调整亮度偏移
cv::Mat AdaptiveEnhance(const cv::Mat& patch, const AdaptiveStats* global_stats) {
  const int rows = patch.rows;
  const int cols = patch.cols;
  const int channels = patch.channels();

  // ---- 统计量：优先使用调用方提供的整幅图统计量，否则按当前 patch 统计 ----
  float original_mean = 0.0f;
  std::vector<float> band_min(channels, 255.0f);
  std::vector<float> band_max(channels, 0.0f);
  const bool use_global = global_stats != nullptr && global_stats->valid &&
                          static_cast<int>(global_stats->band_min.size()) == channels &&
                          static_cast<int>(global_stats->band_max.size()) == channels;
  if (use_global) {
    original_mean = global_stats->original_mean;
    band_min = global_stats->band_min;
    band_max = global_stats->band_max;
  } else {
    double sum = 0.0;
    for (int r = 0; r < rows; ++r) {
      const uchar* row = patch.ptr<uchar>(r);
      for (int c = 0; c < cols; ++c) {
        const uchar* px = row + c * channels;
        for (int ch = 0; ch < channels; ++ch) {
          const uchar v = px[ch];
          sum += v;
          if (v < band_min[ch]) band_min[ch] = static_cast<float>(v);
          if (v > band_max[ch]) band_max[ch] = static_cast<float>(v);
        }
      }
    }
    const double num_values = static_cast<double>(rows) * cols * channels;
    original_mean = static_cast<float>(sum / num_values / 255.0);
  }

  // ---- 自适应亮度偏移（与 Python 完全一致）----
  const float brightness_factor = std::max(0.0f, 1.0f - original_mean / kBrightnessThreshold);
  const float adaptive_offset = kBrightnessOffset * brightness_factor;

  // ---- 逐波段 min-max 拉伸 + 亮度偏移 + clip 到 [0,1] ----
  cv::Mat normalized(rows, cols, CV_32FC(channels));
  for (int r = 0; r < rows; ++r) {
    const uchar* src_row = patch.ptr<uchar>(r);
    float* dst_row = normalized.ptr<float>(r);
    for (int c = 0; c < cols; ++c) {
      const uchar* src_px = src_row + c * channels;
      float* dst_px = dst_row + c * channels;
      for (int ch = 0; ch < channels; ++ch) {
        const float v = static_cast<float>(src_px[ch]);
        const float bmin = band_min[ch];
        const float bmax = band_max[ch];
        float val = (bmax > bmin + kMinMaxEps) ? (v - bmin) / (bmax - bmin) : 0.0f;
        val += adaptive_offset;
        val = std::min(1.0f, std::max(0.0f, val));
        dst_px[ch] = val;
      }
    }
  }
  return normalized;
}

}  // namespace

// 计算整幅图像的自适应统计量（定义在全局命名空间，供 samples/seg.cpp 调用）
void ComputeAdaptiveStats(const cv::Mat& image, AdaptiveStats* stats) {
  const int channels = image.channels();
  if (channels <= 0 || image.type() != CV_8UC(channels)) {
    LOG(WARNING) << "ComputeAdaptiveStats: unsupported image type, adaptive stats disabled.";
    return;
  }
  stats->band_min.assign(channels, 255.0f);
  stats->band_max.assign(channels, 0.0f);
  double sum = 0.0;
  for (int r = 0; r < image.rows; ++r) {
    const uchar* row = image.ptr<uchar>(r);
    for (int c = 0; c < image.cols; ++c) {
      const uchar* px = row + c * channels;
      for (int ch = 0; ch < channels; ++ch) {
        const uchar v = px[ch];
        sum += v;
        if (v < stats->band_min[ch]) stats->band_min[ch] = static_cast<float>(v);
        if (v > stats->band_max[ch]) stats->band_max[ch] = static_cast<float>(v);
      }
    }
  }
  const double num_values = static_cast<double>(image.rows) * image.cols * channels;
  stats->original_mean = static_cast<float>(sum / num_values / 255.0);
  stats->valid = true;
}

/**
 * @brief 预处理，适配 NHWC 和 UINT8 布局的模型 (最终正确版)
 * @note  使用框架提供的 CopyFrom API 将 CPU 数据填入框架提供的 CPU 缓冲区。
 */
bool SegPreProc::operator()(infer_server::ModelIO *model_input,
                                         const infer_server::InferData &infer_data,
                                         const infer_server::ModelInfo *model) {
    // 1. 从 infer_data 获取输入 patch 及调用方计算的全图自适应统计量
    const auto& patch_data = infer_data.GetLref<SegPatch>();
    const AdaptiveStats* adaptive_stats = patch_data.stats.get();

    // 2. 验证模型布局是否为 NHWC (可选，但推荐)
    const auto& model_layout = model->InputLayout(0);
    if (model_layout.order != infer_server::DimOrder::NHWC) {
        LOG(ERROR) << "Model input layout is not NHWC as expected.";
        return false;
    }

    // 3. 确保 patch 内存连续
    cv::Mat patch;
    if (patch_data.patch.isContinuous()) {
        patch = patch_data.patch;
    } else {
        patch = patch_data.patch.clone();
    }

    // 4. 校验输入为 CV_8UC(N) 的原始 uint8 数据（自适应增强的输入要求）
    const int channels = patch.channels();
    if (patch.type() != CV_8UC(channels)) {
        LOG(ERROR) << "Patch type is not CV_8UC" << channels
                   << ", adaptive enhancement requires uint8 input.";
        return false;
    }

    // ======================= 核心修改：亮度/对比度自适应增强 =======================
    // 与 predict_adaptive.py 的 adaptive_enhance() 保持一致：
    //   1) 逐波段 min-max 对比度拉伸
    //   2) 根据原始亮度均值（优先整幅图，否则当前 patch）动态调整亮度偏移
    cv::Mat normalized_patch = AdaptiveEnhance(patch, adaptive_stats);

    // 5. 计算新的源数据大小 (现在是 float 类型)
    const size_t src_data_size = normalized_patch.total() * normalized_patch.elemSize(); // elemSize for CV_32F is 4 bytes

    // 6. 使用框架的 CopyFrom API，将自适应增强后的 patch 填入模型输入 buffer
    try {
        model_input->buffers[0].CopyFrom(normalized_patch.data, src_data_size);
    } catch (const std::exception& e) {
        LOG(FATAL) << "CopyFrom failed with exception: " << e.what() << ". "
                   << "Source size: " << src_data_size << ", "
                   << "Destination buffer capacity: " << model_input->buffers[0].MemorySize();
        return false;
    }

    return true;
}