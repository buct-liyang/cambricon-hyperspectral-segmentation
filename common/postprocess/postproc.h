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
#ifndef EDK_SAMPLES_POSTPROCESS_POSTPROC_H_
#define EDK_SAMPLES_POSTPROCESS_POSTPROC_H_

#include "cnis/contrib/video_helper.h"
#include "cnis/infer_server.h"
#include "cnis/processor.h"

struct DetectObject {
  int label;
  float score;
  infer_server::video::BoundingBox bbox;
};  // struct DetectObject

struct FrameSize {
  int width;
  int height;
};  // struct FrameSize


inline float Clip(float x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }

struct PostprocSeg {
  std::atomic<float> *threshold;

  explicit PostprocSeg(std::atomic<float> *_threshold) : threshold(_threshold) {}

  bool operator()(infer_server::InferData* result, const infer_server::ModelIO& model_output,
                  const infer_server::ModelInfo* model);
};  // struct PostprocSeg


struct PostprocSSD {
  std::atomic<float> *threshold;

  explicit PostprocSSD(std::atomic<float> *_threshold) : threshold(_threshold) {}

  bool operator()(infer_server::InferData* result, const infer_server::ModelIO& model_output,
                  const infer_server::ModelInfo* model);

};  // struct PostprocSSD


// ====================== 高光谱分割后处理 ======================
struct SegPostProcess {
  std::atomic<float> *threshold;

  // 添加一个无参数的默认构造函数，并将 threshold 初始化为 nullptr
  SegPostProcess() : threshold(nullptr) {}

  // 保留原来的构造函数，以兼容可能存在的旧代码
  explicit SegPostProcess(std::atomic<float> *_threshold) : threshold(_threshold) {}

  // 声明 operator()
  bool operator()(infer_server::InferData* result, const infer_server::ModelIO& model_output,
                  const infer_server::ModelInfo* model);
};
// =============================================================

#endif  // EDK_SAMPLES_POSTPROCESS_POSTPROC_H_
