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
#include <vector>
#include <cstring>
#include "cnis/infer_server.h"
#include "cnis/processor.h"
#include "postproc.h"

/**
 * @brief 后处理，仅提取模型输出的原始概率数组
 */
bool SegPostProcess::operator()(infer_server::InferData* result, const infer_server::ModelIO& model_output,
                                             const infer_server::ModelInfo* model) {
    // 1. 获取模型输出 buffer 的指针和大小
    // 模型的输出通常是 FLOAT32
    const float* output_data = reinterpret_cast<const float*>(model_output.buffers[0].Data());
    size_t num_elements = model_output.buffers[0].MemorySize() / sizeof(float);

    // 2. 将原始输出拷贝到一个 vector 中
    std::vector<float> patch_probabilities(num_elements);
    memcpy(patch_probabilities.data(), output_data, model_output.buffers[0].MemorySize());

    // 3. 将此 vector 设置为当前任务的结果，传递给 Observer
    result->Set(patch_probabilities);

    return true;
}
// #include <algorithm>
// #include <utility>
// #include <vector>

// #include "cnis/infer_server.h"
// #include "cnis/processor.h"
// #include "postproc.h"
// #include "collection.h"
// #include "infer_object.h"
// #include "dataframe.h"

// bool SegPostProcess::operator()(infer_server::InferData* result, const infer_server::ModelIO& model_output,
//                                 const infer_server::ModelInfo* model) {
//   // auto collection = result->GetUserData<CollectionPtr>();
//   // auto inferobjs = collection->Get<InferObjsPtr>(kInferObjsTag);
//   // auto frame = collection->Get<DataFramePtr>(kDataFrameTag)->GetImage();

//   // int image_w = frame.cols;
//   // int image_h = frame.rows;

//   // int model_input_w = model->InputShape(0)[2];
//   // int model_input_h = model->InputShape(0)[1];
//   // if (model->InputLayout(0).order == infer_server::DimOrder::NCHW) {
//   //   model_input_w = model->InputShape(0)[3];
//   //   model_input_h = model->InputShape(0)[2];
//   // }

//   // float scaling_factors = std::min(1.0 * model_input_w / image_w, 1.0 * model_input_h / image_h);

//   // // scaled size
//   // const int scaled_w = scaling_factors * image_w;
//   // const int scaled_h = scaling_factors * image_h;

//   // const float* data = reinterpret_cast<const float*>(model_output.buffers[0].Data());
//   // int box_num = data[0];
//   // constexpr int box_step = 7;
//   // for (int bi = 0; bi < box_num; ++bi) {
//   //   auto obj = std::make_shared<CNInferObject>();
//   //   float left = data[64 + bi * box_step + 3];
//   //   float right = data[64 + bi * box_step + 5];
//   //   float top = data[64 + bi * box_step + 4];
//   //   float bottom = data[64 + bi * box_step + 6];

//   //   // rectify
//   //   left = (left - (model_input_w - scaled_w) / 2) / scaled_w;
//   //   right = (right - (model_input_w - scaled_w) / 2) / scaled_w;
//   //   top = (top - (model_input_h - scaled_h) / 2) / scaled_h;
//   //   bottom = (bottom - (model_input_h - scaled_h) / 2) / scaled_h;
//   //   left = Clip(left);
//   //   right = Clip(right);
//   //   top = Clip(top);
//   //   bottom = Clip(bottom);

//   //   obj->label = static_cast<int>(data[64 + bi * box_step + 1]);
//   //   obj->score = data[64 + bi * box_step + 2];
//   //   obj->bbox.x = left;
//   //   obj->bbox.y = top;
//   //   obj->bbox.w = std::min(1.0f - obj->bbox.x, right - left);
//   //   obj->bbox.h = std::min(1.0f - obj->bbox.y, bottom - top);


//   //   if ((threshold->load() > 0 && obj->score < threshold->load()) || obj->bbox.w <= 0 || obj->bbox.h <= 0) continue;

//   //   obj->normal_bbox.x1 = left * frame.cols;
//   //   obj->normal_bbox.y1 = top * frame.rows;
//   //   obj->normal_bbox.x2 = right * frame.cols;
//   //   obj->normal_bbox.y2 = bottom * frame.rows;
//   //   obj->width = obj->normal_bbox.x2 - obj->normal_bbox.x1;
//   //   obj->height = obj->normal_bbox.y2 - obj->normal_bbox.y1;
//   //   cv::Rect rect(obj->normal_bbox.x1, obj->normal_bbox.y1, obj->width, obj->height);
//   //   obj->slice = cv::Mat(frame, rect).clone();
//   //   inferobjs->objs_.emplace_back(obj);
//   // }
//   float* scores = model_output.buffers[0].Data();

//   float max_val = scores[0];
//   //printf("%f,%f,%f,%f,%f\n", scores[0], scores[1], scores[2], scores[3], scores[4]);
//   int max_idx = 0;
//   for (int j = 1; j < 5; ++j) {
//       if (scores[j] > max_val) {
//           max_val = scores[j];
//           max_idx = j;
//       }
//   }

//   //int idx = result->GetUserData<int>();
//   result->Set(max_idx + 1);

//   //printf("Inx:%d,Class PostProcess:%d\n", max_idx + 1);

//   return true;
// }