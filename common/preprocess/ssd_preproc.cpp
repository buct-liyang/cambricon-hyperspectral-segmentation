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
#include <glog/logging.h>
#include <memory>
#include <vector>
#include "cnis/infer_server.h"
#include "cnis/processor.h"
#include <opencv2/opencv.hpp>

#include "preproc.h"
#include "utils.h"
#include "collection.h"
#include "dataframe.h"

bool PreprocSSD::operator()(infer_server::ModelIO *model_input,
                            const infer_server::InferData &infer_data,
                            const infer_server::ModelInfo *model) {
  auto collection = infer_data.GetLref<CollectionPtr>();
  auto data = collection->Get<DataFramePtr>(kDataFrameTag);
  auto img = data->GetImage();
  if(img.channels() == 1)
    cv::cvtColor(img, img, CV_GRAY2BGR);
  auto shape = model->InputShape(0);
  cv::resize(img, img, cv::Size(shape[2], shape[1]));
  cv::Mat frame_rgb;
  cv::cvtColor(img, frame_rgb, cv::COLOR_BGR2RGBA);
  model_input->buffers[0].CopyFrom(frame_rgb.data, model_input->buffers[0].MemorySize());
  return true;
}