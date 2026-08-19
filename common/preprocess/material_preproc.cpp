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
#include "infer_object.h"

bool PreprocSeg::operator()(infer_server::ModelIO *model_input,
                            const infer_server::InferData &infer_data,
                            const infer_server::ModelInfo *model) {

  auto obj = infer_data.GetLref<std::shared_ptr<CNInferObject>>();
  auto img = obj->slice;

  cv::Mat sample_temp_bgr;
  if(img.channels() == 1 && model->InputShape(0)[3] == 3)
    cv::cvtColor(img, sample_temp_bgr, CV_GRAY2RGB);
  else if(img.channels() == 1 && model->InputShape(0)[3] == 4)
    cv::cvtColor(img, sample_temp_bgr, CV_GRAY2RGBA);
  else if(img.channels() == 3 && model->InputShape(0)[3] == 3)
    cv::cvtColor(img, sample_temp_bgr, CV_BGR2RGB);
  else if(img.channels() == 3 && model->InputShape(0)[3] == 4)
    cv::cvtColor(img, sample_temp_bgr, CV_BGR2RGBA);
  else
    sample_temp_bgr = img;

  cv::Mat sample_temp;
  cv::resize(sample_temp_bgr, sample_temp, cv::Size(model->InputShape(0)[2], model->InputShape(0)[1]), CV_INTER_LINEAR);

  model_input->buffers[0].CopyFrom(sample_temp.data, sample_temp.total() * sample_temp.elemSize());
  return true;
}