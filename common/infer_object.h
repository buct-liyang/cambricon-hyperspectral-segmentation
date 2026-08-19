//
// Created by gaocheng on 2024/4/17.
//

#ifndef INFER_OBJECT_H_
#define INFER_OBJECT_H_

#include <iostream>
#include <vector>
#include <map>
#include <mutex>
#include <opencv2/opencv.hpp>

#include "collection.h"

struct InferBoundingBox {
  float x;  ///< The x-axis coordinate in the upper left corner of the bounding box.
  float y;  ///< The y-axis coordinate in the upper left corner of the bounding box.
  float w;  ///< The width of the bounding box.
  float h;  ///< The height of the bounding box.
};

struct DetectionBox {
  uint16_t x1;
  uint16_t y1;
  uint16_t x2;
  uint16_t y2;
};

typedef struct {
  int id = -1;      ///< The unique ID of the segmentation. The value -1 means invalid.
  int value = -1;   ///< The label value of the segmentation.
  float score = 0;  ///< The label score of the segmentation.
} CNInferAttr;

class CNInferObject {
 public:
  CNInferObject() = default;
  ~CNInferObject() = default;

  std::string id;           ///< The ID of the segmentation (label value).
  std::string track_id;     ///< The tracking result.
  int label = -1;
  int cls_label = -1;
  float score = 0;              ///< The label score.
  InferBoundingBox bbox;  ///< The object normalized coordinates.
  DetectionBox normal_bbox;
  Collection collection;    ///< User-defined structured information.
  int width;
  int height;
  cv::Mat slice;


 private:
  std::map<std::string, CNInferAttr> attributes_;
  std::mutex attribute_mutex_;
};

struct CNInferObjs : public NonCopyable {
  std::vector<std::shared_ptr<CNInferObject>> objs_;  /// The objects storing inference results.
  std::mutex mutex_;   /// mutex of CNInferObjs
};

using InferObjsPtr = std::shared_ptr<CNInferObjs>;

#endif //INFER_OBJECT_H_
