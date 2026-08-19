//
// Created by gaocheng on 2024/4/17.
//

#include "dataframe.h"

cv::Mat DataFrame::GetImage() {
  std::lock_guard<std::mutex> lk(mtx);
  if (!mat.empty()) {
    return mat;
  }
  return cv::Mat();
}
bool DataFrame::SetImage(cv::Mat &img) {
  std::lock_guard<std::mutex> lk(mtx);
  mat = img;
}
