//
// Created by gaocheng on 2024/4/17.
//

#ifndef DATAFRAME_H_
#define DATAFRAME_H_

#include <opencv2/opencv.hpp>
#include "collection.h"

enum class DataFormat {
  INVALID = -1,                 /*!< This frame is invalid. */
  PIXEL_FORMAT_YUV420_NV21 = 0, /*!< This frame is in the YUV420SP(NV21) format. */
  PIXEL_FORMAT_YUV420_NV12,     /*!< This frame is in the YUV420sp(NV12) format. */
  PIXEL_FORMAT_BGR24,           /*!< This frame is in the BGR24 format. */
  PIXEL_FORMAT_RGB24,           /*!< This frame is in the RGB24 format. */
  PIXEL_FORMAT_ARGB32,          /*!< This frame is in the ARGB32 format. */
  PIXEL_FORMAT_ABGR32,          /*!< This frame is in the ABGR32 format. */
  PIXEL_FORMAT_RGBA32,          /*!< This frame is in the RGBA32 format. */
  PIXEL_FORMAT_BGRA32           /*!< This frame is in the BGRA32 format. */
};


class DataFrame : public NonCopyable {
 public:
  DataFrame() = default;
  ~DataFrame() = default;

  cv::Mat GetImage();
  bool SetImage(cv::Mat &img);

  DataFormat fmt;  /*!< The format of the frame. */
  int width;       /*!< The width of the frame. */
  int height;      /*!< The height of the frame. */

 private:
  std::mutex mtx;
  cv::Mat mat; /*!< A Mat stores image. */
};

using DataFramePtr = std::shared_ptr<DataFrame>;

#endif //DATAFRAME_H_
