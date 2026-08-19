//
// Created by gaocheng on 2023/3/2.
//

#ifndef REGISTRATION_CENTER_INCLUDE_LOG_H_
#define REGISTRATION_CENTER_INCLUDE_LOG_H_

#include <glog/logging.h>

#ifndef PIPELINE_LOG
#define PIPELINE_LOG
#define LOGF(tag) LOG(FATAL) << "[Pipeline " << (#tag) << " FATAL] "
#define LOGE(tag) LOG(ERROR) << "[Pipeline " << (#tag) << " ERROR] "
#define LOGW(tag) LOG(WARNING) << "[Pipeline " << (#tag) << " WARN] "
#define LOGI(tag) LOG(INFO) << "[Pipeline " << (#tag) << " INFO] "
#define LOGD(tag) VLOG(1) << "[Pipeline " << (#tag) << " DEBUG] "
#define LOGT(tag) VLOG(2) << "[Pipeline " << (#tag) << " TRACE] "
#define VLOG1(tag) VLOG(1) << "[Pipeline " << (#tag) << " V1] "
#define VLOG2(tag) VLOG(2) << "[Pipeline " << (#tag) << " V2] "
#define VLOG3(tag) VLOG(3) << "[Pipeline " << (#tag) << " V3] "
#define VLOG4(tag) VLOG(4) << "[Pipeline " << (#tag) << " V4] "
#define VLOG5(tag) VLOG(5) << "[Pipeline " << (#tag) << " V5] "

#define LOGF_IF(tag, condition) LOG_IF(FATAL, condition) << "[Pipeline " << (#tag) << " FATAL] "
#define LOGE_IF(tag, condition) LOG_IF(ERROR, condition) << "[Pipeline " << (#tag) << " ERROR] "
#define LOGW_IF(tag, condition) LOG_IF(WARNING, condition) << << "[Pipeline " << (#tag) << " WARN] "
#define LOGI_IF(tag, condition) LOG_IF(INFO, condition) << "[Pipeline " << (#tag) << " INFO] "
#define VLOG1_IF(tag, condition) VLOG_IF(1, condition) << "[Pipeline " << (#tag) << " V1] "
#define VLOG2_IF(tag, condition) VLOG_IF(2, condition) << "[Pipeline " << (#tag) << " V2] "
#define VLOG3_IF(tag, condition) VLOG_IF(3, condition) << "[Pipeline " << (#tag) << " V3] "
#define VLOG4_IF(tag, condition) VLOG_IF(4, condition) << "[Pipeline " << (#tag) << " V4] "
#define VLOG5_IF(tag, condition) VLOG_IF(5, condition) << "[Pipeline " << (#tag) << " V5] "

#endif

#endif //REGISTRATION_CENTER_INCLUDE_LOG_H_
