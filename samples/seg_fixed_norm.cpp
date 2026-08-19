//
// Created by gaocheng on 2024/4/11.
//
// ============================================================================
// 方案v2（工业版，固定归一化）动态库入口：Init_ / Process_ / Observer
//
// 与 samples/seg.cpp 的区别（仅前处理部分）：
//   1. 不再计算整幅图的自适应统计量（逐波段 min-max + 亮度偏移，对应旧训练）；
//   2. 整幅图像统一做固定归一化：
//        x' = clip(x - 5, 0, 40) / 40
//      （参数来自方案v2统计：p1≈5、99% 像元 < 40；与 fanganv2/normalize.py
//        及 dataset_enhanced.py 的 norm_mode="fixed" 完全一致）
//   3. patch 直接携带归一化后的 float32 数据交给
//      SegFixedPreProc（仅 HWC 拷贝，不再做统计/增强）。
//
// 其余（滑窗、概率平均、ArgMax、最近邻放大回原尺寸、类别统计）与旧版一致。
// 注意：输入 45 个波段的顺序必须与训练一致（按波段文件名数字 1..45），
//       旧 predict 脚本中"把最后5个波段挪到前面"的写法已在方案v2修正。
// ============================================================================

#include <mutex>
#include <condition_variable>

#include "cnis/infer_server.h"
#include "cnis/processor.h"
#include "preprocess/preproc.h"
#include "postprocess/postproc.h"
#include "device/mlu_context.h"
#include "infer_object.h"
#include "sole.h"

#include <opencv2/opencv.hpp>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <glog/logging.h>

// --- 核心参数 ---
constexpr int NUM_CLASSES = 6;
constexpr int NUM_BANDS = 45;
const cv::Size WINDOW_SIZE(256, 256);
constexpr int STRIDE = 128;

// --- 方案v2 固定归一化参数（与 fanganv2/code/normalize.py 一致） ---
constexpr float kNORM_LOW = 5.0f;   // 暗底/基线（训练集 p1）
constexpr float kNORM_HIGH = 40.0f; // 截断上限（覆盖约 99% 动态范围）

using namespace infer_server;

/**
 * @brief 整幅图像固定归一化：clip(x - kNORM_LOW, 0, kNORM_HIGH) / kNORM_HIGH
 * @param src CV_8UC(N) 或 CV_32FC(N) 的原始 DN 图像（NHWC 布局）
 * @return CV_32FC(N)，值域 [0, 1]
 */
cv::Mat FixedNormalize(const cv::Mat& src) {
  const int rows = src.rows;
  const int cols = src.cols;
  const int channels = src.channels();
  cv::Mat dst(rows, cols, CV_32FC(channels));

  if (src.type() == CV_8UC(channels)) {
    for (int r = 0; r < rows; ++r) {
      const uchar* src_row = src.ptr<uchar>(r);
      float* dst_row = dst.ptr<float>(r);
      for (int i = 0; i < cols * channels; ++i) {
        float v = (static_cast<float>(src_row[i]) - kNORM_LOW) / kNORM_HIGH;
        dst_row[i] = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
      }
    }
  } else if (src.type() == CV_32FC(channels)) {
    for (int r = 0; r < rows; ++r) {
      const float* src_row = src.ptr<float>(r);
      float* dst_row = dst.ptr<float>(r);
      for (int i = 0; i < cols * channels; ++i) {
        float v = (src_row[i] - kNORM_LOW) / kNORM_HIGH;
        dst_row[i] = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
      }
    }
  } else {
    LOG(ERROR) << "FixedNormalize: unsupported input type " << src.type();
  }
  return dst;
}

/**
 * @brief Observer，用于聚合每个patch的推理结果（与旧版一致）
 */
class SegObserver : public Observer {
public:
    void Response(Status status, PackagePtr output, infer_server::any user_data) noexcept override {
        if (status != Status::SUCCESS) {
            return;
        }

        CollectionPtr ptrUserData = any_cast<CollectionPtr>(user_data);
        auto* full_probs = ptrUserData->Get<cv::Mat*>("FullProbs");
        auto* count_map = ptrUserData->Get<cv::Mat*>("CountMap");
        auto* mtx = ptrUserData->Get<std::mutex*>("Mutex");

        // 1. 获取模型输出的展平的概率向量 (NHWC 布局)
        auto& patch_probs_vec = output->data[0]->GetLref<std::vector<float>>();
        cv::Rect roi = output->data[0]->GetUserData<cv::Rect>();

        std::lock_guard<std::mutex> lock(*mtx);
        cv::Mat full_probs_roi = (*full_probs)(roi);

        size_t output_idx = 0;
        const int output_h = WINDOW_SIZE.height;
        const int output_w = WINDOW_SIZE.width;
        const int output_c = NUM_CLASSES;

        for (int h = 0; h < output_h; ++h) {
            for (int w = 0; w < output_w; ++w) {
                float* prob_vec_start = &patch_probs_vec[output_idx];
                auto& target_pixel_probs = full_probs_roi.at<cv::Vec<float, NUM_CLASSES>>(h, w);
                for (int c = 0; c < output_c; ++c) {
                    target_pixel_probs[c] += prob_vec_start[c];
                }
                output_idx += output_c;
            }
        }

        (*count_map)(roi) += 1;
    }

    static std::shared_ptr<SegObserver> ms_observer_;
};
std::shared_ptr<SegObserver> SegObserver::ms_observer_ =
    std::make_shared<SegObserver>();

/**
 * @brief 可视化函数 (与旧版一致)
 */
cv::Mat visualizeResult(const cv::Mat& pred_map) {
    cv::Mat rgb_map(pred_map.size(), CV_8UC3);
    std::map<int, cv::Vec3b> id_to_color_map = {
        {0, {0, 0, 0}}, {1, {0, 0, 255}}, {2, {0, 255, 0}},
        {3, {0, 255, 255}}, {4, {255, 0, 0}}, {5, {255, 255, 0}} // BGR format
    };
    for (int r = 0; r < pred_map.rows; ++r) {
        for (int c = 0; c < pred_map.cols; ++c) {
            rgb_map.at<cv::Vec3b>(r, c) = id_to_color_map[pred_map.at<uchar>(r, c)];
        }
    }
    return rgb_map;
}

extern "C" {
/**
 * @brief 初始化函数（使用固定归一化预处理器）
 */
infer_server::Session_t Init_(infer_server::InferServer* infer_server,
                              infer_server::SessionDesc& desc,
                              std::shared_ptr<infer_server::Observer> observer,
                              std::string model_path,
                              std::atomic<float>* threshold) {
    desc.model = infer_server->LoadModel(model_path, "subnet0");
    desc.host_input_layout.dtype = infer_server::DataType::FLOAT32;
    desc.host_input_layout.order = infer_server::DimOrder::NHWC;

    desc.preproc = infer_server::PreprocessorHost::Create();
    desc.postproc = infer_server::Postprocessor::Create();
    desc.preproc->SetParams("process_function",
                            infer_server::PreprocessorHost::ProcessFunction(
                                SegFixedPreProc()));
    desc.postproc->SetParams("process_function",
                             infer_server::Postprocessor::ProcessFunction(
                                 SegPostProcess(threshold)));

    infer_server::Session_t session_ = infer_server->CreateSession(desc,
        SegObserver::ms_observer_);
    LOG(INFO) << "Seg Session (Sliding Window, Fixed Norm v2) Initialized.";
    return session_;
}
}

extern "C" {
/**
 * @brief 核心处理函数（方案v2：整图固定归一化 + 滑窗推理）
 */
std::vector<int> Process_(infer_server::InferServer* infer_server,
                          infer_server::Session_t session,
                          infer_server::PackagePtr in,
                          infer_server::any user_data) {
    LOG(INFO) << "Starting HSI Seg process (fixed norm v2)...";

    int channel = in->data[0]->Get<int>();
    std::vector<cv::Mat> input_imgs;
    for (int c = 0; c < channel; ++c) {
        input_imgs.push_back(in->data[c + 1]->Get<cv::Mat>());
    }

    if (input_imgs.empty()) {
        LOG(ERROR) << "No input images found in package.";
        return {};
    }

    // 保存原始图像尺寸
    const int orig_H = input_imgs[0].rows;
    const int orig_W = input_imgs[0].cols;
    LOG(INFO) << "Original image size: " << orig_W << "x" << orig_H;

    // 1. 将每个通道图像缩小为原尺寸的 0.65（与旧版一致的部署策略）
    std::vector<cv::Mat> scaled_imgs;
    for (const auto& img : input_imgs) {
        cv::Mat scaled_img;
        cv::resize(img, scaled_img, cv::Size(), 0.65, 0.65, cv::INTER_AREA);
        scaled_imgs.push_back(scaled_img);
    }

    // 2. 合并缩小后的通道（原始 DN 值，CV_8UC45）
    cv::Mat hsi_image;
    cv::merge(scaled_imgs, hsi_image);
    const int scaled_H = hsi_image.rows;
    const int scaled_W = hsi_image.cols;
    LOG(INFO) << "Scaled image size: " << scaled_W << "x" << scaled_H;

    // 2.1 整幅图像固定归一化一次（方案v2：clip(x-5,0,40)/40，保留绝对亮度信息）
    cv::Mat hsi_norm = FixedNormalize(hsi_image);
    LOG(INFO) << "Fixed normalization applied (clip(x-" << kNORM_LOW
              << ",0," << kNORM_HIGH << ")/" << kNORM_HIGH << ").";

    // 3. 准备缩放后的结果聚合器
    cv::Mat full_probs = cv::Mat::zeros(scaled_H, scaled_W, CV_32FC(NUM_CLASSES));
    cv::Mat count_map = cv::Mat::zeros(scaled_H, scaled_W, CV_32S);
    std::mutex mtx;
    CollectionPtr ptrUserData = any_cast<CollectionPtr>(user_data);
    ptrUserData->Add("FullProbs", &full_probs);
    ptrUserData->Add("CountMap", &count_map);
    ptrUserData->Add("Mutex", &mtx);

    // 4. 滑窗并分发推理任务（patch 为已归一化的 float32 数据）
    std::string task_id = sole::uuid0().str();
    int patch_count = 0;
    for (int y = 0; y <= scaled_H - WINDOW_SIZE.height; y += STRIDE) {
        for (int x = 0; x <= scaled_W - WINDOW_SIZE.width; x += STRIDE) {
            cv::Rect roi(x, y, WINDOW_SIZE.width, WINDOW_SIZE.height);

            SegFixedPatch patch_data;
            patch_data.patch = hsi_norm(roi).clone();

            PackagePtr package = Package::Create(1);
            package->tag = task_id;
            package->data[0]->Set(patch_data);
            package->data[0]->SetUserData(roi);

            if (!infer_server->Request(session, package, user_data, -1)) {
                LOG(ERROR) << "Failed to submit request for patch at (" << x << ", " << y << ")";
            }
            patch_count++;
        }
    }
    LOG(INFO) << "Submitted " << patch_count << " patches for inference with task_id: " << task_id;

    // 5. 等待所有任务完成
    infer_server->WaitTaskDone(session, task_id);
    LOG(INFO) << "Inference for all patches completed.";

    // 6. 缩放后图像上的最终后处理 (平均, ArgMax)
    cv::Mat scaled_prediction = cv::Mat::zeros(scaled_H, scaled_W, CV_8UC1);
    for (int r = 0; r < scaled_H; ++r) {
        for (int c = 0; c < scaled_W; ++c) {
            int count = count_map.at<int>(r, c);
            if (count > 0) {
                auto& probs = full_probs.at<cv::Vec<float, NUM_CLASSES>>(r, c);
                for (int i = 0; i < NUM_CLASSES; ++i) probs[i] /= count;

                float max_prob = -1.0f;
                uchar max_id = 0;
                for (int i = 0; i < NUM_CLASSES; ++i) {
                    if (probs[i] > max_prob) {
                        max_prob = probs[i];
                        max_id = static_cast<uchar>(i);
                    }
                }
                scaled_prediction.at<uchar>(r, c) = max_id;
            }
        }
    }

    // 7. 将结果缩放回原始尺寸
    cv::Mat prediction_map;
    cv::resize(scaled_prediction, prediction_map, cv::Size(orig_W, orig_H),
               0, 0, cv::INTER_NEAREST); // 分割结果使用最近邻插值保持类别值

    // 8. 写入结果 buffer 并统计
    uchar* result_buffer = ptrUserData->Get<uchar*>("ClassifyedResultBuffer");
    memcpy(result_buffer, prediction_map.data, orig_H * orig_W);

    std::vector<int>& classifyedCount = *ptrUserData->Get<std::vector<int>*>("ClassifyedCount");
    classifyedCount.assign(NUM_CLASSES, 0);
    for (int i = 0; i < orig_H * orig_W; ++i) {
        uchar pixel_value = prediction_map.data[i];
        if (pixel_value < NUM_CLASSES) {
            classifyedCount[pixel_value]++;
        }
    }

    LOG(INFO) << "Seg process finished. Result scaled back to original size.";
    return std::vector<int>();
}

}