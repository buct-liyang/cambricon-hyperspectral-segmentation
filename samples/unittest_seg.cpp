#include <iostream>
#include <dlfcn.h>
#include <sys/time.h>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <future>
#include <glog/logging.h>

#include "cnosd.h"
#include "easytrack/easy_track.h"
#include "cnis/infer_server.h"
#include "postprocess/postproc.h"
#include "collection.h"
#include "dataframe.h"
#include "infer_object.h"

using namespace cv;
using namespace infer_server;

typedef infer_server::Session_t (*Init_)(infer_server::InferServer *infer_server, 
                                        infer_server::SessionDesc &desc,
                                        std::shared_ptr<infer_server::Observer> observer,
                                        std::string model_path,
                                        std::atomic<float> *threshold);

typedef std::vector<int> (*Process_)(infer_server::InferServer *infer_server,
                                          infer_server::Session_t session,
                                          infer_server::PackagePtr in,
                                          infer_server::any user_data);

// 可视化函数，以防 seg.cpp 中的版本不可见
cv::Mat visualizeResult(const cv::Mat& pred_map) {
    cv::Mat rgb_map(pred_map.size(), CV_8UC3);
    std::map<int, cv::Vec3b> id_to_color_map = {
        {0, {0, 0, 0}}, {1, {255, 0, 0}}, {2, {0, 255, 0}}, 
        {3, {0, 0, 255}}, {4, {255, 255, 0}}, {5, {0, 255, 255}}
    };
    for (int r = 0; r < pred_map.rows; ++r) {
        for (int c = 0; c < pred_map.cols; ++c) {
            uchar class_id = pred_map.at<uchar>(r, c);
            rgb_map.at<cv::Vec3b>(r, c) = id_to_color_map.count(class_id) ? id_to_color_map[class_id] : cv::Vec3b(255, 255, 255);
        }
    }
    return rgb_map;
}

std::vector<cv::Mat> loadPNGsFromFolder(const std::string &folder_path) {
    std::vector<cv::String> file_paths_cv;
    cv::glob(folder_path + "/*.png", file_paths_cv);

    std::vector<std::string> file_paths;
    for(const auto& path : file_paths_cv) {
        file_paths.push_back(path);
    }
    std::sort(file_paths.begin(), file_paths.end());

    std::vector<cv::Mat> images;
    for (const auto &path : file_paths) {
        cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            LOG(WARNING) << "Could not read image: " << path;
        } else {
            images.push_back(img);
        }
    }
    std::cout << "Loaded " << images.size() << " images from " << folder_path << std::endl;
    return images;
}

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = 1;
    auto init_start = std::chrono::high_resolution_clock::now();

    // --- 请确保这些路径相对于您的运行目录是正确的 ---
    std::string model_path = "./models/material_1109.cambricon";
    std::string lib_path = "./lib/libSeg.so";
    std::string input_folder = "./your_path";

    void *handle = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        std::cerr << "Cannot open library: " << dlerror() << '\n';
        return -1;
    }
    LOG(INFO) << "Library " << lib_path << " loaded.";

    Init_ init_func = (Init_)dlsym(handle, "Init_");
    Process_ process_func = (Process_)dlsym(handle, "Process_");
    if (!init_func || !process_func) {
        std::cerr << "Cannot load symbols: " << dlerror() << '\n';
        dlclose(handle);
        return -1;
    }
    
    auto infer_server_ = std::make_shared<infer_server::InferServer>(0);
    infer_server::SessionDesc desc;
    desc.strategy = infer_server::BatchStrategy::STATIC;
    desc.engine_num = 4; // 使用多个引擎并行处理 patches
    desc.show_perf = true;
    desc.name = "Seg";
    
    auto session_ = init_func(infer_server_.get(), desc, nullptr, model_path, nullptr);

    // --- 增加对 Session 创建失败的检查 ---
    // if (!session_) {
    //     LOG(FATAL) << "Failed to initialize session. Please check logs for model loading errors.";
    //     dlclose(handle);
    //     google::ShutdownGoogleLogging();
    //     return -1;
    // }

    auto loadedTestImages = loadPNGsFromFolder(input_folder);
    if (loadedTestImages.empty()) {
        LOG(ERROR) << "No test images loaded from " << input_folder << ". Exiting.";
        return -1;
    }

    PackagePtr pack = Package::Create(loadedTestImages.size() + 1);
    pack->data[0]->Set<int>(loadedTestImages.size());
    for (size_t i = 0; i < loadedTestImages.size(); ++i) {
        pack->data[i + 1]->Set(loadedTestImages[i]);
    }

    const int width = loadedTestImages[0].cols;
    const int height = loadedTestImages[0].rows;
    
    CollectionPtr collection = std::make_shared<Collection>();
    std::vector<uchar> imageResultBuffer(width * height);
    std::vector<int> classifyedCount;

    

    collection->Add("ClassifyedResultBuffer", imageResultBuffer.data());
    collection->Add("ClassifyedCount", &classifyedCount);

    process_func(infer_server_.get(), session_, pack, collection);
    
    auto total_process_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double,std::milli>total_process_time = total_process_end - init_start;
    std::cout << "[Timing]totaltime:" << total_process_time.count() << "ms" << std::endl;

    cv::Mat outputImage(height, width, CV_8UC1, imageResultBuffer.data());
    cv::Mat visualizedImage = visualizeResult(outputImage);
    std::string save_path = "ClassifyResult2.png";

    if (cv::imwrite(save_path, visualizedImage)) {
        std::cout << "\n=====================================\n"
                  << "      Segmentation Complete!\n"
                  << "  Result saved to: " << save_path << "\n"
                  << "=====================================" << std::endl;
    } else {
        std::cerr << "Error: Failed to save result image to " << save_path << std::endl;
    }

    std::cout << "Pixel count per class:" << std::endl;
    for (size_t i = 0; i < classifyedCount.size(); ++i) {
        printf("  - Class ID %zu: %d pixels\n", i, classifyedCount[i]);
    }
    // auto total_process_end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double,std::milli>total_process_time = total_process_end - init_start;
    // std::cout << "[Timing]totaltime:" << total_process_time.count() << "ms" << std::endl;
    infer_server_->DestroySession(session_);
    dlclose(handle);
    google::ShutdownGoogleLogging();

    return 0;
}
