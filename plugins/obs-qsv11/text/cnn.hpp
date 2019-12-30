// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>

#include <util/threading.h>

using namespace InferenceEngine;

class Cnn {
  public:
    Cnn():is_initialized_(false), channels_(0), input_data_(nullptr), time_elapsed_(0), ncalls_(0) {}

    void Init(const std::string &model_path, Core & ie, const std::string & deviceName,
              const cv::Size &new_input_resolution = cv::Size());

    InferenceEngine::BlobMap Infer(const cv::Mat &frame);

    bool is_initialized() const {return is_initialized_;}

    size_t ncalls() const {return ncalls_;}
    double time_elapsed() const {return time_elapsed_;}

    const cv::Size& input_size() const {return input_size_;}

  private:
    std::string model_path_;
    bool is_initialized_;
    cv::Size input_size_;
    int channels_;
    float* input_data_;
    InferRequest infer_request_;
    std::vector<std::string> output_names_;

    double time_elapsed_;
    size_t ncalls_;
};

class Cnn_input
{
public:
	uint8_t * pY;
	uint32_t width;
	uint32_t height;
	pthread_mutex_t * cnn_mutex;
};

#define MULTI_THREAD 1
#define SSD_TEXT 0
#define FORZA_1115 0
#define FORZA_1227 1
#define DOTA_1213 0
#define DOTA_1225 0

int txt_detection(uint8_t * pY, uint32_t width, uint32_t height, pthread_mutex_t * cnn_mutex);
extern bool enable_roi;
extern int gDemoMode;
extern Cnn_input cnn_in;
extern std::vector<cv::Rect> rects_no_rotate;
extern int frame_num;
extern bool cnn_initialized;
extern bool cnn_idle;
extern bool cnn_started;

