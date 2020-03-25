// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
#include <ext_list.hpp>
#include <inference_engine.hpp>

#include "cnn.hpp"
#include "main_text.h"
#include "image_grabber.hpp"

#include <obs-module.h>
#define do_log(level, format, ...) \
    blog(level, "[text detection: '%s'] " format, \
            "aaa", ##__VA_ARGS__)

#define SHOW_CV_OUTPUT_IMAGE 1
#define USE_OBS_INPUT 1

using namespace InferenceEngine;

// OpenVINO internal
//Cnn text_detection, text_recognition;
//std::map<std::string, InferencePlugin> plugins_for_devices;
CNNNetwork network_seg;

// in
//Cnn_input cnn_in;

// out
//std::vector<cv::Rect> rects_no_rotate;

// unique frame index
//int frame_num = 0;
//bool enable_roi = true;
//int gDemoMode = 0;

// cnn state
//--------------------------------------------------
//  OBS thead        Encode thread        CNN init thread        CNN thread
//..load QSV splugin..
//                        cnn_initialized=false
//                        ..init cnn..
//                        cnn_initialized=true
//            ..start recording..
//                                    cnn_started=false
//                                    ..create CNN thread..
//                                    cnn_started=true
//                                    cnn_idle=true
//            ..send CNN task..
//                                    cnn_idle=false
//                                    ..run inference..
//            ..skip CNN task..                ..run inference..
//            ..skip CNN task..                ..run inference..
//            ..skip CNN task..                ..run inference..
//                                    cnn_idle=true
//            ..send CNN task..
//                                    cnn_idle=false
//                                    ..run inference..
//            ..skip CNN task..                ..run inference..
//            ..skip CNN task..                ..run inference..
//            ..skip CNN task..                ..run inference..
//                                    cnn_idle=true
//            ..stop recording..
//                                    ..kill CNN thread..
//                                    cnn_started=false
//                                    cnn_idle=false

//bool cnn_initialized = false;
//bool cnn_started = false;
//bool cnn_idle = false;

static std::string fileNameNoExt(const std::string& filepath) {
	auto pos = filepath.rfind('.');
	if (pos == std::string::npos) return filepath;
	return filepath.substr(0, pos);
}

int cnn_init_seg()
{
    //std::vector<std::string> devices = { "GPU", "CPU" };
    Core ie;
    if (!cnn_initialized)
    {
        do_log(LOG_WARNING, "Init plugins");

        ie.GetVersions("GPU");

        std::string model_path = ".\\semantic-segmentation-adas-0001.xml";
        FILE *fh = fopen(model_path.c_str(), "r");
        if (fh == NULL)
        {
            model_path = "c:\\tmp\\semantic-segmentation-adas-0001.xml";
            fh = fopen(model_path.c_str(), "r");
            if (fh == NULL)
            {
                assert(0); // cannot file cn model
            }
            else
            {
                fclose(fh);
            }
        }
        else
        {
            fclose(fh);
        }

        if (!model_path.empty())
        {
            do_log(LOG_WARNING, "Init text recognition NN");

            CNNNetReader networkReader;
            /** Read network model **/
            networkReader.ReadNetwork(model_path);

            /** Extract model name and load weights **/
            std::string binFileName = fileNameNoExt(model_path) + ".bin";
            networkReader.ReadWeights(binFileName);
            network_seg = networkReader.getNetwork();
        }
        cnn_initialized = true;
        do_log(LOG_WARNING, "Init plugins, done");
    }
    return 1;
}

int seg_detection(uint8_t * pY, uint32_t width, uint32_t height, pthread_mutex_t * cnn_mutex) {
    try {
        int fn = frame_num;
        if (!cnn_initialized)
        {
            do_log(LOG_WARNING, "SHOULD NOT COME HERE! Skip text detection as cnn not initizlized");
            return 0;
        }
        if (!cnn_started)
        {
            do_log(LOG_WARNING, "SHOULD NOT COME HERE! Skip text detection as cnn not started");
            return 0;
        }
        if (cnn_mutex != NULL)
        {
            fn--; //as frame_num has been added 1 in encoding thread
            pthread_mutex_lock(cnn_mutex);
            if (!cnn_idle)
            {
                do_log(LOG_WARNING, "SHOULD NOT COME HERE! Skip text detection for frame %d", fn);
                return 0;
            }
            do_log(LOG_WARNING, "Begin text detection for frame %d", fn);
            cnn_idle = false;
            pthread_mutex_unlock(cnn_mutex);

        }

#if USE_OBS_INPUT
        cv::Mat image;
        cv::Mat imageYUV(height*3/2, width, CV_8UC1, (void*)pY);
        cv::cvtColor(imageYUV, image, cv::COLOR_YUV420sp2RGB, 3);
#else
        do_log(LOG_WARNING, "Init Image Grabber");
        std::string input_type = "image";
        std::string image_path = "c:\\tmp\\forza_540p.jpg";
        std::unique_ptr<Grabber> grabber = Grabber::make_grabber(input_type, image_path);
        cv::Mat image;
        grabber->GrabNextImage(&image);
        bool is_image = (input_type.find("image") != std::string::npos);
        while (!image.empty() || is_image) {
#endif
        do_log(LOG_WARNING, "-------------------------------------------------------");
        cv::Mat demo_image = image.clone();
        cv::Size inference_image_size = image.size();
        if (gDemoMode != 0)
        {
            inference_image_size.width /= 2;
        }
        cv::Mat inference_image = image(cv::Rect(0, 0, inference_image_size.width, inference_image_size.height));

        std::chrono::steady_clock::time_point infer_begin, infer_end, pp_begin, pp_end, draw_begin, draw_end, begin_frame, end_frame;
        begin_frame = std::chrono::steady_clock::now();
        std::vector<cv::Rect> rects;
        if (cnn_initialized) {
            infer_begin = std::chrono::steady_clock::now();
            //todo: inference
            infer_end = std::chrono::steady_clock::now();

	    pp_begin = std::chrono::steady_clock::now();
            //todo: postprocess
            pp_end = std::chrono::steady_clock::now();
        } else {
            rects.emplace_back(cv::Point2f(0.0f, 0.0f), cv::Size2f(100.0f, 100.0f));
        }

        if (cnn_mutex != NULL)
        {
            pthread_mutex_lock(cnn_mutex);
        }
        // copy from local "rects" to global "rects_no_rotate"
        rects_no_rotate.clear();
        for (const auto& rect : rects) {
            rects_no_rotate.emplace_back(rect);
        }
        if (cnn_mutex != NULL)
        {
            pthread_mutex_unlock(cnn_mutex);
        }

        draw_begin = std::chrono::steady_clock::now();
        int num_found = static_cast<int>(rects_no_rotate.size());
        for (const cv::Rect &rect : rects_no_rotate) {
            std::vector<cv::Point2f> points;
            if (rect.size() != cv::Size(0, 0) && cnn_initialized) {
                points.emplace_back(cv::Point2f(float(rect.x),float(rect.y)));
                points.emplace_back(cv::Point2f(float(rect.x+rect.width), float(rect.y)));
                points.emplace_back(cv::Point2f(float(rect.x+rect.width), float(rect.y+rect.height)));
                points.emplace_back(cv::Point2f(float(rect.x), float(rect.y+rect.height)));
            }

            for (size_t i = 0; i < points.size(); i++) {
                cv::line(demo_image, points[i], points[(i + 1) % points.size()], cv::Scalar(50, 205, 50), 2);
            }
        }
        draw_end = std::chrono::steady_clock::now();
        end_frame = std::chrono::steady_clock::now();
        double avg_time = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(end_frame - begin_frame).count());
        int fps = static_cast<int>(1000 / avg_time);

        do_log(LOG_WARNING, "=== inference %d (ms): ", std::chrono::duration_cast<std::chrono::milliseconds>(infer_end - infer_begin).count());
        do_log(LOG_WARNING, "=== postprocess %d (ms): ", std::chrono::duration_cast<std::chrono::milliseconds>(pp_end - pp_begin).count());
        do_log(LOG_WARNING, "=== draw %d (ms): ", std::chrono::duration_cast<std::chrono::milliseconds>(draw_end - draw_begin).count());

        if (SHOW_CV_OUTPUT_IMAGE)
        {
            cv::putText(demo_image,
                "inference(ms): " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(infer_end - infer_begin).count()) + 
                ", postprocess(ms): " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(pp_end - pp_begin).count()) + 
                ", draw(ms): " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(draw_end - draw_begin).count()) + 
                ", fps: " + std::to_string(fps) + \
                ", found: " + std::to_string(num_found) + \
                ", frame: " + std::to_string(fn),
                cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
            cv::namedWindow("Press any key to exit", cv::WINDOW_NORMAL);
            cv::resizeWindow("Press any key to exit", 960, 540);
            cv::imshow("Press any key to exit", demo_image);
            char k = cv::waitKey(10); // 10ms, 0 means infinite, cv::waitKey is a must for cv::imshow
            if (k == 27) cv::destroyAllWindows(); //key 27 is ESC
        }

#if ! USE_OBS_INPUT
        if (!is_image)
        {
            grabber->GrabNextImage(&image);
        }
        }
#endif
        // ---------------------------------------------------------------------------------------------------
        if (cnn_mutex != NULL)
        {
            pthread_mutex_lock(cnn_mutex);
            do_log(LOG_WARNING, "Done text detection for frame %d", fn);
            cnn_idle = true;
            pthread_mutex_unlock(cnn_mutex);
        }
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
