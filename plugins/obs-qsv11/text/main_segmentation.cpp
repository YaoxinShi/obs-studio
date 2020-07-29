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
//#include <ext_list.hpp>
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
ExecutableNetwork executable_network;

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

void alpha_blend(cv::Mat img, int w, int h, std::vector<std::vector<size_t>> seg_class, int seg_w, int seg_h)
{
    static std::vector<std::vector<int>> colors = {
        //seems in B,G,R order, as OpenCV cv::Mat seems in B,G,R byte order. Anyway, this matches OpenVINO demo's writeOutputBmp() function.
        {128, 64,  128}, //road              Dark purple
        {232, 35,  244}, //sidewalk          Bright purple
        {70,  70,  70},  //building          Dark grey
        {156, 102, 102}, //wall              Lead
        {153, 153, 190}, //fence             Bright brown
        {153, 153, 153}, //pole              Bright grey
        {30,  170, 250}, //traffic light     Orange
        {0,   220, 220}, //traffic sign      Yellow
        {35,  142, 107}, //vegetation        Dark green
        {152, 251, 152}, //terrain           Bright green
        {180, 130, 70},  //sky               Cerulean
        {60,  20,  220}, //person            Orange red
        {0,   0,   255}, //rider             Red
        {142, 0,   0},   //car               Blue           <---- 13
        {70,  0,   0},   //truck             Dark blue
        {100, 60,  0},   //bus               Navy           <---- 15
        {90,  0,   0},   //train             Dark blue
        {230, 0,   0},   //motorcycle        Bright blue
        {32,  11,  119}, //bicycle           Dark red
        {0,   74,  111}, //ego-vehicle       Dark brown
        {81,  0,   81},  //                  Dark purple
    };

	int i, j;
	float alpha = 1;// 0.3;
	for (j = 0; j < h; j++)
	{
		for (i = 0; i < w; i++)
		{
			int r = img.data[(w * j + i) * 3];
			int g = img.data[(w * j + i) * 3 + 1];
			int b = img.data[(w * j + i) * 3 + 2];

			int seg_i = i * seg_w / w;
			int seg_j = j * seg_h / h;
			int seg_cls = seg_class[seg_j][seg_i];

			int seg_r = colors[seg_cls][0];
			int seg_g = colors[seg_cls][1];
			int seg_b = colors[seg_cls][2];

			img.data[(w * j + i) * 3] = r * (1 - alpha) + seg_r * alpha;
			img.data[(w * j + i) * 3 + 1] = g * (1 - alpha) + seg_g * alpha;
			img.data[(w * j + i) * 3 + 2] = b * (1 - alpha) + seg_b * alpha;
		}
	}
}

void maskToBoxes(std::vector<cv::Rect>& bboxes, int w, int h, std::vector<std::vector<size_t>> seg_class, int seg_w, int seg_h) {
    float min_area = 300;
    float min_height = 10;

    cv::Mat mask(h, w, CV_8UC1);
    int i, j;
    for (j = 0; j < h; j++)
    {
        for (i = 0; i < w; i++)
        {
            int seg_i = i * seg_w / w;
            int seg_j = j * seg_h / h;
            int seg_cls = seg_class[seg_j][seg_i];
	    //mask.data[w * j + i] = seg_cls;
	    mask.data[w * j + i] = (seg_cls == 13 || seg_cls == 15) ? 1 : 0;
        }
    }

    double min_val;
    double max_val;
    cv::minMaxLoc(mask, &min_val, &max_val);
    int max_bbox_idx = static_cast<int>(max_val);

    for (int i = 1; i <= max_bbox_idx; i++) { // loop all seg_class
        cv::Mat bbox_mask = (mask == i);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(bbox_mask, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
        for (int j = 0; j < contours.size(); j++) // loop current seg_class's all contour
        {
            cv::RotatedRect r = cv::minAreaRect(contours[j]);
            if (std::min(r.size.width, r.size.height) < min_height)
            {
                do_log(LOG_WARNING, "kill rect by height");
                continue;
            }
            if (r.size.area() < min_area)
            {
                do_log(LOG_WARNING, "kill rect by area");
                continue;
            }
            bboxes.emplace_back(r.boundingRect());
        }
    }
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

            // --------------------------- 2. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
            CNNNetReader networkReader;
            /** Read network model **/
            networkReader.ReadNetwork(model_path);

            /** Extract model name and load weights **/
            std::string binFileName = fileNameNoExt(model_path) + ".bin";
            networkReader.ReadWeights(binFileName);
            network_seg = networkReader.getNetwork();

            // --------------------------- 3. Configure input & output ---------------------------------------------
            // --------------------------- Prepare input blobs -----------------------------------------------------
            InputsDataMap inputInfo(network_seg.getInputsInfo());
            if (inputInfo.size() != 1)
            {
                do_log(LOG_WARNING, "Demo supports topologies only with 1 input");
                assert(0);
            }
            auto inputInfoItem = *inputInfo.begin();
            network_seg.setBatchSize(1);
            inputInfoItem.second->setPrecision(Precision::U8);
            // --------------------------- Prepare output blobs ----------------------------------------------------
            OutputsDataMap outputInfo(network_seg.getOutputsInfo());
            for (auto& item : outputInfo) {
                DataPtr outputData = item.second;
                if (!outputData) {
                    throw std::logic_error("output data pointer is not valid");
                }
		item.second->setPrecision(Precision::FP32);
            }
            // -----------------------------------------------------------------------------------------------------

            // --------------------------- 4. Loading model to the device ------------------------------------------
            /** Loading model to the device **/
            executable_network = ie.LoadNetwork(network_seg, "GPU");
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
        std::chrono::steady_clock::time_point infer_begin, infer_end, pp_begin, pp_end, draw_begin, draw_end, begin_frame, end_frame;
        begin_frame = std::chrono::steady_clock::now();

        cv::Mat demo_image = image.clone();
        cv::Size inference_image_size = image.size();
        if (gDemoMode != 0)
        {
            inference_image_size.width /= 2;
        }
        cv::Mat inference_image = image(cv::Rect(0, 0, inference_image_size.width, inference_image_size.height));

        std::vector<cv::Rect> rects;
        if (cnn_initialized) {
            infer_begin = std::chrono::steady_clock::now();
	    // --------------------------- 5. Create infer request -------------------------------------------------
            InferRequest infer_request = executable_network.CreateInferRequest();
            // -----------------------------------------------------------------------------------------------------

	    // --------------------------- 6. Prepare input --------------------------------------------------------
            InputsDataMap inputInfo(network_seg.getInputsInfo());
            auto inputInfoItem = *inputInfo.begin();
            cv::Size input_size = cv::Size(inputInfoItem.second->getTensorDesc().getDims()[3], inputInfoItem.second->getTensorDesc().getDims()[2]);
            cv::resize(inference_image, inference_image, input_size);

	    for (const auto& item : inputInfo) {
                /** Creating input blob **/
                Blob::Ptr input = infer_request.GetBlob(item.first);

		/** Fill input tensor with images. First r channel, then g and b channels **/
                size_t num_channels = input->getTensorDesc().getDims()[1];
                size_t image_size = input->getTensorDesc().getDims()[3] * input->getTensorDesc().getDims()[2];
		auto data = input->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();

		/** Iterate over all pixel in image (r,g,b) **/
                for (size_t pid = 0; pid < image_size; pid++) {
                    /** Iterate over all channels **/
                    for (size_t ch = 0; ch < num_channels; ++ch) {
                        /**          [images stride + channels stride + pixel id ] all in bytes            **/
                        data[ch * image_size + pid] = inference_image.data[pid * num_channels + ch];
                    }
                }
            }
            // -----------------------------------------------------------------------------------------------------

	    // --------------------------- 7. Do inference ---------------------------------------------------------
            infer_request.Infer();
            // -----------------------------------------------------------------------------------------------------
            infer_end = std::chrono::steady_clock::now();

            pp_begin = std::chrono::steady_clock::now();
            // --------------------------- 8. Process output -------------------------------------------------------
            OutputsDataMap outputInfo(network_seg.getOutputsInfo());
            std::string firstOutputName;
            for (auto& item : outputInfo) {
                if (firstOutputName.empty()) {
                    firstOutputName = item.first;
                }
            }
            const Blob::Ptr output_blob = infer_request.GetBlob(firstOutputName);
            const auto output_data = output_blob->buffer().as<float*>();

	    size_t N = output_blob->getTensorDesc().getDims().at(0);
            size_t C, H, W;

	    size_t output_blob_shape_size = output_blob->getTensorDesc().getDims().size();
            //slog::info << "Output blob has " << output_blob_shape_size << " dimensions" << slog::endl;

	    if (output_blob_shape_size == 3) {
                C = 1;
                H = output_blob->getTensorDesc().getDims().at(1);
                W = output_blob->getTensorDesc().getDims().at(2);
            }
            else if (output_blob_shape_size == 4) {
                C = output_blob->getTensorDesc().getDims().at(1);
                H = output_blob->getTensorDesc().getDims().at(2);
                W = output_blob->getTensorDesc().getDims().at(3);
            }
            else {
                do_log(LOG_WARNING, "Unexpected output blob shape. Only 4D and 3D output blobs are supported.");
                assert(0);
            }

	    size_t image_stride = W * H * C;

	    /** Iterating over all images **/
            for (size_t image = 0; image < N; ++image) {
                /** This vector stores pixels classes **/
                std::vector<std::vector<size_t>> outArrayClasses(H, std::vector<size_t>(W, 0));
                std::vector<std::vector<float>> outArrayProb(H, std::vector<float>(W, 0.));
                /** Iterating over each pixel **/
                for (size_t w = 0; w < W; ++w) {
                    for (size_t h = 0; h < H; ++h) {
                        /* number of channels = 1 means that the output is already ArgMax'ed */
                        if (C == 1) {
                            outArrayClasses[h][w] = static_cast<size_t>(output_data[image_stride * image + W * h + w]);
                        }
                        else {
                            /** Iterating over each class probability **/
                            for (size_t ch = 0; ch < C; ++ch) {
                                auto data = output_data[image_stride * image + W * H * ch + W * h + w];
                                if (data > outArrayProb[h][w]) {
                                    outArrayClasses[h][w] = ch;
                                    outArrayProb[h][w] = data;
                                }
                            }
                        }
                    }
                }
                /* alpha blend outArrayProb to demo image*/
	        alpha_blend(demo_image, demo_image.size().width, demo_image.size().height, outArrayClasses, W, H);
		maskToBoxes(rects, demo_image.size().width, demo_image.size().height, outArrayClasses, W, H);
            }
            // -----------------------------------------------------------------------------------------------------

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
	do_log(LOG_WARNING, "error: %s", ex.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
