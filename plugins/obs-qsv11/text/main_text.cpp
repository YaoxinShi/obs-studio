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
#include "image_grabber.hpp"
#include "text_detection.hpp"
#include "text_recognition.hpp"

#define DISABLE_ROTATE_RECT 1
#define SHOW_CV_OUTPUT_IMAGE 1

using namespace InferenceEngine;


std::vector<cv::Point2f> floatPointsFromRotatedRect(const cv::RotatedRect &rect);
std::vector<cv::Point> boundedIntPointsFromRotatedRect(const cv::RotatedRect &rect, const cv::Size& image_size);
cv::Point topLeftPoint(const std::vector<cv::Point2f> & points, int *idx);
cv::Mat cropImage(const cv::Mat &image, const std::vector<cv::Point2f> &points, const cv::Size& target_size, int top_left_point_idx);
void setLabel(cv::Mat& im, const std::string label, const cv::Point & p);

bool rect_is_near(const cv::Rect r1, const cv::Rect r2)
{
    int delta = 3;
    bool ret = false;

    cv::Rect new_r1 = r1 + cv::Point(-delta, -delta);
    new_r1 = new_r1 + cv::Size(delta*2, delta*2);

    cv::Rect new_r2 = r2 + cv::Point(-delta, -delta);
    new_r2 = new_r2 + cv::Size(delta * 2, delta * 2);

    if (new_r1.contains(cv::Point(new_r2.x, new_r2.y)))
        ret = true;
    if (new_r1.contains(cv::Point(new_r2.x + new_r2.width, new_r2.y)))
        ret = true;
    if (new_r1.contains(cv::Point(new_r2.x, new_r2.y + new_r2.height)))
        ret = true;
    if (new_r1.contains(cv::Point(new_r2.x + new_r2.width, new_r2.y + new_r2.height)))
        ret = true;

    if (new_r2.contains(cv::Point(new_r1.x, new_r1.y)))
        ret = true;
    if (new_r2.contains(cv::Point(new_r1.x + new_r1.width, new_r1.y)))
        ret = true;
    if (new_r2.contains(cv::Point(new_r1.x, new_r1.y + new_r1.height)))
        ret = true;
    if (new_r2.contains(cv::Point(new_r1.x + new_r1.width, new_r1.y + new_r1.height)))
        ret = true;

    return ret;
}

cv::Rect combine_two_rect(const cv::Rect r1, const cv::Rect r2)
{
    int left = r1.x < r2.x ? r1.x : r2.x;
    int right = (r1.x + r1.width) > (r2.x + r2.width) ? (r1.x + r1.width) : (r2.x + r2.width);
    int top = r1.y < r2.y ? r1.y : r2.y;
    int bottom = (r1.y + r1.height) > (r2.y + r2.height) ? (r1.y + r1.height) : (r2.y + r2.height);

    return cv::Rect(left, top, right-left, bottom-top);
}

void merge_rect(std::vector<cv::Rect>& rects)
{
    int index = 0;
    cv::Rect rect, rect2;
    while (index < rects.size() - 1)
    {
    L_start:
        rect = rects.at(index);
        for (int j = index + 1; j < rects.size(); j++)
        {
            rect2 = rects.at(j);
            if (rect_is_near(rect, rect2))
            {
                rects.erase(rects.begin() + j);
                rects.erase(rects.begin() + index);
                rects.emplace_back(combine_two_rect(rect,rect2));
                goto L_start;
            }
        }
        index++;
    }
}

int clip(int x, int max_val) {
    return std::min(std::max(x, 0), max_val);
}

int txt_detection() {
    try {
        // ----------------------------- Parsing and validating input arguments ------------------------------

        double text_detection_postproc_time = 0;
        double text_recognition_postproc_time = 0;
        double text_crop_time = 0;
        double avg_time = 0;
        const double avg_time_decay = 0.8;

        const char kPadSymbol = '#';
        std::string kAlphabet = std::string("0123456789abcdefghijklmnopqrstuvwxyz") + kPadSymbol;

        const double min_text_recognition_confidence = 0.2;

        std::map<std::string, InferencePlugin> plugins_for_devices;
        std::vector<std::string> devices = {"GPU", "CPU"};

        float cls_conf_threshold = static_cast<float>(0.8);
        float link_conf_threshold = static_cast<float>(0.8);

        std::cout << "Init plugins" << std::endl;
        for (const auto &device : devices) {
            if (plugins_for_devices.find(device) != plugins_for_devices.end()) {
                continue;
            }
            InferencePlugin plugin = PluginDispatcher().getPluginByDevice(device);
            plugins_for_devices[device] = plugin;
        }

	std::string image_path = "c:\\tmp\\forza_540p.jpg";
	std::string text_detection_model_path = "c:\\tmp\\text-detection-0002_FP16.xml";
	std::string text_recognition_model_path = "";

        Cnn text_detection, text_recognition;

        if (!text_detection_model_path.empty())
        {
            std::cout << "Init text detection NN" << std::endl;
            text_detection.Init(text_detection_model_path, &plugins_for_devices[devices[0]], cv::Size(1280, 768));
        }

        if (!text_recognition_model_path.empty())
        {
            std::cout << "Init text recognition NN" << std::endl;
            text_recognition.Init(text_recognition_model_path, &plugins_for_devices[devices[1]]);
        }

        std::cout << "Init Image Grabber" << std::endl;
	std::string input_type = "image";
        std::unique_ptr<Grabber> grabber = Grabber::make_grabber(input_type, image_path);

        cv::Mat image;
        grabber->GrabNextImage(&image);

        bool is_image = (input_type.find("image") != std::string::npos);
        while (!image.empty() || is_image) {
            std::cout << "-------------------------------------------------------" << std::endl;
            cv::Mat demo_image = image.clone();
            cv::Size orig_image_size = image.size();

            std::chrono::steady_clock::time_point infer_begin, infer_end, pp_begin, pp_end, draw_begin, draw_end;

            std::chrono::steady_clock::time_point begin_frame = std::chrono::steady_clock::now();
            std::vector<cv::RotatedRect> rects;
            if (text_detection.is_initialized()) {
                infer_begin = std::chrono::steady_clock::now();
                auto blobs = text_detection.Infer(image);
                infer_end = std::chrono::steady_clock::now();

                pp_begin = std::chrono::steady_clock::now();
                rects = postProcess(blobs, orig_image_size, cls_conf_threshold, link_conf_threshold);
                pp_end = std::chrono::steady_clock::now();
                text_detection_postproc_time += std::chrono::duration_cast<std::chrono::milliseconds>(pp_end - pp_begin).count();
            } else {
                rects.emplace_back(cv::Point2f(0.0f, 0.0f), cv::Size2f(0.0f, 0.0f), 0.0f);
            }

            draw_begin = std::chrono::steady_clock::now();
	    int max_rect_num = 50;
            if (static_cast<int>(rects.size()) > max_rect_num) {
                std::sort(rects.begin(), rects.end(), [](const cv::RotatedRect & a, const cv::RotatedRect & b) {
                    return a.size.area() > b.size.area();
                });
                rects.resize(max_rect_num);
            }

#if DISABLE_ROTATE_RECT
            std::vector<cv::Rect> rects_no_rotate;
            for (const auto &rect : rects) {
                rects_no_rotate.emplace_back(rect.boundingRect());
            }
            //cv::groupRectangles(rects_no_rotate, 1, 2);
            //merge_rect(rects_no_rotate);
            int num_found = static_cast<int>(rects_no_rotate.size());
#else
            int num_found = text_recognition.is_initialized() ? 0 : static_cast<int>(rects.size());
#endif

#if DISABLE_ROTATE_RECT
            for (const cv::Rect &rect : rects_no_rotate) {
#else
            for (const auto &rect : rects) {
#endif
                cv::Mat cropped_text;
                std::vector<cv::Point2f> points;
                int top_left_point_idx = -1;

#if DISABLE_ROTATE_RECT
                if (rect.size() != cv::Size(0, 0) && text_detection.is_initialized()) {
#else
                if (rect.size != cv::Size2f(0, 0) && text_detection.is_initialized()) {
#endif
                    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#if DISABLE_ROTATE_RECT
                    points.emplace_back(cv::Point2f(float(rect.x),float(rect.y)));
                    points.emplace_back(cv::Point2f(float(rect.x+rect.width), float(rect.y)));
                    points.emplace_back(cv::Point2f(float(rect.x+rect.width), float(rect.y+rect.height)));
                    points.emplace_back(cv::Point2f(float(rect.x), float(rect.y+rect.height)));
#else
                    points = floatPointsFromRotatedRect(rect);
#endif
                    topLeftPoint(points, &top_left_point_idx);
                    cropped_text = cropImage(image, points, text_recognition.input_size(), top_left_point_idx);
                    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                    text_crop_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                } else {
                    cropped_text = image;
                }

                std::string res = "";
                double conf = 1.0;
                if (text_recognition.is_initialized()) {
                    auto blobs = text_recognition.Infer(cropped_text);
                    auto output_shape = blobs.begin()->second->getTensorDesc().getDims();
                    if (output_shape[2] != kAlphabet.length())
                        throw std::runtime_error("The text recognition model does not correspond to alphabet.");

                    float *ouput_data_pointer = blobs.begin()->second->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
                    std::vector<float> output_data(ouput_data_pointer, ouput_data_pointer + output_shape[0] * output_shape[2]);

                    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                    res = CTCGreedyDecoder(output_data, kAlphabet, kPadSymbol, &conf);
                    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                    text_recognition_postproc_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

                    res = conf >= min_text_recognition_confidence ? res : "";
                    num_found += !res.empty() ? 1 : 0;
                }

                if (1) {
                    for (size_t i = 0; i < points.size(); i++) {
                        std::cout << clip(static_cast<int>(points[i].x), image.cols - 1) << "," <<
                                     clip(static_cast<int>(points[i].y), image.rows - 1);
                        if (i != points.size() - 1)
                            std::cout << ",";
                    }

                    if (text_recognition.is_initialized()) {
                        std::cout << "," << res;
                    }

                    std::cout << std::endl;
                }

		if (SHOW_CV_OUTPUT_IMAGE)
		{
			if (!res.empty() || !text_recognition.is_initialized()) {
				for (size_t i = 0; i < points.size(); i++) {
					cv::line(demo_image, points[i], points[(i + 1) % points.size()], cv::Scalar(50, 205, 50), 2);
				}

				if (!points.empty() && !res.empty()) {
					setLabel(demo_image, res, points[top_left_point_idx]);
				}
			}
		}
            }
            std::chrono::steady_clock::time_point end_frame = std::chrono::steady_clock::now();

            if (avg_time == 0) {
                avg_time = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(end_frame - begin_frame).count());
            } else {
                auto cur_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_frame - begin_frame).count();
                avg_time = avg_time * avg_time_decay + (1.0 - avg_time_decay) * cur_time;
            }
            int fps = static_cast<int>(1000 / avg_time);

            if (SHOW_CV_OUTPUT_IMAGE)
	    {
                std::cout << "To close the application, press 'CTRL+C' or any key with focus on the output window" << std::endl;
                cv::putText(demo_image, "fps: " + std::to_string(fps) + " found: " + std::to_string(num_found),
                            cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 1);
                cv::imshow("Press any key to exit", demo_image);
                char k = cv::waitKey(3);
                if (k == 27) break;
            }

            draw_end = std::chrono::steady_clock::now();
            std::cout << "=== inference (ms): "
                << std::chrono::duration_cast<std::chrono::milliseconds>(infer_end - infer_begin).count() << std::endl;
            std::cout << "=== postprocess (ms): "
                << std::chrono::duration_cast<std::chrono::milliseconds>(pp_end - pp_begin).count() << std::endl;
            std::cout << "=== draw (ms): "
                << std::chrono::duration_cast<std::chrono::milliseconds>(draw_end - draw_begin).count() << std::endl;

            if (!is_image)
            {
                grabber->GrabNextImage(&image);
            }
        }
        // ---------------------------------------------------------------------------------------------------
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

std::vector<cv::Point2f> floatPointsFromRotatedRect(const cv::RotatedRect &rect) {
    cv::Point2f vertices[4];
    rect.points(vertices);

    std::vector<cv::Point2f> points;
    for (int i = 0; i < 4; i++) {
        points.emplace_back(vertices[i].x, vertices[i].y);
    }
    return points;
}

cv::Point topLeftPoint(const std::vector<cv::Point2f> & points, int *idx) {
    cv::Point2f most_left(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    cv::Point2f almost_most_left(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());

    int most_left_idx = -1;
    int almost_most_left_idx = -1;

    for (size_t i = 0; i < points.size() ; i++) {
        if (most_left.x > points[i].x) {
            if (most_left.x != std::numeric_limits<float>::max()) {
                almost_most_left = most_left;
                almost_most_left_idx = most_left_idx;
            }
            most_left = points[i];
            most_left_idx = i;
        }
        if (almost_most_left.x > points[i].x && points[i] != most_left) {
            almost_most_left = points[i];
            almost_most_left_idx = i;
        }
    }

    if (almost_most_left.y < most_left.y) {
        most_left = almost_most_left;
        most_left_idx = almost_most_left_idx;
    }

    *idx = most_left_idx;
    return most_left;
}

cv::Mat cropImage(const cv::Mat &image, const std::vector<cv::Point2f> &points, const cv::Size& target_size, int top_left_point_idx) {
    cv::Point2f point0 = points[top_left_point_idx];
    cv::Point2f point1 = points[(top_left_point_idx + 1) % 4];
    cv::Point2f point2 = points[(top_left_point_idx + 2) % 4];

    cv::Mat crop(target_size, CV_8UC3, cv::Scalar(0));

    std::vector<cv::Point2f> from{point0, point1, point2};
    std::vector<cv::Point2f> to{cv::Point2f(0.0f, 0.0f), cv::Point2f(static_cast<float>(target_size.width-1), 0.0f),
                                cv::Point2f(static_cast<float>(target_size.width-1), static_cast<float>(target_size.height-1))};

    cv::Mat M = cv::getAffineTransform(from, to);

    cv::warpAffine(image, crop, M, crop.size());

    return crop;
}

void setLabel(cv::Mat& im, const std::string label, const cv::Point & p) {
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.7;
    int thickness = 1;
    int baseline = 0;

    cv::Size text_size = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    auto text_position = p;
    text_position.x = std::max(0, p.x);
    text_position.y = std::max(text_size.height, p.y);

    cv::rectangle(im, text_position + cv::Point(0, baseline), text_position + cv::Point(text_size.width, -text_size.height), CV_RGB(50, 205, 50), cv::FILLED);
    cv::putText(im, label, text_position, fontface, scale, CV_RGB(255, 255, 255), thickness, 8);
}
