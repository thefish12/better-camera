#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#include <vector>
#include <iostream>

void tools_visualize(cv::Mat& image, const std::vector<cv::Rect>& faces, const std::vector<std::vector<cv::Point>>& landmarks, const std::vector<float>& scores);
std::vector<cv::Rect> get_face_loc(cv::Mat& image) ;
