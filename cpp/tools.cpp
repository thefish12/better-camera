// #include "opencv2/opencv.hpp"
// #include "opencv2/dnn.hpp"
// #include <vector>
// #include <iostream>
#include "tools.hpp"

void tools_visualize(cv::Mat& image, const std::vector<cv::Rect>& faces, const std::vector<std::vector<cv::Point>>& landmarks, const std::vector<float>& scores) {
    cv::Mat output = image.clone();
    for (size_t i = 0; i < faces.size(); ++i) {
        // Draw face bounding box
        cv::rectangle(output, faces[i], cv::Scalar(0, 255, 0), 2);
        // Draw landmarks
        for (const auto& point : landmarks[i]) {
            cv::circle(output, point, 2, cv::Scalar(255, 0, 0), 2);
        }
        // Put score
        cv::putText(output, cv::format("%.4f", scores[i]), cv::Point(faces[i].x, faces[i].y + 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
    }
    cv::imshow("Output", output);
    cv::waitKey(0);
}

std::vector<cv::Rect> get_face_loc(cv::Mat& image) {
    float score_threshold = 0.85;
    float nms_threshold = 0.3;
    int backend = cv::dnn::DNN_BACKEND_DEFAULT;
    int target = cv::dnn::DNN_TARGET_CPU;

    // Instantiate yunet
    auto yunet = cv::FaceDetectorYN::create(
        "./face_detection_yunet_2023mar.onnx",
        "",
        cv::Size(320, 320),
        score_threshold,
        nms_threshold,
        5000,
        backend,
        target
    );

    yunet->setInputSize(cv::Size(image.cols, image.rows));
    std::vector<cv::Rect> faces_rects;
    cv::Mat faces;
    yunet->detect(image, faces);

    if (faces.empty()) {
        return faces_rects;
    }

    for (int i = 0; i < faces.rows; ++i) {
        cv::Rect face = cv::Rect(faces.at<float>(i, 0), faces.at<float>(i, 1), faces.at<float>(i, 2), faces.at<float>(i, 3));
        faces_rects.push_back(face);
    }

    bool vis = true;
    if (vis) {
        std::cout << "\n\ntools: " << faces_rects.size() << std::endl;
        // Note: landmarks and scores are not handled in this example
        std::vector<std::vector<cv::Point>> landmarks; // Placeholder
        std::vector<float> scores; // Placeholder
        tools_visualize(image, faces_rects, landmarks, scores);
    }

    return faces_rects;
}

// int main() {
//     std::string model = "./face_detection_yunet_2023mar.onnx";
//     cv::Mat image = cv::imread("fulltest.jpg");
//     if (image.empty()) {
//         std::cerr << "Could not read the image" << std::endl;
//         return 1;
//     }
//     get_face_loc(image);
//     return 0;
// }