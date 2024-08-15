// #define USE_IPEX
// #define USE_CUDA
// #define USE_BF16
// #define JITTER
#define OUTVID

#include <opencv2/opencv.hpp>
#include <opencv2/freetype.hpp>
#include <opencv2/videoio.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/nn/functional.h>
// #include <torchvision/vision.h>
// #include <argparse/argparse.hpp>
#include <random>
#include <chrono>
#include <iostream>
#include <fstream>
#include <filesystem>
#include "model.hpp"
#include "tools.hpp"
// #include "fail2link.hpp"

// #include<114/514.h>

#ifdef USE_IPEX
    #include <intel-extension-for-pytorch/csrc/cpu/vec/vec.h>
#endif
// argparse::ArgumentParser program("example-app");

// namespace fs = std::filesystem;

// program.add_argument("--video")
//     .help("input video path. live cam is used when not specified")
//     .default_value(std::string(""));
// program.add_argument("--model_weight")
//     .help("path to model weights file")
//     .default_value(std::string("data/model_weights.pth"));
// program.add_argument("--jitter")
//     .help("jitter bbox n times, and average results")
//     .default_value(0)
//     .scan<'i', int>();
// program.add_argument("-save_vis")
//     .help("saves output as video")
//     .default_value(false)
//     .implicit_value(true);
// program.add_argument("-save_text")
//     .help("saves output as text")
//     .default_value(false)
//     .implicit_value(true);
// program.add_argument("-display_off")
//     .help("do not display frames")
//     .default_value(false)
//     .implicit_value(true);
// program.add_argument("-cuda")
//     .help("use cuda")
//     .default_value(false)
//     .implicit_value(true);
// program.add_argument("-ipex")
//     .help("use intel extension for pytorch")
//     .default_value(false)
//     .implicit_value(true);
// program.add_argument("-bf16")
//     .help("use bfloat16")
//     .default_value(false)
//     .implicit_value(true);

// auto args = program.parse_args();

torch::Tensor preprocess_image(const cv::Mat& image) {
    // 调整图像大小
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(224, 224));

    // 中心裁剪
    int crop_size = 224;
    int x = (resized_image.cols - crop_size) / 2;
    int y = (resized_image.rows - crop_size) / 2;
    cv::Rect crop_region(x, y, crop_size, crop_size);
    cv::Mat cropped_image = resized_image(crop_region);

    // 转换为张量
    torch::Tensor img_tensor = torch::from_blob(cropped_image.data, {cropped_image.rows, cropped_image.cols, 3}, torch::kByte);

    // 交换维度从HWC到CHW
    img_tensor = img_tensor.permute({2, 0, 1});

    // 转换为浮点型并归一化到[0, 1]
    img_tensor = img_tensor.toType(torch::kFloat) / 255.0;

    // 归一化
    img_tensor = torch::data::transforms::Normalize<>(
        {0.485, 0.456, 0.406}, 
        {0.229, 0.224, 0.225}
    )(img_tensor);

    return img_tensor;
}


// const std::string CNN_FACE_MODEL = "data/mmod_human_face_detector.dat"; // from http://dlib.net/files/mmod_human_face_detector.dat.bz2

#ifdef JITTER
std::tuple<float, float, float, float> bbox_jitter(float bbox_left, float bbox_top, float bbox_right, float bbox_bottom) {
    float cx = (bbox_right + bbox_left) / 2.0;
    float cy = (bbox_bottom + bbox_top) / 2.0;
    float scale = static_cast<float>(rand()) / RAND_MAX * 0.4 + 0.8;
    bbox_right = (bbox_right - cx) * scale + cx;
    bbox_left = (bbox_left - cx) * scale + cx;
    bbox_top = (bbox_top - cy) * scale + cy;
    bbox_bottom = (bbox_bottom - cy) * scale + cy;
    return std::make_tuple(bbox_left, bbox_top, bbox_right, bbox_bottom);
}
#endif
void drawrect(cv::Mat &image, cv::Rect rect, cv::Scalar color, int thickness) {
    cv::rectangle(image, rect, color, thickness);
}

void run(const std::string &video_path, const std::string &model_weight, int jitter, bool vis, bool display_off, bool save_text) {
    auto total_start_time = std::chrono::high_resolution_clock::now();

    // set up vis settings
    cv::Scalar colors[10];
    for (int i = 0; i < 10; ++i) {
        colors[i] = cv::Scalar(0, 255 * i / 10, 255 * (10 - i) / 10);
    }
    std::cerr<<"Start loading font file..."<<std::endl;
    cv::Ptr<cv::freetype::FreeType2> ft2;
    ft2 = cv::freetype::createFreeType2();
    ft2->loadFontData("data/arial.ttf", 0);
    std::cerr<<"Font file loaded."<<std::endl;

    // set up video source
    std::cerr<<"Start loading video..."<<std::endl;
    cv::VideoCapture cap;
    std::string video_output_path = video_path;
    if (video_path.empty()) {
        cap.open(0);
        video_output_path = "live.avi";
    } else {
        cap.open(video_path);
    }
    std::cerr<<"Video loaded."<<std::endl;

    // set up output file
    std::cerr<<"Start loading output file..."<<std::endl;
    std::ofstream f;
    if (save_text) {
        std::string outtext_name = std::filesystem::path(video_output_path).stem().string() + "_output.txt";
        f.open(outtext_name);
    }
    #ifdef OUTVID
    cv::VideoWriter outvid;
    if (vis) {
        std::string outvis_name = std::filesystem::path(video_output_path).stem().string() + "_output.avi";
        int imwidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int imheight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        outvid.open(outvis_name, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), cap.get(cv::CAP_PROP_FPS), cv::Size(imwidth, imheight));
    }
    #endif
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream or file" << std::endl;
        return;
    }
    std::cerr<<"Output file loaded."<<std::endl;

    // set up data transformation
    // auto test_transforms = torch::data::transforms::Compose({
    //     torch::data::transforms::Resize(224),
    //     torch::data::transforms::CenterCrop(224),
    //     torch::data::transforms::ToTensor(),
    //     torch::data::transforms::Normalize({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225})
    // });

    // load model weights
    std::cerr<<"Start loading model..."<<std::endl;
    ResNet model({3, 4, 6, 3}); // 使用您的层配置创建模型实例
    torch::load(model, model_weight); // 从保存的文件导入参数
    // torch::jit::script::Module model;
    // try {
    // // Deserialize the ScriptModule from a file using torch::jit::load().
    //     model = torch::jit::load("./data/traced_resnet_model.pt");
    // }
    // catch (const c10::Error& e) {
    //     std::cerr << "error loading the model\n";
    // }


    std::cerr<<"Model weights defined."<<std::endl;
    std::cerr<<"Model loaded."<<std::endl;
    #ifdef USE_CUDA
        model.to(torch::kCUDA);
    #endif
    model.eval();

    #ifdef USE_IPEX
        #ifdef USE_BF16
            model = ipex::optimize(model, torch::kBFloat16);
        #else
            model = ipex::optimize(model);
        #endif
    #endif
    std::cerr<<"Model set complete."<<std::endl;

    std::cerr<<"Start running reading loop..."<<std::endl;
    // video reading loop
    int frame_cnt = 0;
    while (cap.isOpened()) {
        auto frame_start_time = std::chrono::high_resolution_clock::now();
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        frame_cnt++;
        std::cerr<<"Getting face locations..."<<std::endl;
        auto bbox = get_face_loc(frame);
        std::cerr<<"Face locations got. Size:"<<bbox.size()<<std::endl<<"Start running face loop..."<<std::endl;
        if (!bbox.empty()) {
            for (const auto &b : bbox) {
                cv::Rect face_rect = b;
                auto face = frame(face_rect);
                auto img = preprocess_image(face);
                img.unsqueeze_(0);
                #ifdef JITTER
                if (jitter > 0) {
                    for (int i = 0; i < jitter; ++i) {
                        auto [bj_left, bj_top, bj_right, bj_bottom] = bbox_jitter(b[0], b[1], b[2], b[3]);
                        cv::Rect jittered_rect(bj_left, bj_top, bj_right - bj_left, bj_bottom - bj_top);
                        auto facej = frame(jittered_rect);
                        auto img_jittered = test_transforms(facej);
                        img_jittered.unsqueeze_(0);
                        img = torch::cat({img, img_jittered});
                    }
                }
                #endif
                // forward pass
                std::cerr<<"Start running forward pass..."<<std::endl;
                #ifdef USE_BF16
                    img = img.to(torch::kBFloat16);
                #endif
                torch::Tensor output;
                #ifdef USE_CUDA
                    output = model.forward(img.to(torch::kCUDA));
                #else
                    output = model.forward({img});
                #endif
                #ifdef JITTER
                if (jitter > 0) {
                    output = torch::mean(output, 0);
                }
                #endif
                float score = torch::sigmoid(output).item<float>();
                std::cerr<<"Forward pass done. Scode got."<<std::endl<<"Start drawing rect..."<<std::endl;

                int coloridx = std::min(static_cast<int>(std::round(score * 10)), 9);
                drawrect(frame, face_rect, colors[coloridx], 5);
                ft2->putText(frame, std::to_string(score), cv::Point(b.height, b.x), 40, cv::Scalar(255, 255, 255, 128), -1, cv::LINE_AA, true);
                std::cerr<<"Rect drawn."<<std::endl;
                if (save_text) {
                    f << frame_cnt << "," << score << "\n";
                }
            }
        }
        std::cerr<<"Face loop done."<<std::endl;
        if (!display_off) {
            cv::imshow("Result", frame);
            if (cv::waitKey(20) == 32) { // wait for space key to exit
                break;
            }
            #ifdef OUTVID
            if (vis) {
                outvid.write(frame);
            }
            #endif
        }

        auto frame_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> frame_time = frame_end_time - frame_start_time;
        double frame_rate = 1.0 / frame_time.count();
        std::cout << "frame time: " << frame_time.count() << "\trealtime frame rate: " << frame_rate << std::endl;
    }
    #ifdef OUTVID
    if (vis) {
        outvid.release();
    }
    #endif
    if (save_text) {
        f.close();
    }
    cap.release();
    std::cout << "DONE!" << std::endl;

    auto total_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_time = total_end_time - total_start_time;
    std::cout << "total time: " << total_time.count() << std::endl;
}

int main(int argc, char **argv) {
    // program.parse_args(argc, argv);
    // run(args.get<std::string>("video"), args.get<std::string>("model_weight"), args.get<int>("jitter"), args.get<bool>("save_vis"), args.get<bool>("display_off"), args.get<bool>("save_text"), args.get<bool>("cuda"), args.get<bool>("ipex"), args.get<bool>("bf16"));
    run("./videoplayback.avi","./data/model_weights.pth",0,false,true,false);
    return 0;
}