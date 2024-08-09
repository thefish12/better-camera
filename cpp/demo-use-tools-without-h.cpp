// #define USE_IPEX
// #define USE_CUDA
// #define USE_BF16
// #define JITTER

#include <opencv2/opencv.hpp>
#include <opencv2/freetype.hpp>
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

// model.cpp

struct Bottleneck : torch::nn::Module {
    static const int expansion = 4;
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
    torch::nn::Sequential downsample{nullptr};
    int stride;

    Bottleneck(int inplanes, int planes, int stride = 1, torch::nn::Sequential downsample = nullptr)
        : stride(stride), downsample(downsample) {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(inplanes, planes, 1).bias(false)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(planes));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(planes, planes, 3).stride(stride).padding(1).bias(false)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(planes));
        conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(planes, planes * expansion, 1).bias(false)));
        bn3 = register_module("bn3", torch::nn::BatchNorm2d(planes * expansion));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto residual = x;

        auto out = conv1->forward(x);
        out = bn1->forward(out);
        out = torch::relu(out);

        out = conv2->forward(out);
        out = bn2->forward(out);
        out = torch::relu(out);

        out = conv3->forward(out);
        out = bn3->forward(out);

        if (downsample) {
            residual = downsample->forward(x);
        }

        out += residual;
        out = torch::relu(out);

        return out;
    }
};

struct ResNet : torch::nn::Module {
    int inplanes;
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::ReLU relu{nullptr};
    torch::nn::MaxPool2d maxpool{nullptr};
    torch::nn::Sequential layer1{nullptr}, layer2{nullptr}, layer3{nullptr}, layer4{nullptr};
    torch::nn::AvgPool2d avgpool{nullptr};
    torch::nn::Linear fc_theta{nullptr}, fc_phi{nullptr}, fc_ec{nullptr};

    ResNet(std::vector<int> layers) {
        inplanes = 64;
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3).bias(false)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
        relu = register_module("relu", torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
        maxpool = register_module("maxpool", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
        layer1 = register_module("layer1", _make_layer(64, layers[0]));
        layer2 = register_module("layer2", _make_layer(128, layers[1], 2));
        layer3 = register_module("layer3", _make_layer(256, layers[2], 2));
        layer4 = register_module("layer4", _make_layer(512, layers[3], 2));
        avgpool = register_module("avgpool", torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(7).stride(1)));
        fc_theta = register_module("fc_theta", torch::nn::Linear(512 * Bottleneck::expansion, 34));
        fc_phi = register_module("fc_phi", torch::nn::Linear(512 * Bottleneck::expansion, 34));
        fc_ec = register_module("fc_ec", torch::nn::Linear(512 * Bottleneck::expansion, 1));
        init_param();
    }

    void init_param() {
        for (auto& module : modules(/*include_self=*/false)) {
            if (auto* conv = dynamic_cast<torch::nn::Conv2dImpl*>(module.get())) {
                auto n = conv->options.kernel_size()->at(0) * conv->options.kernel_size()->at(1) * conv->options.out_channels();
                conv->weight.data().normal_(0, std::sqrt(2.0 / n));
            } else if (auto* bn = dynamic_cast<torch::nn::BatchNorm2dImpl*>(module.get())) {
                bn->weight.data().fill_(1);
                bn->bias.data().zero_();
            } else if (auto* linear = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
                auto n = linear->weight.size(0) * linear->weight.size(1);
                linear->weight.data().normal_(0, std::sqrt(2.0 / n));
                linear->bias.data().zero_();
            }
        }
    }

    torch::nn::Sequential _make_layer(int planes, int blocks, int stride = 1) {
        torch::nn::Sequential downsample = nullptr;
        if (stride != 1 || inplanes != planes * Bottleneck::expansion) {
            downsample = torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(inplanes, planes * Bottleneck::expansion, 1).stride(stride).bias(false)),
                torch::nn::BatchNorm2d(planes * Bottleneck::expansion)
            );
        }

        torch::nn::Sequential layers;
        layers->push_back(Bottleneck(inplanes, planes, stride, downsample));
        inplanes = planes * Bottleneck::expansion;
        for (int i = 1; i < blocks; ++i) {
            layers->push_back(Bottleneck(inplanes, planes));
        }

        return layers;
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv1->forward(x);
        x = bn1->forward(x);
        x = relu->forward(x);
        x = maxpool->forward(x);

        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);

        x = avgpool->forward(x);
        x = x.view({x.size(0), -1});
        x = fc_ec->forward(x);

        return x;
    }
};

std::shared_ptr<ResNet> model_static(bool pretrained = false, bool USE_CUDA = false, const std::string& pretrained_path = ""){    auto model = std::make_shared<ResNet>(std::vector<int>{3, 4, 6, 3});
    if (pretrained) {
        std::cout << "loading saved model weights" << std::endl;
        torch::load(model, pretrained_path);
    }
    return model;
}
//end of model.cpp

//tools.cpp

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
//end of tools.cpp

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
//     .default_value(std::string("data/model_weights.pkl"));
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


const std::string CNN_FACE_MODEL = "data/mmod_human_face_detector.dat"; // from http://dlib.net/files/mmod_human_face_detector.dat.bz2

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
    cv::Ptr<cv::freetype::FreeType2> ft2;
    ft2 = cv::freetype::createFreeType2();
    ft2->loadFontData("data/arial.ttf", 0);

    // set up video source
    cv::VideoCapture cap;
    std::string video_output_path = video_path;
    if (video_path.empty()) {
        cap.open(0);
        video_output_path = "live.avi";
    } else {
        cap.open(video_path);
    }

    // set up output file
    std::ofstream f;
    if (save_text) {
        std::string outtext_name = std::filesystem::path(video_output_path).stem().string() + "_output.txt";
        f.open(outtext_name);
    }
    cv::VideoWriter outvid;
    if (vis) {
        std::string outvis_name = std::filesystem::path(video_output_path).stem().string() + "_output.avi";
        int imwidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int imheight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        outvid.open(outvis_name, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), cap.get(cv::CAP_PROP_FPS), cv::Size(imwidth, imheight));
    }

    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream or file" << std::endl;
        return;
    }

    // set up data transformation
    // auto test_transforms = torch::data::transforms::Compose({
    //     torch::data::transforms::Resize(224),
    //     torch::data::transforms::CenterCrop(224),
    //     torch::data::transforms::ToTensor(),
    //     torch::data::transforms::Normalize({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225})
    // });

    // load model weights
    #ifdef USE_CUDA
    auto model = model_static(true,true,model_weight);
    #else
    auto model = model_static(true,false,model_weight);
    #endif
    torch::load(model, model_weight);
    #ifdef USE_CUDA
        model->to(torch::kCUDA);
    #endif
    model->eval();

    #ifdef USE_IPEX
        #ifdef USE_BF16
            model = ipex::optimize(model, torch::kBFloat16);
        #else
            model = ipex::optimize(model);
        #endif
    #endif

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
        auto bbox = get_face_loc(frame);
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
                #ifdef USE_BF16
                    img = img.to(torch::kBFloat16);
                #endif
                torch::Tensor output;
                #ifdef USE_CUDA
                    output = model->forward(img.to(torch::kCUDA));
                #else
                    output = model->forward(img);
                #endif
                #ifdef JITTER
                if (jitter > 0) {
                    output = torch::mean(output, 0);
                }
                #endif
                float score = torch::sigmoid(output).item<float>();

                int coloridx = std::min(static_cast<int>(std::round(score * 10)), 9);
                drawrect(frame, face_rect, colors[coloridx], 5);
                ft2->putText(frame, std::to_string(score), cv::Point(b.height, b.x), 40, cv::Scalar(255, 255, 255, 128), -1, cv::LINE_AA, true);
                if (save_text) {
                    f << frame_cnt << "," << score << "\n";
                }
            }
        }

        if (!display_off) {
            cv::imshow("Result", frame);
            if (cv::waitKey(20) == 32) { // wait for space key to exit
                break;
            }
            if (vis) {
                outvid.write(frame);
            }
        }

        auto frame_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> frame_time = frame_end_time - frame_start_time;
        double frame_rate = 1.0 / frame_time.count();
        std::cout << "frame time: " << frame_time.count() << "\trealtime frame rate: " << frame_rate << std::endl;
    }

    if (vis) {
        outvid.release();
    }
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
    run("./videoplayback.avi","",0,false,true,false);
    return 0;
}