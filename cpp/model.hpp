#include <torch/torch.h>
#include <iostream>
#include <cmath>

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

std::shared_ptr<ResNet> model_static(bool pretrained = false, bool USE_CUDA = false, const std::string& pretrained_path = "");