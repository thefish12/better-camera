#include <torch/torch.h>
#include <iostream>
#include <cmath>

class Bottleneck : public torch::nn::Module {
public:
    static const int expansion = 4;

    Bottleneck(int64_t inplanes, int64_t planes, int64_t stride = 1, torch::nn::Sequential downsample = nullptr)
        : conv1(torch::nn::Conv2d(torch::nn::Conv2dOptions(inplanes, planes, 1).bias(false))),
          bn1(planes),
          conv2(torch::nn::Conv2d(torch::nn::Conv2dOptions(planes, planes, 3).stride(stride).padding(1).bias(false))),
          bn2(planes),
          conv3(torch::nn::Conv2d(torch::nn::Conv2dOptions(planes, planes * 4, 1).bias(false))),
          bn3(planes * 4),
          relu(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))),
          downsample(downsample),
          stride(stride) {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        register_module("conv3", conv3);
        register_module("bn3", bn3);
        if (downsample) {
            register_module("downsample", downsample);
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor residual = x;

        torch::Tensor out = conv1->forward(x);
        out = bn1->forward(out);
        out = relu->forward(out);

        out = conv2->forward(out);
        out = bn2->forward(out);
        out = relu->forward(out);

        out = conv3->forward(out);
        out = bn3->forward(out);

        if (downsample) {
            residual = downsample->forward(x);
        }

        out += residual;
        out = relu->forward(out);

        return out;
    }

private:
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm2d bn1;
    torch::nn::Conv2d conv2;
    torch::nn::BatchNorm2d bn2;
    torch::nn::Conv2d conv3;
    torch::nn::BatchNorm2d bn3;
    torch::nn::ReLU relu;
    torch::nn::Sequential downsample;
    int64_t stride;
};

class ResNet : public torch::nn::Module {
public:
    ResNet(const std::vector<int>& layers) {
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
        for (auto& module : modules(false)) {
            if (auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get())) {
                int64_t n = M->weight.size(1) * M->weight.size(2) * M->weight.size(3);
                torch::nn::init::normal_(M->weight, 0, std::sqrt(2. / n));
            } else if (auto M = dynamic_cast<torch::nn::BatchNorm2dImpl*>(module.get())) {
                torch::nn::init::constant_(M->weight, 1);
                torch::nn::init::constant_(M->bias, 0);
            } else if (auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
                int64_t n = M->weight.size(0) * M->weight.size(1);
                torch::nn::init::normal_(M->weight, 0, std::sqrt(2. / n));
                torch::nn::init::constant_(M->bias, 0);
            }
        }
    }

    torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks, int64_t stride = 1) {
        torch::nn::Sequential downsample = nullptr;
        torch::nn::Sequential layers;

        if (stride != 1 || inplanes != planes * Bottleneck::expansion) {
            downsample = torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(inplanes, planes * Bottleneck::expansion, 1).stride(stride).bias(false)),
                torch::nn::BatchNorm2d(planes * Bottleneck::expansion)
            );
        }

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

private:
    int64_t inplanes;
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm2d bn1;
    torch::nn::ReLU relu;
    torch::nn::MaxPool2d maxpool;
    torch::nn::Sequential layer1, layer2, layer3, layer4;
    torch::nn::AvgPool2d avgpool;
    torch::nn::Linear fc_theta, fc_phi, fc_ec;
};

std::shared_ptr<ResNet> model_static(bool pretrained = false, bool USE_CUDA = false, const std::string& pretrained_path = "");