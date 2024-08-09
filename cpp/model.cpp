// #include <torch/torch.h>
// #include <iostream>
// #include <cmath>
#include "model.hpp"


std::shared_ptr<ResNet> model_static(bool pretrained, bool USE_CUDA, const std::string& pretrained_path) {
    auto model = std::make_shared<ResNet>(std::vector<int>{3, 4, 6, 3});
    if (pretrained) {
        std::cout << "loading saved model weights" << std::endl;
        torch::load(model, pretrained_path);
    }
    return model;
}