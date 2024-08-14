import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms


from model import ResNet
def model_static(pretrained=False, USE_CUDA=False, **kwargs):
    model = ResNet([3, 4, 6, 3], **kwargs)
    if pretrained:
        print('loading saved model weights')
        model_dict = model.state_dict()
        if USE_CUDA:
            snapshot = torch.load(pretrained)
        else:
            snapshot = torch.load(pretrained, map_location=torch.device('cpu'))
        snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
        model_dict.update(snapshot)
        model.load_state_dict(model_dict)
    return model


# 1. 导入必要的库
# 2. 生成一个随机的 (224, 224, 3) 的图像输入
random_image = np.random.rand(224, 224, 3).astype(np.float32)

# 3. 将图像转换为 PyTorch 张量，并调整维度
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

image_tensor = transform(random_image)
image_tensor = image_tensor.unsqueeze(0)  # 添加 batch 维度


# 加载预训练模型
model_weight = './data/model_weights.pkl'  # 你的预训练模型文件路径
USE_CUDA = False  # 根据需要设置

model = model_static(pretrained=model_weight, USE_CUDA=USE_CUDA)

model.eval()
# 保存模型权重为纯权重文件
# weights_path = 'model_weights.pth'
# torch.save(model.state_dict(), weights_path)
script_path = "./data/model_weights.pt"
traced_model = torch.jit.trace(model, image_tensor)
traced_model.save(script_path)

print(f"模型权重已保存到 {script_path}")