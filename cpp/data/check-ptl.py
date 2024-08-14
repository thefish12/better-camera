import pickle
import torch

file_path = 'model_weights.pkl'

# 尝试使用 pickle 加载
try:
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    print("文件是一个Python对象。")
    print(f"对象类型: {type(obj)}")

    # 如果对象是字典，打印其键
    if isinstance(obj, dict):
        print("对象是一个字典，包含以下键:")
        for key in obj.keys():
            print(f"  - {key}")

    # 如果对象是列表，打印其长度和前几个元素
    elif isinstance(obj, list):
        print(f"对象是一个列表，包含 {len(obj)} 个元素。")
        print("前几个元素:")
        for i, item in enumerate(obj[:5]):
            print(f"  - 元素 {i}: {item}")

    # 如果对象是 PyTorch 模型，打印其结构
    elif isinstance(obj, torch.nn.Module):
        print("对象是一个 PyTorch 模型。")
        print(obj)

    # 处理其他类型的对象
    else:
        print(f"对象是其他类型: {type(obj)}")
        print(obj)

except Exception as e:
    print("文件不是一个Python对象。尝试使用torch加载。")

    # 尝试使用 torch 加载
    try:
        model = torch.load(file_path)
        print("文件是一个纯权重文件。")
        print(model)
    except Exception as e:
        print("无法加载文件。文件可能已损坏或格式不正确。")
        print(f"错误信息: {e}")