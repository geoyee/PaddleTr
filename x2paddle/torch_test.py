import torch
import numpy as np
from deit.models import deit_tiny_patch16_224


# 构建输入
input_data = np.random.rand(1, 3, 224, 224).astype("float32")
# 获取PyTorch Module
torch_module = deit_tiny_patch16_224(pretrained=True)
# 设置为eval模式
torch_module.eval()
# 输出测试
out = torch_module(torch.tensor(input_data))
print(out.size())