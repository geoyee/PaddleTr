import torch
import numpy as np
from deit.cait_models import cait_XXS24_224


# 构建输入
# input_data = np.random.rand(1, 3, 224, 224).astype("float32")
input_data = np.ones((1, 3, 224, 224)).astype("float32")
# 获取PyTorch Module
torch_module = cait_XXS24_224(pretrained=True)
# 设置为eval模式
torch_module.eval()
# 输出测试
out = torch_module(torch.tensor(input_data))
print(out.size())
print(out.detach().numpy()[:10, :10])