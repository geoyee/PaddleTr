import torch
import numpy as np
from torch2paddle.deit.models import deit_tiny_patch16_224
from x2paddle.convert import pytorch2paddle


# 构建输入
input_data = np.random.rand(1, 3, 224, 224).astype("float32")
# 获取PyTorch Module
torch_module = deit_tiny_patch16_224(pretrained=True)
# 设置为eval模式
torch_module.eval()
# 进行转换
pytorch2paddle(torch_module,
               save_dir="pd_model_trace",
               jit_type="trace",
               input_examples=[torch.tensor(input_data)])