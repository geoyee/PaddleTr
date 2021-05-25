import torch
import numpy as np
from deit.cait_models import cait_XXS24_224
from x2paddle.convert import pytorch2paddle


# 构建输入
input_data = np.ones((1, 3, 224, 224)).astype("float32")
# 获取PyTorch Module
torch_module = cait_XXS24_224(pretrained=True)
# 设置为eval模式
torch_module.eval()
# 进行转换
pytorch2paddle(torch_module,
               save_dir="pd_model_trace",
               jit_type="trace",
               input_examples=[torch.tensor(input_data)])