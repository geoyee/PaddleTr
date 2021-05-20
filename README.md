# PaddleTr: Deit-Paddle
- Going deeper with Image Transformers
- 百度顶会论文复现营第2期

## Dataset

ILSVRC2012

## Parameters

目前直接由x2paddle转换得到的模型及参数计算结果与官方代码基本相同，现在需要将torch的参数转到自己复现的模型上。

```python
# pytorch
import torch
import numpy as np
from deit.models import deit_tiny_patch16_224
input_data = np.ones((1, 3, 224, 224)).astype("float32")
torch_module = deit_tiny_patch16_224(pretrained=True)
torch_module.eval()
out = torch_module(torch.tensor(input_data))
print(out.size())
print(out.detach().numpy()[:10, :10])

# x2paddle
import paddle
from pd_model_trace.x2paddle_code import VisionTransformer
x0 = paddle.ones([1, 3, 224, 224], dtype='float32')
params = paddle.load('pd_model_trace/model.pdparams')
model = VisionTransformer()
model.set_state_dict(params)
model.eval()
out = model(x0)
print(out.shape)
print(out.numpy()[:10, :10])
```

```
torch.Size([1, 1000])
[[-0.26245788 -0.03369677 -0.17609525  0.00215465  0.20975402  0.35119367
  -0.15210731 -0.17428762 -0.14198136  0.02958778]]
  
[1, 1000]
[[-0.2624576  -0.03369737 -0.17609504  0.00215527  0.20975456  0.35119566
  -0.15210636 -0.17428783 -0.14198156  0.02958757]]
```

## Reference

[https://github.com/facebookresearch/deit](https://github.com/facebookresearch/deit)

[https://github.com/PaddlePaddle/X2Paddle/blob/develop/docs/pytorch_project_convertor/API_docs/README.md](https://github.com/PaddlePaddle/X2Paddle/blob/develop/docs/pytorch_project_convertor/API_docs/README.md)

## AI Studio

[https://aistudio.baidu.com/aistudio/projectdetail/1956552](https://aistudio.baidu.com/aistudio/projectdetail/1956552)