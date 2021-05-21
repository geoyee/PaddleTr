from deit_paddle.cait_models import cait_XXS24_224 as pd_cait
import torch
import pickle
from collections import OrderedDict


# paddle模型参数
pd_model = pd_cait()
pd_pw = pd_model.state_dict()
print(len(pd_pw))
# pth参数
pt_pw = torch.load('XXS24_224.pth')['model']
print(len(pt_pw))
# 查看对应
for kt, kp in zip(pt_pw.keys(), pd_pw.keys()):
    print(kt, kp)
# 错误处理，形状不对，需要转置
ex_tab = [
    'blocks.0.attn.qkv.weight',
    'blocks.0.mlp.fc1.weight',
    'blocks.0.mlp.fc2.weight',
    'blocks.1.attn.qkv.weight',
    'blocks.1.mlp.fc1.weight',
    'blocks.1.mlp.fc2.weight',
    'blocks.10.attn.qkv.weight',
    'blocks.10.mlp.fc1.weight',
    'blocks.10.mlp.fc2.weight',
    'blocks.11.attn.qkv.weight',
    'blocks.11.mlp.fc1.weight',
    'blocks.11.mlp.fc2.weight',
    'blocks.12.attn.qkv.weight',
    'blocks.12.mlp.fc1.weight',
    'blocks.12.mlp.fc2.weight',
    'blocks.13.attn.qkv.weight',
    'blocks.13.mlp.fc1.weight',
    'blocks.13.mlp.fc2.weight',
    'blocks.14.attn.qkv.weight',
    'blocks.14.mlp.fc1.weight',
    'blocks.14.mlp.fc2.weight',
    'blocks.15.attn.qkv.weight',
    'blocks.15.mlp.fc1.weight',
    'blocks.15.mlp.fc2.weight',
    'blocks.16.attn.qkv.weight',
    'blocks.16.mlp.fc1.weight',
    'blocks.16.mlp.fc2.weight',
    'blocks.17.attn.qkv.weight',
    'blocks.17.mlp.fc1.weight',
    'blocks.17.mlp.fc2.weight',
    'blocks.18.attn.qkv.weight',
    'blocks.18.mlp.fc1.weight',
    'blocks.18.mlp.fc2.weight',
    'blocks.19.attn.qkv.weight',
    'blocks.19.mlp.fc1.weight',
    'blocks.19.mlp.fc2.weight',
    'blocks.2.attn.qkv.weight',
    'blocks.2.mlp.fc1.weight',
    'blocks.2.mlp.fc2.weight',
    'blocks.20.attn.qkv.weight',
    'blocks.20.mlp.fc1.weight',
    'blocks.20.mlp.fc2.weight',
    'blocks.21.attn.qkv.weight',
    'blocks.21.mlp.fc1.weight',
    'blocks.21.mlp.fc2.weight',
    'blocks.22.attn.qkv.weight',
    'blocks.22.mlp.fc1.weight',
    'blocks.22.mlp.fc2.weight',
    'blocks.23.attn.qkv.weight',
    'blocks.23.mlp.fc1.weight',
    'blocks.23.mlp.fc2.weight',
    'blocks.3.attn.qkv.weight',
    'blocks.3.mlp.fc1.weight',
    'blocks.3.mlp.fc2.weight',
    'blocks.4.attn.qkv.weight',
    'blocks.4.mlp.fc1.weight',
    'blocks.4.mlp.fc2.weight',
    'blocks.5.attn.qkv.weight',
    'blocks.5.mlp.fc1.weight',
    'blocks.5.mlp.fc2.weight',
    'blocks.6.attn.qkv.weight',
    'blocks.6.mlp.fc1.weight',
    'blocks.6.mlp.fc2.weight',
    'blocks.7.attn.qkv.weight',
    'blocks.7.mlp.fc1.weight',
    'blocks.7.mlp.fc2.weight',
    'blocks.8.attn.qkv.weight',
    'blocks.8.mlp.fc1.weight',
    'blocks.8.mlp.fc2.weight',
    'blocks.9.attn.qkv.weight',
    'blocks.9.mlp.fc1.weight',
    'blocks.9.mlp.fc2.weight',
    'blocks_token_only.0.mlp.fc1.weight',
    'blocks_token_only.0.mlp.fc2.weight',
    'blocks_token_only.1.mlp.fc1.weight',
    'blocks_token_only.1.mlp.fc2.weight',
    'head.weight'
]
# 参数写入
pd_new_dict = OrderedDict()
for kt, kp in zip(pt_pw.keys(), pd_pw.keys()):
    if kp in ex_tab:
        pd_new_dict[kp] = pt_pw[kt].detach().numpy().T
    else:
        pd_new_dict[kp] = pt_pw[kt].detach().numpy()
    # pd_new_dict[kp] = pt_pw[kt].detach().numpy()
with open('XXS24_224.pdparams', 'wb') as f:
    pickle.dump(pd_new_dict, f)
print('finished!')