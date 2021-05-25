import paddle
from deit_paddle.cait_models import cait_XXS24_224

a = paddle.ones([1, 3, 224, 224], dtype='float32')
model = cait_XXS24_224()
params = paddle.load('CaiT_XXS24_224.pdparams')
model.set_state_dict(params)
b = model(a)
print(a.shape)
print(b.shape)
print(b[:10, :10].numpy())