import paddle
from pd_model_trace.x2paddle_code import cait_models


# 测试
x0 = paddle.ones([1, 3, 224, 224], dtype='float32')
params = paddle.load('torch2paddle/pd_model_trace/model.pdparams')
model = cait_models()
model.set_state_dict(params)
model.eval()
out = model(x0)
print(out.shape)
print(out.numpy()[:10, :10])