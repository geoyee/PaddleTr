import paddle
from pd_model_trace.x2paddle_code import VisionTransformer


# 测试
x0 = paddle.randn([1, 3, 224, 224], dtype='float32')
params = paddle.load('pd_model_trace/model.pdparams')
model = VisionTransformer()
model.set_state_dict(params)
model.eval()
out = model(x0)
print(out.shape)
print(out)