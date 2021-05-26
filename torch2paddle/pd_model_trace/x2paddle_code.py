import paddle
import math
from x2paddle.op_mapper.dygraph.pytorch2paddle import pytorch_custom_layer as x2paddle_nn
class Attention_talking_head(paddle.nn.Layer):
    def __init__(self, ):
        super(Attention_talking_head, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=576)
        self.linear1 = paddle.nn.Linear(in_features=4, out_features=4)
        self.softmax0 = paddle.nn.Softmax()
        self.linear2 = paddle.nn.Linear(in_features=4, out_features=4)
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear3 = paddle.nn.Linear(in_features=192, out_features=192)
        self.dropout1 = paddle.nn.Dropout(p=0.0)
    def forward(self, x0, x4, x6, x8, x10, x15, x17):
        x1 = self.linear0(x0)
        x2 = paddle.reshape(x=x1, shape=[1, 196, 3, 4, 48])
        x3 = paddle.transpose(x=x2, perm=[2, 0, 3, 1, 4])
        x5 = x3[x4]
        x7 = x5 * x6
        x9 = x3[x8]
        x11 = x3[x10]
        x12 = x9.shape
        x13 = len(x12)
        x14 = []
        for i in range(x13):
            x14.append(i)
        if x15 < 0:
            x16 = x15 + x13
        else:
            x16 = x15
        if x17 < 0:
            x18 = x17 + x13
        else:
            x18 = x17
        x14[x16] = x18
        x14[x18] = x16
        x19 = paddle.transpose(x=x9, perm=x14)
        x20 = paddle.matmul(x=x7, y=x19)
        x21 = paddle.transpose(x=x20, perm=[0, 2, 3, 1])
        x22 = self.linear1(x21)
        x23 = paddle.transpose(x=x22, perm=[0, 3, 1, 2])
        x24 = self.softmax0(x23)
        x25 = paddle.transpose(x=x24, perm=[0, 2, 3, 1])
        x26 = self.linear2(x25)
        x27 = paddle.transpose(x=x26, perm=[0, 3, 1, 2])
        x28 = self.dropout0(x27)
        x29 = paddle.matmul(x=x28, y=x11)
        x30 = x29.shape
        x31 = len(x30)
        x32 = []
        for i in range(x31):
            x32.append(i)
        if x8 < 0:
            x33 = x8 + x31
        else:
            x33 = x8
        if x10 < 0:
            x34 = x10 + x31
        else:
            x34 = x10
        x32[x33] = x34
        x32[x34] = x33
        x35 = paddle.transpose(x=x29, perm=x32)
        x36 = paddle.reshape(x=x35, shape=[1, 196, 192])
        x37 = self.linear3(x36)
        x38 = self.dropout1(x37)
        return x38

class Mlp0(paddle.nn.Layer):
    def __init__(self, ):
        super(Mlp0, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=768)
        self.gelu0 = paddle.nn.GELU()
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear1 = paddle.nn.Linear(in_features=768, out_features=192)
        self.dropout1 = paddle.nn.Dropout(p=0.0)
    def forward(self, x0):
        x1 = self.linear0(x0)
        x2 = self.gelu0(x1)
        x3 = self.dropout0(x2)
        x4 = self.linear1(x3)
        x5 = self.dropout1(x4)
        return x5

class Mlp1(paddle.nn.Layer):
    def __init__(self, ):
        super(Mlp1, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=768)
        self.gelu0 = paddle.nn.GELU()
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear1 = paddle.nn.Linear(in_features=768, out_features=192)
        self.dropout1 = paddle.nn.Dropout(p=0.0)
    def forward(self, x0):
        x1 = self.linear0(x0)
        x2 = self.gelu0(x1)
        x3 = self.dropout0(x2)
        x4 = self.linear1(x3)
        x5 = self.dropout1(x4)
        return x5

class Mlp2(paddle.nn.Layer):
    def __init__(self, ):
        super(Mlp2, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=768)
        self.gelu0 = paddle.nn.GELU()
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear1 = paddle.nn.Linear(in_features=768, out_features=192)
        self.dropout1 = paddle.nn.Dropout(p=0.0)
    def forward(self, x0):
        x1 = self.linear0(x0)
        x2 = self.gelu0(x1)
        x3 = self.dropout0(x2)
        x4 = self.linear1(x3)
        x5 = self.dropout1(x4)
        return x5

class Mlp3(paddle.nn.Layer):
    def __init__(self, ):
        super(Mlp3, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=768)
        self.gelu0 = paddle.nn.GELU()
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear1 = paddle.nn.Linear(in_features=768, out_features=192)
        self.dropout1 = paddle.nn.Dropout(p=0.0)
    def forward(self, x0):
        x1 = self.linear0(x0)
        x2 = self.gelu0(x1)
        x3 = self.dropout0(x2)
        x4 = self.linear1(x3)
        x5 = self.dropout1(x4)
        return x5

class Mlp4(paddle.nn.Layer):
    def __init__(self, ):
        super(Mlp4, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=768)
        self.gelu0 = paddle.nn.GELU()
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear1 = paddle.nn.Linear(in_features=768, out_features=192)
        self.dropout1 = paddle.nn.Dropout(p=0.0)
    def forward(self, x0):
        x1 = self.linear0(x0)
        x2 = self.gelu0(x1)
        x3 = self.dropout0(x2)
        x4 = self.linear1(x3)
        x5 = self.dropout1(x4)
        return x5

class Mlp5(paddle.nn.Layer):
    def __init__(self, ):
        super(Mlp5, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=768)
        self.gelu0 = paddle.nn.GELU()
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear1 = paddle.nn.Linear(in_features=768, out_features=192)
        self.dropout1 = paddle.nn.Dropout(p=0.0)
    def forward(self, x0):
        x1 = self.linear0(x0)
        x2 = self.gelu0(x1)
        x3 = self.dropout0(x2)
        x4 = self.linear1(x3)
        x5 = self.dropout1(x4)
        return x5

class Mlp6(paddle.nn.Layer):
    def __init__(self, ):
        super(Mlp6, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=768)
        self.gelu0 = paddle.nn.GELU()
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear1 = paddle.nn.Linear(in_features=768, out_features=192)
        self.dropout1 = paddle.nn.Dropout(p=0.0)
    def forward(self, x0):
        x1 = self.linear0(x0)
        x2 = self.gelu0(x1)
        x3 = self.dropout0(x2)
        x4 = self.linear1(x3)
        x5 = self.dropout1(x4)
        return x5

class Mlp7(paddle.nn.Layer):
    def __init__(self, ):
        super(Mlp7, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=768)
        self.gelu0 = paddle.nn.GELU()
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear1 = paddle.nn.Linear(in_features=768, out_features=192)
        self.dropout1 = paddle.nn.Dropout(p=0.0)
    def forward(self, x0):
        x1 = self.linear0(x0)
        x2 = self.gelu0(x1)
        x3 = self.dropout0(x2)
        x4 = self.linear1(x3)
        x5 = self.dropout1(x4)
        return x5

class Mlp8(paddle.nn.Layer):
    def __init__(self, ):
        super(Mlp8, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=768)
        self.gelu0 = paddle.nn.GELU()
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear1 = paddle.nn.Linear(in_features=768, out_features=192)
        self.dropout1 = paddle.nn.Dropout(p=0.0)
    def forward(self, x0):
        x1 = self.linear0(x0)
        x2 = self.gelu0(x1)
        x3 = self.dropout0(x2)
        x4 = self.linear1(x3)
        x5 = self.dropout1(x4)
        return x5

class Mlp9(paddle.nn.Layer):
    def __init__(self, ):
        super(Mlp9, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=768)
        self.gelu0 = paddle.nn.GELU()
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear1 = paddle.nn.Linear(in_features=768, out_features=192)
        self.dropout1 = paddle.nn.Dropout(p=0.0)
    def forward(self, x0):
        x1 = self.linear0(x0)
        x2 = self.gelu0(x1)
        x3 = self.dropout0(x2)
        x4 = self.linear1(x3)
        x5 = self.dropout1(x4)
        return x5

class Mlp10(paddle.nn.Layer):
    def __init__(self, ):
        super(Mlp10, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=768)
        self.gelu0 = paddle.nn.GELU()
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear1 = paddle.nn.Linear(in_features=768, out_features=192)
        self.dropout1 = paddle.nn.Dropout(p=0.0)
    def forward(self, x0):
        x1 = self.linear0(x0)
        x2 = self.gelu0(x1)
        x3 = self.dropout0(x2)
        x4 = self.linear1(x3)
        x5 = self.dropout1(x4)
        return x5

class Mlp11(paddle.nn.Layer):
    def __init__(self, ):
        super(Mlp11, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=768)
        self.gelu0 = paddle.nn.GELU()
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear1 = paddle.nn.Linear(in_features=768, out_features=192)
        self.dropout1 = paddle.nn.Dropout(p=0.0)
    def forward(self, x0):
        x1 = self.linear0(x0)
        x2 = self.gelu0(x1)
        x3 = self.dropout0(x2)
        x4 = self.linear1(x3)
        x5 = self.dropout1(x4)
        return x5

class Mlp12(paddle.nn.Layer):
    def __init__(self, ):
        super(Mlp12, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=768)
        self.gelu0 = paddle.nn.GELU()
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear1 = paddle.nn.Linear(in_features=768, out_features=192)
        self.dropout1 = paddle.nn.Dropout(p=0.0)
    def forward(self, x0):
        x1 = self.linear0(x0)
        x2 = self.gelu0(x1)
        x3 = self.dropout0(x2)
        x4 = self.linear1(x3)
        x5 = self.dropout1(x4)
        return x5

class Mlp13(paddle.nn.Layer):
    def __init__(self, ):
        super(Mlp13, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=768)
        self.gelu0 = paddle.nn.GELU()
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear1 = paddle.nn.Linear(in_features=768, out_features=192)
        self.dropout1 = paddle.nn.Dropout(p=0.0)
    def forward(self, x0):
        x1 = self.linear0(x0)
        x2 = self.gelu0(x1)
        x3 = self.dropout0(x2)
        x4 = self.linear1(x3)
        x5 = self.dropout1(x4)
        return x5

class Mlp14(paddle.nn.Layer):
    def __init__(self, ):
        super(Mlp14, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=768)
        self.gelu0 = paddle.nn.GELU()
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear1 = paddle.nn.Linear(in_features=768, out_features=192)
        self.dropout1 = paddle.nn.Dropout(p=0.0)
    def forward(self, x0):
        x1 = self.linear0(x0)
        x2 = self.gelu0(x1)
        x3 = self.dropout0(x2)
        x4 = self.linear1(x3)
        x5 = self.dropout1(x4)
        return x5

class Mlp15(paddle.nn.Layer):
    def __init__(self, ):
        super(Mlp15, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=768)
        self.gelu0 = paddle.nn.GELU()
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear1 = paddle.nn.Linear(in_features=768, out_features=192)
        self.dropout1 = paddle.nn.Dropout(p=0.0)
    def forward(self, x0):
        x1 = self.linear0(x0)
        x2 = self.gelu0(x1)
        x3 = self.dropout0(x2)
        x4 = self.linear1(x3)
        x5 = self.dropout1(x4)
        return x5

class Mlp16(paddle.nn.Layer):
    def __init__(self, ):
        super(Mlp16, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=768)
        self.gelu0 = paddle.nn.GELU()
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear1 = paddle.nn.Linear(in_features=768, out_features=192)
        self.dropout1 = paddle.nn.Dropout(p=0.0)
    def forward(self, x0):
        x1 = self.linear0(x0)
        x2 = self.gelu0(x1)
        x3 = self.dropout0(x2)
        x4 = self.linear1(x3)
        x5 = self.dropout1(x4)
        return x5

class Mlp17(paddle.nn.Layer):
    def __init__(self, ):
        super(Mlp17, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=768)
        self.gelu0 = paddle.nn.GELU()
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear1 = paddle.nn.Linear(in_features=768, out_features=192)
        self.dropout1 = paddle.nn.Dropout(p=0.0)
    def forward(self, x0):
        x1 = self.linear0(x0)
        x2 = self.gelu0(x1)
        x3 = self.dropout0(x2)
        x4 = self.linear1(x3)
        x5 = self.dropout1(x4)
        return x5

class Mlp18(paddle.nn.Layer):
    def __init__(self, ):
        super(Mlp18, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=768)
        self.gelu0 = paddle.nn.GELU()
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear1 = paddle.nn.Linear(in_features=768, out_features=192)
        self.dropout1 = paddle.nn.Dropout(p=0.0)
    def forward(self, x0):
        x1 = self.linear0(x0)
        x2 = self.gelu0(x1)
        x3 = self.dropout0(x2)
        x4 = self.linear1(x3)
        x5 = self.dropout1(x4)
        return x5

class Mlp19(paddle.nn.Layer):
    def __init__(self, ):
        super(Mlp19, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=768)
        self.gelu0 = paddle.nn.GELU()
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear1 = paddle.nn.Linear(in_features=768, out_features=192)
        self.dropout1 = paddle.nn.Dropout(p=0.0)
    def forward(self, x0):
        x1 = self.linear0(x0)
        x2 = self.gelu0(x1)
        x3 = self.dropout0(x2)
        x4 = self.linear1(x3)
        x5 = self.dropout1(x4)
        return x5

class Mlp20(paddle.nn.Layer):
    def __init__(self, ):
        super(Mlp20, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=768)
        self.gelu0 = paddle.nn.GELU()
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear1 = paddle.nn.Linear(in_features=768, out_features=192)
        self.dropout1 = paddle.nn.Dropout(p=0.0)
    def forward(self, x0):
        x1 = self.linear0(x0)
        x2 = self.gelu0(x1)
        x3 = self.dropout0(x2)
        x4 = self.linear1(x3)
        x5 = self.dropout1(x4)
        return x5

class Mlp21(paddle.nn.Layer):
    def __init__(self, ):
        super(Mlp21, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=768)
        self.gelu0 = paddle.nn.GELU()
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear1 = paddle.nn.Linear(in_features=768, out_features=192)
        self.dropout1 = paddle.nn.Dropout(p=0.0)
    def forward(self, x0):
        x1 = self.linear0(x0)
        x2 = self.gelu0(x1)
        x3 = self.dropout0(x2)
        x4 = self.linear1(x3)
        x5 = self.dropout1(x4)
        return x5

class Mlp22(paddle.nn.Layer):
    def __init__(self, ):
        super(Mlp22, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=768)
        self.gelu0 = paddle.nn.GELU()
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear1 = paddle.nn.Linear(in_features=768, out_features=192)
        self.dropout1 = paddle.nn.Dropout(p=0.0)
    def forward(self, x0):
        x1 = self.linear0(x0)
        x2 = self.gelu0(x1)
        x3 = self.dropout0(x2)
        x4 = self.linear1(x3)
        x5 = self.dropout1(x4)
        return x5

class Mlp23(paddle.nn.Layer):
    def __init__(self, ):
        super(Mlp23, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=768)
        self.gelu0 = paddle.nn.GELU()
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear1 = paddle.nn.Linear(in_features=768, out_features=192)
        self.dropout1 = paddle.nn.Dropout(p=0.0)
    def forward(self, x0):
        x1 = self.linear0(x0)
        x2 = self.gelu0(x1)
        x3 = self.dropout0(x2)
        x4 = self.linear1(x3)
        x5 = self.dropout1(x4)
        return x5

class Mlp24(paddle.nn.Layer):
    def __init__(self, ):
        super(Mlp24, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=768)
        self.gelu0 = paddle.nn.GELU()
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear1 = paddle.nn.Linear(in_features=768, out_features=192)
        self.dropout1 = paddle.nn.Dropout(p=0.0)
    def forward(self, x0):
        x1 = self.linear0(x0)
        x2 = self.gelu0(x1)
        x3 = self.dropout0(x2)
        x4 = self.linear1(x3)
        x5 = self.dropout1(x4)
        return x5

class Mlp25(paddle.nn.Layer):
    def __init__(self, ):
        super(Mlp25, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=768)
        self.gelu0 = paddle.nn.GELU()
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear1 = paddle.nn.Linear(in_features=768, out_features=192)
        self.dropout1 = paddle.nn.Dropout(p=0.0)
    def forward(self, x0):
        x1 = self.linear0(x0)
        x2 = self.gelu0(x1)
        x3 = self.dropout0(x2)
        x4 = self.linear1(x3)
        x5 = self.dropout1(x4)
        return x5

class Class_Attention(paddle.nn.Layer):
    def __init__(self, ):
        super(Class_Attention, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=192)
        self.linear1 = paddle.nn.Linear(in_features=192, out_features=192)
        self.linear2 = paddle.nn.Linear(in_features=192, out_features=192)
        self.softmax0 = paddle.nn.Softmax()
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear3 = paddle.nn.Linear(in_features=192, out_features=192)
        self.dropout1 = paddle.nn.Dropout(p=0.0)
    def forward(self, x0, x5, x9, x17, x19, x29, x31):
        x1 = self.linear0(x0)
        x2 = paddle.unsqueeze(x=x1, axis=1)
        x3 = paddle.reshape(x=x2, shape=[1, 1, 4, 48])
        x4 = paddle.transpose(x=x3, perm=[0, 2, 1, 3])
        x6 = self.linear1(x5)
        x7 = paddle.reshape(x=x6, shape=[1, 197, 4, 48])
        x8 = paddle.transpose(x=x7, perm=[0, 2, 1, 3])
        x10 = x4 * x9
        x11 = self.linear2(x5)
        x12 = paddle.reshape(x=x11, shape=[1, 197, 4, 48])
        x13 = paddle.transpose(x=x12, perm=[0, 2, 1, 3])
        x14 = x8.shape
        x15 = len(x14)
        x16 = []
        for i in range(x15):
            x16.append(i)
        if x17 < 0:
            x18 = x17 + x15
        else:
            x18 = x17
        if x19 < 0:
            x20 = x19 + x15
        else:
            x20 = x19
        x16[x18] = x20
        x16[x20] = x18
        x21 = paddle.transpose(x=x8, perm=x16)
        x22 = paddle.matmul(x=x10, y=x21)
        x23 = self.softmax0(x22)
        x24 = self.dropout0(x23)
        x25 = paddle.matmul(x=x24, y=x13)
        x26 = x25.shape
        x27 = len(x26)
        x28 = []
        for i in range(x27):
            x28.append(i)
        if x29 < 0:
            x30 = x29 + x27
        else:
            x30 = x29
        if x31 < 0:
            x32 = x31 + x27
        else:
            x32 = x31
        x28[x30] = x32
        x28[x32] = x30
        x33 = paddle.transpose(x=x25, perm=x28)
        x34 = paddle.reshape(x=x33, shape=[1, 1, 192])
        x35 = self.linear3(x34)
        x36 = self.dropout1(x35)
        return x36

class LayerScale_Block0(paddle.nn.Layer):
    def __init__(self, ):
        super(LayerScale_Block0, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention_talking_head0 = Attention_talking_head()
        self.x9 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp130 = Mlp13()
        self.x14 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
    def forward(self, x6):
        x0 = 0
        x1 = 1
        x2 = 2
        x3 = 0.14433756729740643
        x4 = -2
        x5 = -1
        x7 = self.layernorm0(x6)
        x8 = self.attention_talking_head0(x7, x0, x3, x1, x2, x4, x5)
        x9 = self.x9
        x10 = x9 * x8
        x11 = x6 + x10
        x12 = self.layernorm1(x11)
        x13 = self.mlp130(x12)
        x14 = self.x14
        x15 = x14 * x13
        x16 = x11 + x15
        return x16

class LayerScale_Block1(paddle.nn.Layer):
    def __init__(self, ):
        super(LayerScale_Block1, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention_talking_head0 = Attention_talking_head()
        self.x9 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp220 = Mlp22()
        self.x14 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
    def forward(self, x6):
        x0 = 0
        x1 = 1
        x2 = 2
        x3 = 0.14433756729740643
        x4 = -2
        x5 = -1
        x7 = self.layernorm0(x6)
        x8 = self.attention_talking_head0(x7, x0, x3, x1, x2, x4, x5)
        x9 = self.x9
        x10 = x9 * x8
        x11 = x6 + x10
        x12 = self.layernorm1(x11)
        x13 = self.mlp220(x12)
        x14 = self.x14
        x15 = x14 * x13
        x16 = x11 + x15
        return x16

class LayerScale_Block2(paddle.nn.Layer):
    def __init__(self, ):
        super(LayerScale_Block2, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention_talking_head0 = Attention_talking_head()
        self.x9 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp200 = Mlp20()
        self.x14 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
    def forward(self, x6):
        x0 = 0
        x1 = 1
        x2 = 2
        x3 = 0.14433756729740643
        x4 = -2
        x5 = -1
        x7 = self.layernorm0(x6)
        x8 = self.attention_talking_head0(x7, x0, x3, x1, x2, x4, x5)
        x9 = self.x9
        x10 = x9 * x8
        x11 = x6 + x10
        x12 = self.layernorm1(x11)
        x13 = self.mlp200(x12)
        x14 = self.x14
        x15 = x14 * x13
        x16 = x11 + x15
        return x16

class LayerScale_Block3(paddle.nn.Layer):
    def __init__(self, ):
        super(LayerScale_Block3, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention_talking_head0 = Attention_talking_head()
        self.x9 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp190 = Mlp19()
        self.x14 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
    def forward(self, x6):
        x0 = 0
        x1 = 1
        x2 = 2
        x3 = 0.14433756729740643
        x4 = -2
        x5 = -1
        x7 = self.layernorm0(x6)
        x8 = self.attention_talking_head0(x7, x0, x3, x1, x2, x4, x5)
        x9 = self.x9
        x10 = x9 * x8
        x11 = x6 + x10
        x12 = self.layernorm1(x11)
        x13 = self.mlp190(x12)
        x14 = self.x14
        x15 = x14 * x13
        x16 = x11 + x15
        return x16

class LayerScale_Block4(paddle.nn.Layer):
    def __init__(self, ):
        super(LayerScale_Block4, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention_talking_head0 = Attention_talking_head()
        self.x9 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp40 = Mlp4()
        self.x14 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
    def forward(self, x6):
        x0 = 0
        x1 = 1
        x2 = 2
        x3 = 0.14433756729740643
        x4 = -2
        x5 = -1
        x7 = self.layernorm0(x6)
        x8 = self.attention_talking_head0(x7, x0, x3, x1, x2, x4, x5)
        x9 = self.x9
        x10 = x9 * x8
        x11 = x6 + x10
        x12 = self.layernorm1(x11)
        x13 = self.mlp40(x12)
        x14 = self.x14
        x15 = x14 * x13
        x16 = x11 + x15
        return x16

class LayerScale_Block5(paddle.nn.Layer):
    def __init__(self, ):
        super(LayerScale_Block5, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention_talking_head0 = Attention_talking_head()
        self.x9 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp210 = Mlp21()
        self.x14 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
    def forward(self, x6):
        x0 = 0
        x1 = 1
        x2 = 2
        x3 = 0.14433756729740643
        x4 = -2
        x5 = -1
        x7 = self.layernorm0(x6)
        x8 = self.attention_talking_head0(x7, x0, x3, x1, x2, x4, x5)
        x9 = self.x9
        x10 = x9 * x8
        x11 = x6 + x10
        x12 = self.layernorm1(x11)
        x13 = self.mlp210(x12)
        x14 = self.x14
        x15 = x14 * x13
        x16 = x11 + x15
        return x16

class LayerScale_Block6(paddle.nn.Layer):
    def __init__(self, ):
        super(LayerScale_Block6, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention_talking_head0 = Attention_talking_head()
        self.x9 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp90 = Mlp9()
        self.x14 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
    def forward(self, x6):
        x0 = 0
        x1 = 1
        x2 = 2
        x3 = 0.14433756729740643
        x4 = -2
        x5 = -1
        x7 = self.layernorm0(x6)
        x8 = self.attention_talking_head0(x7, x0, x3, x1, x2, x4, x5)
        x9 = self.x9
        x10 = x9 * x8
        x11 = x6 + x10
        x12 = self.layernorm1(x11)
        x13 = self.mlp90(x12)
        x14 = self.x14
        x15 = x14 * x13
        x16 = x11 + x15
        return x16

class LayerScale_Block7(paddle.nn.Layer):
    def __init__(self, ):
        super(LayerScale_Block7, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention_talking_head0 = Attention_talking_head()
        self.x9 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp60 = Mlp6()
        self.x14 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
    def forward(self, x6):
        x0 = 0
        x1 = 1
        x2 = 2
        x3 = 0.14433756729740643
        x4 = -2
        x5 = -1
        x7 = self.layernorm0(x6)
        x8 = self.attention_talking_head0(x7, x0, x3, x1, x2, x4, x5)
        x9 = self.x9
        x10 = x9 * x8
        x11 = x6 + x10
        x12 = self.layernorm1(x11)
        x13 = self.mlp60(x12)
        x14 = self.x14
        x15 = x14 * x13
        x16 = x11 + x15
        return x16

class LayerScale_Block8(paddle.nn.Layer):
    def __init__(self, ):
        super(LayerScale_Block8, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention_talking_head0 = Attention_talking_head()
        self.x9 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp240 = Mlp24()
        self.x14 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
    def forward(self, x6):
        x0 = 0
        x1 = 1
        x2 = 2
        x3 = 0.14433756729740643
        x4 = -2
        x5 = -1
        x7 = self.layernorm0(x6)
        x8 = self.attention_talking_head0(x7, x0, x3, x1, x2, x4, x5)
        x9 = self.x9
        x10 = x9 * x8
        x11 = x6 + x10
        x12 = self.layernorm1(x11)
        x13 = self.mlp240(x12)
        x14 = self.x14
        x15 = x14 * x13
        x16 = x11 + x15
        return x16

class LayerScale_Block9(paddle.nn.Layer):
    def __init__(self, ):
        super(LayerScale_Block9, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention_talking_head0 = Attention_talking_head()
        self.x9 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp250 = Mlp25()
        self.x14 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
    def forward(self, x6):
        x0 = 0
        x1 = 1
        x2 = 2
        x3 = 0.14433756729740643
        x4 = -2
        x5 = -1
        x7 = self.layernorm0(x6)
        x8 = self.attention_talking_head0(x7, x0, x3, x1, x2, x4, x5)
        x9 = self.x9
        x10 = x9 * x8
        x11 = x6 + x10
        x12 = self.layernorm1(x11)
        x13 = self.mlp250(x12)
        x14 = self.x14
        x15 = x14 * x13
        x16 = x11 + x15
        return x16

class LayerScale_Block10(paddle.nn.Layer):
    def __init__(self, ):
        super(LayerScale_Block10, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention_talking_head0 = Attention_talking_head()
        self.x9 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp160 = Mlp16()
        self.x14 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
    def forward(self, x6):
        x0 = 0
        x1 = 1
        x2 = 2
        x3 = 0.14433756729740643
        x4 = -2
        x5 = -1
        x7 = self.layernorm0(x6)
        x8 = self.attention_talking_head0(x7, x0, x3, x1, x2, x4, x5)
        x9 = self.x9
        x10 = x9 * x8
        x11 = x6 + x10
        x12 = self.layernorm1(x11)
        x13 = self.mlp160(x12)
        x14 = self.x14
        x15 = x14 * x13
        x16 = x11 + x15
        return x16

class LayerScale_Block11(paddle.nn.Layer):
    def __init__(self, ):
        super(LayerScale_Block11, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention_talking_head0 = Attention_talking_head()
        self.x9 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp110 = Mlp11()
        self.x14 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
    def forward(self, x6):
        x0 = 0
        x1 = 1
        x2 = 2
        x3 = 0.14433756729740643
        x4 = -2
        x5 = -1
        x7 = self.layernorm0(x6)
        x8 = self.attention_talking_head0(x7, x0, x3, x1, x2, x4, x5)
        x9 = self.x9
        x10 = x9 * x8
        x11 = x6 + x10
        x12 = self.layernorm1(x11)
        x13 = self.mlp110(x12)
        x14 = self.x14
        x15 = x14 * x13
        x16 = x11 + x15
        return x16

class LayerScale_Block12(paddle.nn.Layer):
    def __init__(self, ):
        super(LayerScale_Block12, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention_talking_head0 = Attention_talking_head()
        self.x9 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp100 = Mlp10()
        self.x14 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
    def forward(self, x6):
        x0 = 0
        x1 = 1
        x2 = 2
        x3 = 0.14433756729740643
        x4 = -2
        x5 = -1
        x7 = self.layernorm0(x6)
        x8 = self.attention_talking_head0(x7, x0, x3, x1, x2, x4, x5)
        x9 = self.x9
        x10 = x9 * x8
        x11 = x6 + x10
        x12 = self.layernorm1(x11)
        x13 = self.mlp100(x12)
        x14 = self.x14
        x15 = x14 * x13
        x16 = x11 + x15
        return x16

class LayerScale_Block13(paddle.nn.Layer):
    def __init__(self, ):
        super(LayerScale_Block13, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention_talking_head0 = Attention_talking_head()
        self.x9 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp30 = Mlp3()
        self.x14 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
    def forward(self, x6):
        x0 = 0
        x1 = 1
        x2 = 2
        x3 = 0.14433756729740643
        x4 = -2
        x5 = -1
        x7 = self.layernorm0(x6)
        x8 = self.attention_talking_head0(x7, x0, x3, x1, x2, x4, x5)
        x9 = self.x9
        x10 = x9 * x8
        x11 = x6 + x10
        x12 = self.layernorm1(x11)
        x13 = self.mlp30(x12)
        x14 = self.x14
        x15 = x14 * x13
        x16 = x11 + x15
        return x16

class LayerScale_Block14(paddle.nn.Layer):
    def __init__(self, ):
        super(LayerScale_Block14, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention_talking_head0 = Attention_talking_head()
        self.x9 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp180 = Mlp18()
        self.x14 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
    def forward(self, x6):
        x0 = 0
        x1 = 1
        x2 = 2
        x3 = 0.14433756729740643
        x4 = -2
        x5 = -1
        x7 = self.layernorm0(x6)
        x8 = self.attention_talking_head0(x7, x0, x3, x1, x2, x4, x5)
        x9 = self.x9
        x10 = x9 * x8
        x11 = x6 + x10
        x12 = self.layernorm1(x11)
        x13 = self.mlp180(x12)
        x14 = self.x14
        x15 = x14 * x13
        x16 = x11 + x15
        return x16

class LayerScale_Block15(paddle.nn.Layer):
    def __init__(self, ):
        super(LayerScale_Block15, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention_talking_head0 = Attention_talking_head()
        self.x9 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp70 = Mlp7()
        self.x14 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
    def forward(self, x6):
        x0 = 0
        x1 = 1
        x2 = 2
        x3 = 0.14433756729740643
        x4 = -2
        x5 = -1
        x7 = self.layernorm0(x6)
        x8 = self.attention_talking_head0(x7, x0, x3, x1, x2, x4, x5)
        x9 = self.x9
        x10 = x9 * x8
        x11 = x6 + x10
        x12 = self.layernorm1(x11)
        x13 = self.mlp70(x12)
        x14 = self.x14
        x15 = x14 * x13
        x16 = x11 + x15
        return x16

class LayerScale_Block16(paddle.nn.Layer):
    def __init__(self, ):
        super(LayerScale_Block16, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention_talking_head0 = Attention_talking_head()
        self.x9 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp10 = Mlp1()
        self.x14 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
    def forward(self, x6):
        x0 = 0
        x1 = 1
        x2 = 2
        x3 = 0.14433756729740643
        x4 = -2
        x5 = -1
        x7 = self.layernorm0(x6)
        x8 = self.attention_talking_head0(x7, x0, x3, x1, x2, x4, x5)
        x9 = self.x9
        x10 = x9 * x8
        x11 = x6 + x10
        x12 = self.layernorm1(x11)
        x13 = self.mlp10(x12)
        x14 = self.x14
        x15 = x14 * x13
        x16 = x11 + x15
        return x16

class LayerScale_Block17(paddle.nn.Layer):
    def __init__(self, ):
        super(LayerScale_Block17, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention_talking_head0 = Attention_talking_head()
        self.x9 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp20 = Mlp2()
        self.x14 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
    def forward(self, x6):
        x0 = 0
        x1 = 1
        x2 = 2
        x3 = 0.14433756729740643
        x4 = -2
        x5 = -1
        x7 = self.layernorm0(x6)
        x8 = self.attention_talking_head0(x7, x0, x3, x1, x2, x4, x5)
        x9 = self.x9
        x10 = x9 * x8
        x11 = x6 + x10
        x12 = self.layernorm1(x11)
        x13 = self.mlp20(x12)
        x14 = self.x14
        x15 = x14 * x13
        x16 = x11 + x15
        return x16

class LayerScale_Block18(paddle.nn.Layer):
    def __init__(self, ):
        super(LayerScale_Block18, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention_talking_head0 = Attention_talking_head()
        self.x9 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp120 = Mlp12()
        self.x14 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
    def forward(self, x6):
        x0 = 0
        x1 = 1
        x2 = 2
        x3 = 0.14433756729740643
        x4 = -2
        x5 = -1
        x7 = self.layernorm0(x6)
        x8 = self.attention_talking_head0(x7, x0, x3, x1, x2, x4, x5)
        x9 = self.x9
        x10 = x9 * x8
        x11 = x6 + x10
        x12 = self.layernorm1(x11)
        x13 = self.mlp120(x12)
        x14 = self.x14
        x15 = x14 * x13
        x16 = x11 + x15
        return x16

class LayerScale_Block19(paddle.nn.Layer):
    def __init__(self, ):
        super(LayerScale_Block19, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention_talking_head0 = Attention_talking_head()
        self.x9 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp50 = Mlp5()
        self.x14 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
    def forward(self, x6):
        x0 = 0
        x1 = 1
        x2 = 2
        x3 = 0.14433756729740643
        x4 = -2
        x5 = -1
        x7 = self.layernorm0(x6)
        x8 = self.attention_talking_head0(x7, x0, x3, x1, x2, x4, x5)
        x9 = self.x9
        x10 = x9 * x8
        x11 = x6 + x10
        x12 = self.layernorm1(x11)
        x13 = self.mlp50(x12)
        x14 = self.x14
        x15 = x14 * x13
        x16 = x11 + x15
        return x16

class LayerScale_Block20(paddle.nn.Layer):
    def __init__(self, ):
        super(LayerScale_Block20, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention_talking_head0 = Attention_talking_head()
        self.x9 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp150 = Mlp15()
        self.x14 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
    def forward(self, x6):
        x0 = 0
        x1 = 1
        x2 = 2
        x3 = 0.14433756729740643
        x4 = -2
        x5 = -1
        x7 = self.layernorm0(x6)
        x8 = self.attention_talking_head0(x7, x0, x3, x1, x2, x4, x5)
        x9 = self.x9
        x10 = x9 * x8
        x11 = x6 + x10
        x12 = self.layernorm1(x11)
        x13 = self.mlp150(x12)
        x14 = self.x14
        x15 = x14 * x13
        x16 = x11 + x15
        return x16

class LayerScale_Block21(paddle.nn.Layer):
    def __init__(self, ):
        super(LayerScale_Block21, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention_talking_head0 = Attention_talking_head()
        self.x9 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp170 = Mlp17()
        self.x14 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
    def forward(self, x6):
        x0 = 0
        x1 = 1
        x2 = 2
        x3 = 0.14433756729740643
        x4 = -2
        x5 = -1
        x7 = self.layernorm0(x6)
        x8 = self.attention_talking_head0(x7, x0, x3, x1, x2, x4, x5)
        x9 = self.x9
        x10 = x9 * x8
        x11 = x6 + x10
        x12 = self.layernorm1(x11)
        x13 = self.mlp170(x12)
        x14 = self.x14
        x15 = x14 * x13
        x16 = x11 + x15
        return x16

class LayerScale_Block22(paddle.nn.Layer):
    def __init__(self, ):
        super(LayerScale_Block22, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention_talking_head0 = Attention_talking_head()
        self.x9 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp00 = Mlp0()
        self.x14 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
    def forward(self, x6):
        x0 = 0
        x1 = 1
        x2 = 2
        x3 = 0.14433756729740643
        x4 = -2
        x5 = -1
        x7 = self.layernorm0(x6)
        x8 = self.attention_talking_head0(x7, x0, x3, x1, x2, x4, x5)
        x9 = self.x9
        x10 = x9 * x8
        x11 = x6 + x10
        x12 = self.layernorm1(x11)
        x13 = self.mlp00(x12)
        x14 = self.x14
        x15 = x14 * x13
        x16 = x11 + x15
        return x16

class LayerScale_Block23(paddle.nn.Layer):
    def __init__(self, ):
        super(LayerScale_Block23, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention_talking_head0 = Attention_talking_head()
        self.x9 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp80 = Mlp8()
        self.x14 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
    def forward(self, x6):
        x0 = 0
        x1 = 1
        x2 = 2
        x3 = 0.14433756729740643
        x4 = -2
        x5 = -1
        x7 = self.layernorm0(x6)
        x8 = self.attention_talking_head0(x7, x0, x3, x1, x2, x4, x5)
        x9 = self.x9
        x10 = x9 * x8
        x11 = x6 + x10
        x12 = self.layernorm1(x11)
        x13 = self.mlp80(x12)
        x14 = self.x14
        x15 = x14 * x13
        x16 = x11 + x15
        return x16

class LayerScale_Block_CA0(paddle.nn.Layer):
    def __init__(self, ):
        super(LayerScale_Block_CA0, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.class_attention0 = Class_Attention()
        self.x17 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp230 = Mlp23()
        self.x22 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
    def forward(self, x6, x7):
        x0 = 0
        x1 = 2
        x2 = 0.14433756729740643
        x3 = -2
        x4 = -1
        x5 = 1
        x8 = [x6, x7]
        x9 = paddle.concat(x=x8, axis=1)
        x10 = self.layernorm0(x9)
        x11 = [0]
        x12 = [2147483647]
        x13 = [1]
        x14 = paddle.strided_slice(x=x10, axes=x11, starts=x11, ends=x12, strides=x13)
        x15 = x14[:, x0]
        x16 = self.class_attention0(x15, x10, x2, x3, x4, x5, x1)
        x17 = self.x17
        x18 = x17 * x16
        x19 = x6 + x18
        x20 = self.layernorm1(x19)
        x21 = self.mlp230(x20)
        x22 = self.x22
        x23 = x22 * x21
        x24 = x19 + x23
        return x24

class LayerScale_Block_CA1(paddle.nn.Layer):
    def __init__(self, ):
        super(LayerScale_Block_CA1, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.class_attention0 = Class_Attention()
        self.x17 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp140 = Mlp14()
        self.x22 = self.create_parameter(dtype='float32', shape=(192,), default_initializer=paddle.nn.initializer.Constant(value=0.0))
    def forward(self, x6, x7):
        x0 = 0
        x1 = 2
        x2 = 0.14433756729740643
        x3 = -2
        x4 = -1
        x5 = 1
        x8 = [x6, x7]
        x9 = paddle.concat(x=x8, axis=1)
        x10 = self.layernorm0(x9)
        x11 = [0]
        x12 = [2147483647]
        x13 = [1]
        x14 = paddle.strided_slice(x=x10, axes=x11, starts=x11, ends=x12, strides=x13)
        x15 = x14[:, x0]
        x16 = self.class_attention0(x15, x10, x2, x3, x4, x5, x1)
        x17 = self.x17
        x18 = x17 * x16
        x19 = x6 + x18
        x20 = self.layernorm1(x19)
        x21 = self.mlp140(x20)
        x22 = self.x22
        x23 = x22 * x21
        x24 = x19 + x23
        return x24

class PatchEmbed(paddle.nn.Layer):
    def __init__(self, ):
        super(PatchEmbed, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=192, kernel_size=(16, 16), stride=16, in_channels=3)
    def forward(self, x1):
        x0 = 1
        x2 = self.conv0(x1)
        x3 = 2
        x4 = paddle.flatten(x=x2, start_axis=2)
        x5 = x4.shape
        x6 = len(x5)
        x7 = []
        for i in range(x6):
            x7.append(i)
        if x0 < 0:
            x8 = x0 + x6
        else:
            x8 = x0
        if x3 < 0:
            x9 = x3 + x6
        else:
            x9 = x3
        x7[x8] = x9
        x7[x9] = x8
        x10 = paddle.transpose(x=x4, perm=x7)
        return x10

class ModuleList2(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList2, self).__init__()
        self.layerscale_block180 = LayerScale_Block18()
        self.layerscale_block40 = LayerScale_Block4()
        self.layerscale_block120 = LayerScale_Block12()
        self.layerscale_block80 = LayerScale_Block8()
        self.layerscale_block90 = LayerScale_Block9()
        self.layerscale_block140 = LayerScale_Block14()
        self.layerscale_block200 = LayerScale_Block20()
        self.layerscale_block150 = LayerScale_Block15()
        self.layerscale_block230 = LayerScale_Block23()
        self.layerscale_block100 = LayerScale_Block10()
        self.layerscale_block50 = LayerScale_Block5()
        self.layerscale_block110 = LayerScale_Block11()
        self.layerscale_block00 = LayerScale_Block0()
        self.layerscale_block60 = LayerScale_Block6()
        self.layerscale_block210 = LayerScale_Block21()
        self.layerscale_block220 = LayerScale_Block22()
        self.layerscale_block10 = LayerScale_Block1()
        self.layerscale_block70 = LayerScale_Block7()
        self.layerscale_block160 = LayerScale_Block16()
        self.layerscale_block170 = LayerScale_Block17()
        self.layerscale_block190 = LayerScale_Block19()
        self.layerscale_block130 = LayerScale_Block13()
        self.layerscale_block30 = LayerScale_Block3()
        self.layerscale_block20 = LayerScale_Block2()
    def forward(self, x0):
        x1 = self.layerscale_block180(x0)
        x2 = self.layerscale_block40(x1)
        x3 = self.layerscale_block120(x2)
        x4 = self.layerscale_block80(x3)
        x5 = self.layerscale_block90(x4)
        x6 = self.layerscale_block140(x5)
        x7 = self.layerscale_block200(x6)
        x8 = self.layerscale_block150(x7)
        x9 = self.layerscale_block230(x8)
        x10 = self.layerscale_block100(x9)
        x11 = self.layerscale_block50(x10)
        x12 = self.layerscale_block110(x11)
        x13 = self.layerscale_block00(x12)
        x14 = self.layerscale_block60(x13)
        x15 = self.layerscale_block210(x14)
        x16 = self.layerscale_block220(x15)
        x17 = self.layerscale_block10(x16)
        x18 = self.layerscale_block70(x17)
        x19 = self.layerscale_block160(x18)
        x20 = self.layerscale_block170(x19)
        x21 = self.layerscale_block190(x20)
        x22 = self.layerscale_block130(x21)
        x23 = self.layerscale_block30(x22)
        x24 = self.layerscale_block20(x23)
        return x24

class ModuleList3(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList3, self).__init__()
        self.layerscale_block_ca10 = LayerScale_Block_CA1()
        self.layerscale_block_ca00 = LayerScale_Block_CA0()
    def forward(self, x0, x1):
        x2 = self.layerscale_block_ca10(x0, x1)
        x3 = self.layerscale_block_ca00(x2, x1)
        return x3

class cait_models(paddle.nn.Layer):
    def __init__(self, ):
        super(cait_models, self).__init__()
        self.patchembed0 = PatchEmbed()
        self.x2 = self.create_parameter(dtype='float32', shape=(1, 1, 192), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.x4 = self.create_parameter(dtype='float32', shape=(1, 196, 192), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.modulelist20 = ModuleList2()
        self.modulelist30 = ModuleList3()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=1000)
    def forward(self, x0):
        x0 = paddle.to_tensor(data=x0)
        x1 = self.patchembed0(x0)
        x2 = self.x2
        x3 = paddle.expand(x=x2, shape=[1, -1, -1])
        x4 = self.x4
        x5 = x1 + x4
        x6 = self.dropout0(x5)
        x7 = self.modulelist20(x6)
        x8 = self.modulelist30(x3, x7)
        x9 = [x8, x7]
        x10 = paddle.concat(x=x9, axis=1)
        x11 = self.layernorm0(x10)
        x12 = [0]
        x13 = [0]
        x14 = [2147483647]
        x15 = [1]
        x16 = paddle.strided_slice(x=x11, axes=x12, starts=x13, ends=x14, strides=x15)
        x17 = 0
        x18 = x16[:, x17]
        x19 = self.linear0(x18)
        return x19

def main(x0):
    # There are 1 inputs.
    # x0: shape-[1, 3, 224, 224], type-float32.
    paddle.disable_static()
    params = paddle.load(r'E:\dataFiles\github\PaddleTr\torch2paddle\pd_model_trace\model.pdparams')
    model = cait_models()
    model.set_dict(params)
    model.eval()
    out = model(x0)
    return out