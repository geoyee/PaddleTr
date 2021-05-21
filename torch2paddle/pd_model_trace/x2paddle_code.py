import paddle
import math
from paddle.nn.functional.loss import l1_loss
from x2paddle.op_mapper.dygraph.pytorch2paddle import pytorch_custom_layer as x2paddle_nn
class Attention(paddle.nn.Layer):
    def __init__(self, ):
        super(Attention, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=576)
        self.softmax0 = paddle.nn.Softmax()
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear1 = paddle.nn.Linear(in_features=192, out_features=192)
        self.dropout1 = paddle.nn.Dropout(p=0.0)
    def forward(self, x0, x4, x6, x8, x13, x15, x19):
        x1 = self.linear0(x0)
        x2 = paddle.reshape(x=x1, shape=[1, 197, 3, 3, 64])
        x3 = paddle.transpose(x=x2, perm=[2, 0, 3, 1, 4])
        x5 = x3[x4]
        x7 = x3[x6]
        x9 = x3[x8]
        x10 = x7.shape
        x11 = len(x10)
        x12 = []
        for i in range(x11):
            x12.append(i)
        if x13 < 0:
            x14 = x13 + x11
        else:
            x14 = x13
        if x15 < 0:
            x16 = x15 + x11
        else:
            x16 = x15
        x12[x14] = x16
        x12[x16] = x14
        x17 = paddle.transpose(x=x7, perm=x12)
        x18 = paddle.matmul(x=x5, y=x17)
        x20 = x18 * x19
        x21 = self.softmax0(x20)
        x22 = self.dropout0(x21)
        x23 = paddle.matmul(x=x22, y=x9)
        x24 = x23.shape
        x25 = len(x24)
        x26 = []
        for i in range(x25):
            x26.append(i)
        if x6 < 0:
            x27 = x6 + x25
        else:
            x27 = x6
        if x8 < 0:
            x28 = x8 + x25
        else:
            x28 = x8
        x26[x27] = x28
        x26[x28] = x27
        x29 = paddle.transpose(x=x23, perm=x26)
        x30 = paddle.reshape(x=x29, shape=[1, 197, 192])
        x31 = self.linear1(x30)
        x32 = self.dropout1(x31)
        return x32

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

class Block0(paddle.nn.Layer):
    def __init__(self, ):
        super(Block0, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention0 = Attention()
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp70 = Mlp7()
    def forward(self, x0, x2, x3, x4, x5, x6, x7):
        x1 = self.layernorm0(x0)
        x8 = self.attention0(x1, x2, x3, x4, x5, x6, x7)
        x9 = x0 + x8
        x10 = self.layernorm1(x9)
        x11 = self.mlp70(x10)
        x12 = x9 + x11
        return x12

class Block1(paddle.nn.Layer):
    def __init__(self, ):
        super(Block1, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention0 = Attention()
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp10 = Mlp1()
    def forward(self, x0, x2, x3, x4, x5, x6, x7):
        x1 = self.layernorm0(x0)
        x8 = self.attention0(x1, x2, x3, x4, x5, x6, x7)
        x9 = x0 + x8
        x10 = self.layernorm1(x9)
        x11 = self.mlp10(x10)
        x12 = x9 + x11
        return x12

class Block2(paddle.nn.Layer):
    def __init__(self, ):
        super(Block2, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention0 = Attention()
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp30 = Mlp3()
    def forward(self, x0, x2, x3, x4, x5, x6, x7):
        x1 = self.layernorm0(x0)
        x8 = self.attention0(x1, x2, x3, x4, x5, x6, x7)
        x9 = x0 + x8
        x10 = self.layernorm1(x9)
        x11 = self.mlp30(x10)
        x12 = x9 + x11
        return x12

class Block3(paddle.nn.Layer):
    def __init__(self, ):
        super(Block3, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention0 = Attention()
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp80 = Mlp8()
    def forward(self, x0, x2, x3, x4, x5, x6, x7):
        x1 = self.layernorm0(x0)
        x8 = self.attention0(x1, x2, x3, x4, x5, x6, x7)
        x9 = x0 + x8
        x10 = self.layernorm1(x9)
        x11 = self.mlp80(x10)
        x12 = x9 + x11
        return x12

class Block4(paddle.nn.Layer):
    def __init__(self, ):
        super(Block4, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention0 = Attention()
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp20 = Mlp2()
    def forward(self, x0, x2, x3, x4, x5, x6, x7):
        x1 = self.layernorm0(x0)
        x8 = self.attention0(x1, x2, x3, x4, x5, x6, x7)
        x9 = x0 + x8
        x10 = self.layernorm1(x9)
        x11 = self.mlp20(x10)
        x12 = x9 + x11
        return x12

class Block5(paddle.nn.Layer):
    def __init__(self, ):
        super(Block5, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention0 = Attention()
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp110 = Mlp11()
    def forward(self, x0, x2, x3, x4, x5, x6, x7):
        x1 = self.layernorm0(x0)
        x8 = self.attention0(x1, x2, x3, x4, x5, x6, x7)
        x9 = x0 + x8
        x10 = self.layernorm1(x9)
        x11 = self.mlp110(x10)
        x12 = x9 + x11
        return x12

class Block6(paddle.nn.Layer):
    def __init__(self, ):
        super(Block6, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention0 = Attention()
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp50 = Mlp5()
    def forward(self, x0, x2, x3, x4, x5, x6, x7):
        x1 = self.layernorm0(x0)
        x8 = self.attention0(x1, x2, x3, x4, x5, x6, x7)
        x9 = x0 + x8
        x10 = self.layernorm1(x9)
        x11 = self.mlp50(x10)
        x12 = x9 + x11
        return x12

class Block7(paddle.nn.Layer):
    def __init__(self, ):
        super(Block7, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention0 = Attention()
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp100 = Mlp10()
    def forward(self, x0, x2, x3, x4, x5, x6, x7):
        x1 = self.layernorm0(x0)
        x8 = self.attention0(x1, x2, x3, x4, x5, x6, x7)
        x9 = x0 + x8
        x10 = self.layernorm1(x9)
        x11 = self.mlp100(x10)
        x12 = x9 + x11
        return x12

class Block8(paddle.nn.Layer):
    def __init__(self, ):
        super(Block8, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention0 = Attention()
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp00 = Mlp0()
    def forward(self, x6):
        x0 = 0.125
        x1 = -1
        x2 = -2
        x3 = 2
        x4 = 1
        x5 = 0
        x7 = self.layernorm0(x6)
        x8 = self.attention0(x7, x5, x4, x3, x2, x1, x0)
        x9 = x6 + x8
        x10 = self.layernorm1(x9)
        x11 = self.mlp00(x10)
        x12 = x9 + x11
        return x0, x1, x2, x3, x4, x5, x12

class Block9(paddle.nn.Layer):
    def __init__(self, ):
        super(Block9, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention0 = Attention()
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp40 = Mlp4()
    def forward(self, x0, x2, x3, x4, x5, x6, x7):
        x1 = self.layernorm0(x0)
        x8 = self.attention0(x1, x2, x3, x4, x5, x6, x7)
        x9 = x0 + x8
        x10 = self.layernorm1(x9)
        x11 = self.mlp40(x10)
        x12 = x9 + x11
        return x12

class Block10(paddle.nn.Layer):
    def __init__(self, ):
        super(Block10, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention0 = Attention()
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp90 = Mlp9()
    def forward(self, x0, x2, x3, x4, x5, x6, x7):
        x1 = self.layernorm0(x0)
        x8 = self.attention0(x1, x2, x3, x4, x5, x6, x7)
        x9 = x0 + x8
        x10 = self.layernorm1(x9)
        x11 = self.mlp90(x10)
        x12 = x9 + x11
        return x12

class Block11(paddle.nn.Layer):
    def __init__(self, ):
        super(Block11, self).__init__()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.attention0 = Attention()
        self.layernorm1 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.mlp60 = Mlp6()
    def forward(self, x0, x2, x3, x4, x5, x6, x7):
        x1 = self.layernorm0(x0)
        x8 = self.attention0(x1, x2, x3, x4, x5, x6, x7)
        x9 = x0 + x8
        x10 = self.layernorm1(x9)
        x11 = self.mlp60(x10)
        x12 = x9 + x11
        return x12

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

class Blocks(paddle.nn.Layer):
    def __init__(self, ):
        super(Blocks, self).__init__()
        self.block80 = Block8()
        self.block40 = Block4()
        self.block00 = Block0()
        self.block100 = Block10()
        self.block110 = Block11()
        self.block10 = Block1()
        self.block30 = Block3()
        self.block50 = Block5()
        self.block70 = Block7()
        self.block60 = Block6()
        self.block20 = Block2()
        self.block90 = Block9()
    def forward(self, x0):
        x1,x2,x3,x4,x5,x6,x7 = self.block80(x0)
        x8 = self.block40(x7, x6, x5, x4, x3, x2, x1)
        x9 = self.block00(x8, x6, x5, x4, x3, x2, x1)
        x10 = self.block100(x9, x6, x5, x4, x3, x2, x1)
        x11 = self.block110(x10, x6, x5, x4, x3, x2, x1)
        x12 = self.block10(x11, x6, x5, x4, x3, x2, x1)
        x13 = self.block30(x12, x6, x5, x4, x3, x2, x1)
        x14 = self.block50(x13, x6, x5, x4, x3, x2, x1)
        x15 = self.block70(x14, x6, x5, x4, x3, x2, x1)
        x16 = self.block60(x15, x6, x5, x4, x3, x2, x1)
        x17 = self.block20(x16, x6, x5, x4, x3, x2, x1)
        x18 = self.block90(x17, x6, x5, x4, x3, x2, x1)
        return x18

class VisionTransformer(paddle.nn.Layer):
    def __init__(self, ):
        super(VisionTransformer, self).__init__()
        self.patchembed0 = PatchEmbed()
        self.x2 = self.create_parameter(dtype='float32', shape=(1, 1, 192), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.x6 = self.create_parameter(dtype='float32', shape=(1, 197, 192), default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.blocks0 = Blocks()
        self.layernorm0 = paddle.nn.LayerNorm(normalized_shape=[192], epsilon=1e-06)
        self.linear0 = paddle.nn.Linear(in_features=192, out_features=1000)
    def forward(self, x0):
        x0 = paddle.to_tensor(data=x0)
        x1 = self.patchembed0(x0)
        x2 = self.x2
        x3 = paddle.expand(x=x2, shape=[1, -1, -1])
        x4 = [x3, x1]
        x5 = paddle.concat(x=x4, axis=1)
        x6 = self.x6
        x7 = x5 + x6
        x8 = self.dropout0(x7)
        x9 = self.blocks0(x8)
        x10 = self.layernorm0(x9)
        x11 = [0]
        x12 = [0]
        x13 = [2147483647]
        x14 = [1]
        x15 = paddle.strided_slice(x=x10, axes=x11, starts=x12, ends=x13, strides=x14)
        x16 = 0
        x17 = x15[:, x16]
        x18 = self.linear0(x17)
        return x18

def main(x0):
    # There are 1 inputs.
    # x0: shape-[1, 3, 224, 224], type-float32.
    paddle.disable_static()
    params = paddle.load('E:\dataFiles\github\PaddleTr\torch2paddle\pd_model_trace\model.pdparams')
    model = VisionTransformer()
    model.set_dict(params)
    model.eval()
    out = model(x0)
    return out