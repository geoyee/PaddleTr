import paddle.nn as nn


def constant_init(param, **kwargs):
    initializer = nn.initializer.Constant(**kwargs)
    initializer(param, param.block)


def trunc_normal_(param, **kwargs):
    initializer = nn.initializer.TruncatedNormal(**kwargs)
    initializer(param, param.block)