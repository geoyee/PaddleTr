import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from functools import partial

from deit_paddle.pimm.models.vision_transformer import Mlp, PatchEmbed , _cfg
from deit_paddle.pimm.models.registry import register_model
from deit_paddle.pimm.models.layers import trunc_normal_, constant_init, DropPath


# __all__ = [
#     'cait_M48', 'cait_M36', 'cait_M4',
#     'cait_S36', 'cait_S24','cait_S24_224',
#     'cait_XS24','cait_XXS24','cait_XXS24_224',
#     'cait_XXS36','cait_XXS36_224'
# ]


class Class_Attention(nn.Layer):
    # taken from https://github.com/rwightman/pypaddle-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA 
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.k = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.v = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x ):
        B, N, C = x.shape
        q = self.q(x[:, 0]).unsqueeze(1).reshape((B, 1, self.num_heads, C // self.num_heads)).transpose((0, 2, 1, 3))
        k = self.k(x).reshape((B, N, self.num_heads, C //
                               self.num_heads)).transpose((0, 2, 1, 3))
        q = q * self.scale
        v = self.v(x).reshape((B, N, self.num_heads, C //
                               self.num_heads)).transpose((0, 2, 1, 3))
        attn = q.matmul(k.transpose((0, 1, 3, 2)))
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x_cls = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((B, 1, C))
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        return x_cls    
        

class LayerScale_Block_CA(nn.Layer):
    # taken from https://github.com/rwightman/pypaddle-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add CA and LayerScale
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Class_Attention,
                 Mlp_block=Mlp, init_values=1e-4, epsilon=1e-6):
        super().__init__()
        self.norm1 = norm_layer(dim, epsilon=epsilon)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else None
        self.norm2 = norm_layer(dim, epsilon=epsilon)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = paddle.create_parameter(shape=(dim,), dtype='float32', \
                       default_initializer=nn.initializer.Constant(value=init_values))
        self.gamma_2 = paddle.create_parameter(shape=(dim,), dtype='float32', \
                       default_initializer=nn.initializer.Constant(value=init_values))
    
    def forward(self, x, x_cls):
        u = paddle.concat((x_cls, x), axis=1)
        if self.drop_path is not None:
            x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
            x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        else:
            x_cls = x_cls + self.gamma_1 * self.attn(self.norm1(u))
            x_cls = x_cls + self.gamma_2 * self.mlp(self.norm2(x_cls))
        return x_cls 
        
        
class Attention_talking_head(nn.Layer):
    # taken from https://github.com/rwightman/pypaddle-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape((B, N, 3, self.num_heads, C //
                                   self.num_heads)).transpose((2, 0, 3, 1, 4))
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = (q.matmul(k.transpose((0, 1, 3, 2))))
        attn = self.proj_l(attn.transpose((0, 2, 3, 1))).transpose((0, 3, 1, 2))
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.proj_w(attn.transpose((0, 2, 3, 1))).transpose((0, 3, 1, 2))
        attn = self.attn_drop(attn)
        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale_Block(nn.Layer):
    # taken from https://github.com/rwightman/pypaddle-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add layerScale
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Attention_talking_head,
                 Mlp_block=Mlp, init_values=1e-4, epsilon=1e-6):
        super().__init__()
        self.norm1 = norm_layer(dim, epsilon=epsilon)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else None
        self.norm2 = norm_layer(dim, epsilon=epsilon)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = paddle.create_parameter(shape=(dim,), dtype='float32', \
                       default_initializer=nn.initializer.Constant(value=init_values))
        self.gamma_2 = paddle.create_parameter(shape=(dim,), dtype='float32', \
                       default_initializer=nn.initializer.Constant(value=init_values))

    def forward(self, x):
        if self.drop_path is not None:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.gamma_1 * self.attn(self.norm1(x))
            x = x + self.gamma_2 * self.mlp(self.norm2(x))
        return x 
    
    
class cait_models(nn.Layer):
    # taken from https://github.com/rwightman/pypaddle-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to adapt to our cait models
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers = LayerScale_Block,
                 block_layers_token = LayerScale_Block_CA,
                 Patch_layer=PatchEmbed,
                 act_layer=nn.GELU,
                 Attention_block = Attention_talking_head,
                 Mlp_block=Mlp,
                 init_scale=0,
                 Attention_block_token_only=Class_Attention,
                 Mlp_block_token_only= Mlp, 
                 depth_token_only=2,
                 mlp_ratio_clstk = 4.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  
        self.patch_embed = Patch_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = paddle.create_parameter(shape=(1, 1, embed_dim), dtype='float32', \
                         default_initializer=nn.initializer.Constant(value=0))
        self.pos_embed = paddle.create_parameter(shape=(1, num_patches, embed_dim), dtype='float32', \
                         default_initializer=nn.initializer.Constant(value=0))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.LayerList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, Attention_block=Attention_block, Mlp_block=Mlp_block, init_values=init_scale)
            for i in range(depth)])
        self.blocks_token_only = nn.LayerList([
            block_layers_token(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio_clstk, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=norm_layer,
                act_layer=act_layer, Attention_block=Attention_block_token_only,
                Mlp_block=Mlp_block_token_only, init_values=init_scale)
            for i in range(depth_token_only)])   
        self.norm = norm_layer(embed_dim)
        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else None
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                constant_init(m.bias, value=0)
        elif isinstance(m, nn.LayerNorm):
            constant_init(m.bias, value=0)
            constant_init(m.weight, value=1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = paddle.expand(self.cls_token, [B, -1, -1])
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for i , blk in enumerate(self.blocks):
            x = blk(x)
        for i , blk in enumerate(self.blocks_token_only):
            cls_tokens = blk(x, cls_tokens)
        x = paddle.concat((cls_tokens, x), axis=1)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x) if self.head is not None else x
        return x 

       
@register_model
def cait_XXS24_224(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 224, patch_size=16, embed_dim=192, depth=24, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        init_scale=1e-5,
        depth_token_only=2, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        params = paddle.load(pretrained)
        model.set_state_dict(params)
    return model 

## 下面和cait_XXS24_224差不多的
# @register_model
# def cait_XXS24(pretrained=False, **kwargs):
#     model = cait_models(
#         img_size= 384,patch_size=16, embed_dim=192, depth=24, num_heads=4, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         init_scale=1e-5,
#         depth_token_only=2,**kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = paddle.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/XXS24_384.pth",
#             map_location="cpu", check_hash=True
#         )
#         checkpoint_no_module = {}
#         for k in model.state_dict().keys():
#             checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
#         model.load_state_dict(checkpoint_no_module)      
#     return model 


# @register_model
# def cait_XXS36_224(pretrained=False, **kwargs):
#     model = cait_models(
#         img_size= 224,patch_size=16, embed_dim=192, depth=36, num_heads=4, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         init_scale=1e-5,
#         depth_token_only=2,**kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = paddle.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/XXS36_224.pth",
#             map_location="cpu", check_hash=True
#         )
#         checkpoint_no_module = {}
#         for k in model.state_dict().keys():
#             checkpoint_no_module[k] = checkpoint["model"]['module.'+k] 
#         model.load_state_dict(checkpoint_no_module)  
#     return model 


# @register_model
# def cait_XXS36(pretrained=False, **kwargs):
#     model = cait_models(
#         img_size= 384,patch_size=16, embed_dim=192, depth=36, num_heads=4, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         init_scale=1e-5,
#         depth_token_only=2,**kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = paddle.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/XXS36_384.pth",
#             map_location="cpu", check_hash=True
#         )
#         checkpoint_no_module = {}
#         for k in model.state_dict().keys():
#             checkpoint_no_module[k] = checkpoint["model"]['module.'+k] 
#         model.load_state_dict(checkpoint_no_module)
#     return model 


# @register_model
# def cait_XS24(pretrained=False, **kwargs):
#     model = cait_models(
#         img_size= 384,patch_size=16, embed_dim=288, depth=24, num_heads=6, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         init_scale=1e-5,
#         depth_token_only=2,**kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = paddle.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/XS24_384.pth",
#             map_location="cpu", check_hash=True
#         )
#         checkpoint_no_module = {}
#         for k in model.state_dict().keys():
#             checkpoint_no_module[k] = checkpoint["model"]['module.'+k]        
#         model.load_state_dict(checkpoint_no_module)     
#     return model 


# @register_model
# def cait_S24_224(pretrained=False, **kwargs):
#     model = cait_models(
#         img_size= 224,patch_size=16, embed_dim=384, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         init_scale=1e-5,
#         depth_token_only=2,**kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = paddle.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/S24_224.pth",
#             map_location="cpu", check_hash=True
#         )
#         checkpoint_no_module = {}
#         for k in model.state_dict().keys():
#             checkpoint_no_module[k] = checkpoint["model"]['module.'+k]         
#         model.load_state_dict(checkpoint_no_module)   
#     return model 


# @register_model
# def cait_S24(pretrained=False, **kwargs):
#     model = cait_models(
#         img_size= 384,patch_size=16, embed_dim=384, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         init_scale=1e-5,
#         depth_token_only=2,**kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = paddle.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/S24_384.pth",
#             map_location="cpu", check_hash=True
#         )
#         checkpoint_no_module = {}
#         for k in model.state_dict().keys():
#             checkpoint_no_module[k] = checkpoint["model"]['module.'+k]           
#         model.load_state_dict(checkpoint_no_module)
#     return model 


# @register_model
# def cait_S36(pretrained=False, **kwargs):
#     model = cait_models(
#         img_size= 384,patch_size=16, embed_dim=384, depth=36, num_heads=8, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         init_scale=1e-6,
#         depth_token_only=2,**kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = paddle.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/S36_384.pth",
#             map_location="cpu", check_hash=True
#         )
#         checkpoint_no_module = {}
#         for k in model.state_dict().keys():
#             checkpoint_no_module[k] = checkpoint["model"]['module.'+k]    
#         model.load_state_dict(checkpoint_no_module)
#     return model 


# @register_model
# def cait_M36(pretrained=False, **kwargs):
#     model = cait_models(
#         img_size= 384, patch_size=16, embed_dim=768, depth=36, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         init_scale=1e-6,
#         depth_token_only=2,**kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = paddle.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/M36_384.pth",
#             map_location="cpu", check_hash=True
#         )
#         checkpoint_no_module = {}
#         for k in model.state_dict().keys():
#             checkpoint_no_module[k] = checkpoint["model"]['module.'+k]   
#         model.load_state_dict(checkpoint_no_module)
#     return model 


# @register_model
# def cait_M48(pretrained=False, **kwargs):
#     model = cait_models(
#         img_size= 448 , patch_size=16, embed_dim=768, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         init_scale=1e-6,
#         depth_token_only=2,**kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = paddle.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/M48_448.pth",
#             map_location="cpu", check_hash=True
#         )
#         checkpoint_no_module = {}
#         for k in model.state_dict().keys():
#             checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
#         model.load_state_dict(checkpoint_no_module)
#     return model