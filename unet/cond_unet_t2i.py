import clip
import fvcore.common.config
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from functools import partial
from einops import rearrange, reduce


### Compared to unet4:
# 1. add group norm in ConditionEncoder
######## Attention Layer ##########

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        # self.class_token_pos = nn.Parameter(torch.zeros(1, 1, num_pos_feats * 2))
        # self.class_token_pos

    def forward(self, x):
        # x: b, h, w, d
        num_feats = x.shape[3]
        num_pos_feats = num_feats // 2
        # mask = tensor_list.mask
        mask = torch.zeros(x.shape[0], x.shape[1], x.shape[2], device=x.device).to(torch.bool)
        batch = mask.shape[0]
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-5
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # pos = torch.cat((pos_y, pos_x), dim=3).flatten(1, 2)
        pos = torch.cat((pos_y, pos_x), dim=3).contiguous()
        '''
        pos_x: b ,h, w, d//2
        pos_y: b, h, w, d//2
        pos: b, h, w, d
        '''
        return pos

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, feature_size, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(feature_size[0], num_pos_feats)
        self.col_embed = nn.Embedding(feature_size[1], num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return torch.cat([x, pos], dim=1)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAtt(nn.Module):
    def __init__(self, in_dim):
        super(SpatialAtt, self).__init__()
        self.map = nn.Conv2d(in_dim, 1, 1)
        self.q_conv = nn.Conv2d(1, 1, 1)
        self.k_conv = nn.Conv2d(1, 1, 1)
        self.activation = nn.Softsign()

    def forward(self, x):
        b, _, h, w = x.shape
        att = self.map(x) # b, 1, h, w
        q = self.q_conv(att) # b, 1, h, w
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = self.k_conv(att)
        k = rearrange(k, 'b c h w -> b c (h w)')
        att = rearrange(att, 'b c h w -> b (h w) c')
        att = F.softmax(q @ k, dim=-1) @ att # b, hw, 1
        att = att.reshape(b, 1, h, w)
        return self.activation(att) * x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        # self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class BasicAttetnionLayer(nn.Module):
    def __init__(self, embed_dim=128, nhead=8, ffn_dim=512, window_size1=[4, 4],
                 window_size2=[1, 1], dropout=0.1):
        super().__init__()
        self.window_size1 = window_size1
        self.window_size2 = window_size2
        self.avgpool_q = nn.AvgPool2d(kernel_size=window_size1)
        self.avgpool_k = nn.AvgPool2d(kernel_size=window_size2)
        self.softmax = nn.Softmax(dim=-1)
        self.nhead = nhead

        self.q_lin = nn.Linear(embed_dim, embed_dim)
        self.k_lin = nn.Linear(embed_dim, embed_dim)
        self.v_lin = nn.Linear(embed_dim, embed_dim)

        self.mlp = Mlp(in_features=embed_dim, hidden_features=ffn_dim, drop=dropout)
        self.pos_enc = PositionEmbeddingSine(embed_dim)
        self.concat_conv = nn.Conv2d(2 * embed_dim, embed_dim, 1)
        self.gn = nn.GroupNorm(8, embed_dim)

        self.out_conv = nn.Conv2d(embed_dim, embed_dim, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x1, x2): # x1 for q (conditional input), x2 for k,v
        B, C1, H1, W1 = x1.shape
        _, C2, H2, W2 = x2.shape
        # x1 = x1.permute(0, 2, 3, 1).contiguous() # B, H1, W1, C1
        shortcut = x2 + self.concat_conv(torch.cat(
            [F.interpolate(x1, size=(H2, W2), mode='bilinear', align_corners=True),
             x2], dim=1))
        shortcut = self.gn(shortcut)
        pad_l = pad_t = 0
        pad_r = (self.window_size1[1] - W1 % self.window_size1[1]) % self.window_size1[1]
        pad_b = (self.window_size1[0] - H1 % self.window_size1[0]) % self.window_size1[0]
        x1 = F.pad(x1, (pad_l, pad_r, pad_t, pad_b, 0, 0))
        _, _, H1p, W1p = x1.shape
        # x2 = x2.permute(0, 2, 3, 1).contiguous()  # B, H2, W2, C2
        pad_l = pad_t = 0
        pad_r = (self.window_size2[1] - W2 % self.window_size2[1]) % self.window_size2[1]
        pad_b = (self.window_size2[0] - H2 % self.window_size2[0]) % self.window_size2[0]
        x2 = F.pad(x2, (pad_l, pad_r, pad_t, pad_b, 0, 0))
        _, _, H2p, W2p = x2.shape
        # x1g = x1 #B, C1, H1p, W1p
        # x2g = x2 #B, C2, H2p, W2p
        x1_s = self.avgpool_q(x1)
        qg = self.avgpool_q(x1).permute(0, 2, 3, 1).contiguous()
        qg = qg + self.pos_enc(qg)
        qg= qg.view(B, -1, C2)
        kg = self.avgpool_k(x2).permute(0, 2, 3, 1).contiguous()
        kg = kg + self.pos_enc(kg)
        kg = kg.view(B, -1, C1)
        num_window_q = qg.shape[1]
        num_window_k = kg.shape[1]
        qg = self.q_lin(qg).reshape(B, num_window_q, self.nhead, C1 // self.nhead).permute(0, 2, 1,
                                                                                                      3).contiguous()
        kg2 = self.k_lin(kg).reshape(B, num_window_k, self.nhead, C1 // self.nhead).permute(0, 2, 1,
                                                                                                       3).contiguous()
        vg = self.v_lin(kg).reshape(B, num_window_k, self.nhead, C1 // self.nhead).permute(0, 2, 1,
                                                                                                      3).contiguous()
        kg = kg2
        attn = (qg @ kg.transpose(-2, -1))
        attn = self.softmax(attn)
        qg = (attn @ vg).transpose(1, 2).reshape(B, num_window_q, C1)
        qg = qg.transpose(1, 2).reshape(B, C1, H1p // self.window_size1[0], W1p // self.window_size1[1])
        # qg = F.interpolate(qg, size=(H1p, W1p), mode='bilinear', align_corners=False)
        x1_s = x1_s + qg
        x1_s = x1_s + self.mlp(x1_s)
        x1_s = F.interpolate(x1_s, size=(H2, W2), mode='bilinear', align_corners=True)
        x1_s = shortcut + self.out_conv(x1_s)
        # x1_s = self.out_norm(x1_s)
        return x1_s

class RelationNet(nn.Module):
    def __init__(self, in_channel1=128, in_channel2=128, nhead=8, layers=3, embed_dim=128, ffn_dim=512,
                 window_size1= [4, 4], window_size2=[1, 1]):
        # self.attention = BasicAttetnionLayer(embed_dim=embed_dim, nhead=nhead, ffn_dim=ffn_dim,
        #                                      window_size1=window_size1, window_size2=window_size2, dropout=0.1)
        super().__init__()
        self.layers = layers
        self.input_conv1 = nn.Sequential(
            nn.Conv2d(in_channel1, embed_dim, 1),
            nn.BatchNorm2d(embed_dim, momentum=0.03, eps=0.001),
        )
        self.input_conv2 = nn.Sequential(
            nn.Conv2d(in_channel2, embed_dim, 1),
            nn.BatchNorm2d(embed_dim, momentum=0.03, eps=0.001),
        )
        # self.input_conv1 = ConvModule(in_channel1,
        #                               embed_dim,
        #                               1,
        #                               norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        #                               act_cfg=None)
        # self.input_conv2 = ConvModule(in_channel2,
        #                               embed_dim,
        #                               1,
        #                               norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        #                               act_cfg=None)
        # self.input_conv2 = nn.Linear(in_channel2, embed_dim)
        self.attentions = nn.ModuleList()
        for i in range(layers):
            self.attentions.append(
                BasicAttetnionLayer(embed_dim=embed_dim, nhead=nhead, ffn_dim=ffn_dim,
                                     window_size1=window_size1, window_size2=window_size2, dropout=0.1)
            )

    def forward(self, cond, feat):
        # cluster = cluster.unsqueeze(0).repeat(feature.shape[0], 1, 1, 1)
        cond = self.input_conv1(cond)
        feat = self.input_conv2(feat)
        for att in self.attentions:
            feat = att(cond, feat)
        return feat



################# U-Net model defenition ####################

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class ResnetDownsampleBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = nn.Sequential(
            WeightStandardizedConv2d(dim_out, dim_out, 3, stride=2, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU()
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(
                F.interpolate(x, size=h.shape[-2:], mode="bilinear")
                )

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)

class CondAttention(nn.Module): # Multi-Scale Window-Attention
    def __init__(self, dim, dim2, hidden_dim, heads=4, window_size_q=[4, 4],
                 window_size_k=[[4, 4], [2, 2], [1, 1]], drop=0.1):
        super(CondAttention, self).__init__()
        # assert  dim == heads * dim_head
        dim_head = hidden_dim // heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        # hidden_dim = dim_head * heads
        # self.qkv = nn.Conv3d(dim, hidden_dim*3, 1)
        self.q_lin = nn.Linear(dim, hidden_dim)
        self.k_lin = nn.Linear(dim2, hidden_dim)
        self.v_lin = nn.Linear(dim2, hidden_dim)
        self.pos_enc = PositionEmbeddingSine(hidden_dim)
        self.window_size_q = window_size_q
        self.avgpool_q = nn.AdaptiveAvgPool2d(output_size=window_size_q)
        # self.avgpool_ks = nn.ModuleList()
        # for i in range(len(window_size_k)):
        #     self.avgpool_ks.append(nn.AdaptiveAvgPool2d(output_size=window_size_k[i]))
        self.softmax = nn.Softmax(dim=-1)
        self.mlp = Mlp(in_features=hidden_dim, hidden_features=hidden_dim*2, out_features=dim, drop=drop)
        self.out_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GroupNorm(8, dim)
        )

    def forward(self, x, cond):
        # x: B, C, H, W
        # cond: B, C
        B, C, H, W = x.shape
        shortcut = x
        q_s = self.avgpool_q(x)
        length = q_s.shape[-2] * q_s.shape[-1]
        kg = cond.unsqueeze(1)#.expand(-1, length, -1)
        qg = self.avgpool_q(x).permute(0, 2, 3, 1).contiguous()
        qg = qg + self.pos_enc(qg)
        qg = qg.view(B, -1, C)


        num_window_q = qg.shape[1]
        num_window_k = kg.shape[1]
        qg = self.q_lin(qg).reshape(B, num_window_q, self.heads, C // self.heads).permute(0, 2, 1,
                                                                                           3).contiguous()
        kg2 = self.k_lin(kg).reshape(B, num_window_k, self.heads, C // self.heads).permute(0, 2, 1,
                                                                                            3).contiguous()
        vg = self.v_lin(kg).reshape(B, num_window_k, self.heads, C // self.heads).permute(0, 2, 1,
                                                                                           3).contiguous()
        kg = kg2
        attn = (qg @ kg.transpose(-2, -1))
        attn = self.softmax(attn)
        qg = (attn @ vg).transpose(1, 2).reshape(B, num_window_q, C)
        qg = qg.transpose(1, 2).reshape(B, C, self.window_size_q[0], self.window_size_q[1])
        # qg = F.interpolate(qg, size=(H1p, W1p), mode='bilinear', align_corners=False)
        q_s = q_s + qg
        q_s = q_s + self.mlp(q_s)
        q_s = F.interpolate(q_s, size=(H, W), mode='bilinear', align_corners=False)
        out = shortcut + self.out_conv(q_s)
        return out

class ConditionEncoder(nn.Module):
    def __init__(self,
                 down_dim_mults=(2, 4, 8),
                 dim=64,
                 in_dim=1,
                 out_dim=64):
        super(ConditionEncoder, self).__init__()
        self.init_conv = nn.Sequential(
                            nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1),
                            nn.GroupNorm(num_groups=min(dim // 4, 8), num_channels=dim),
        )
        self.num_resolutions = len(down_dim_mults)
        self.downs = nn.ModuleList()
        in_mults = (1,) + tuple(down_dim_mults[:-1])
        in_dims = [mult*dim for mult in in_mults]
        out_dims = [mult*dim for mult in down_dim_mults]
        for i_level in range(self.num_resolutions):
            block_in = in_dims[i_level]
            block_out = out_dims[i_level]
            self.downs.append(ResnetDownsampleBlock(dim=block_in,
                                     dim_out=block_out))
        if self.num_resolutions < 1:
            self.out_conv = nn.Conv2d(dim, out_dim, 1)
        else:
            self.out_conv = nn.Conv2d(out_dims[-1], out_dim, 1)

    def forward(self, x):
        x = self.init_conv(x)
        for down_layer in self.downs:
            x = down_layer(x)
        x = self.out_conv(x)
        return x


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        cond_in_dim=1,
        cond_dim=64,
        cond_dim_mults=(2, 4, 8),
        channels=1,
        out_mul=1,
        self_condition=False,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        window_sizes1=[[16, 16], [8, 8], [4, 4], [2, 2]],
        window_sizes2=[[16, 16], [8, 8], [4, 4], [2, 2]],
        fourier_scale=16,
        precondition=False,
        ckpt_path=None,
        ignore_keys=[],
        cfg={},
        **kwargs
    ):
        super().__init__()

        # determine dimensions
        self.cond_pe = cfg.get('cond_pe', False)
        num_pos_feats = cfg.num_pos_feats if self.cond_pe else 0
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        # self.init_conv_mask = nn.Sequential(
        #     nn.Conv2d(cond_in_dim, cond_dim, 3, padding=1),
        #     nn.GroupNorm(num_groups=min(init_dim // 4, 8), num_channels=init_dim),
        #     nn.SiLU(),
        #     nn.Conv2d(cond_dim, cond_dim, 3, padding=1),
        # )
        # self.init_conv_mask = ConditionEncoder(down_dim_mults=cond_dim_mults, dim=cond_dim,
        #                                        in_dim=cond_in_dim, out_dim=init_dim)

        self.clip, _ = clip.load(cfg.cond_net, device='cpu')
        self.init_conv = nn.Sequential(
            nn.Conv2d(input_channels, init_dim, 7, padding=3),
            nn.GroupNorm(num_groups=min(init_dim // 4, 8), num_channels=init_dim),
        )

        if self.cond_pe:
            self.cond_pos_embedding = nn.Sequential(
                PositionEmbeddingLearned(
                    feature_size=cfg.cond_feature_size, num_pos_feats=cfg.num_pos_feats//2),
                nn.Conv2d(num_pos_feats + init_dim, init_dim, 1)
            )
        # self.init_conv_mask = nn.Conv2d(1, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        dims_rev = dims[::-1]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.projects = nn.ModuleList()
        if cfg.cond_net == 'ViT-B/32':
            self.projects.append(nn.Linear(512, dims[1]))
            self.projects.append(nn.Linear(512, dims[2]))
            self.projects.append(nn.Linear(512, dims[3]))
            self.projects.append(nn.Linear(512, dims[4]))
        elif cfg.cond_net == 'ViT-B/16':
            self.projects.append(nn.Linear(512, dims[1]))
            self.projects.append(nn.Linear(512, dims[2]))
            self.projects.append(nn.Linear(512, dims[3]))
            self.projects.append(nn.Linear(512, dims[4]))
        else:
            raise NotImplementedError

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = GaussianFourierProjection(dim//2, scale=fourier_scale)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        # self.downs_mask = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.relation_layers_down = nn.ModuleList([])
        self.relation_layers_up = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))
            # self.downs_mask.append(nn.ModuleList([
            #     block_klass(dim_in, dim_in, time_emb_dim=time_dim),
            #     # block_klass(dim_in, dim_in, time_emb_dim=time_dim),
            #     Residual(PreNorm(dim_in, LinearAttention(dim_in))),
            #     Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            # ]))
            self.relation_layers_down.append(CondAttention(dim=dims[ind], dim2=dims[ind+1], hidden_dim=dims[ind], heads=8,
                                                  window_size_q=window_sizes1[ind], window_size_k=window_sizes2[ind])
                                      )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.decouple1 = nn.Sequential(
            nn.GroupNorm(num_groups=min(mid_dim // 4, 8), num_channels=mid_dim),
            nn.Conv2d(mid_dim, mid_dim, 3, padding=1),
            SpatialAtt(mid_dim))

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))
            self.relation_layers_up.append(CondAttention(dim=dims_rev[ind], dim2=dims_rev[ind],
                                                         hidden_dim=dims_rev[ind], heads=8,
                                                         window_size_q=window_sizes1[::-1][ind],
                                                         window_size_k=window_sizes2[::-1][ind])
                                           )

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim * out_mul, 1)

        self.precondition = precondition
        # self.init_weights()
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        fix_bb = cfg.get('fix_bb', False)
        if fix_bb:
            for n, p in self.clip.named_parameters():
                p.requires_grad = False

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["model"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        msg = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
        print('==>Load Unet Info: ', msg)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x, time, mask, x_self_cond = None, sigma_max=1, *args, **kwargs):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)
        t = time.reshape(-1, 1, 1, 1)
        # sigma = t.sqrt()
        # s_sigma = sigma / sigma_max
        c_skip1 = -1 + t
        c_skip2 = t.sqrt()
        # c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_out1 = t / (t + 1).sqrt()
        c_out2 = (1 - t).sqrt() / (1 + t).sqrt()
        c_in = 1
        c_noise = time.log()

        x_clone = x.clone()
        x = c_in * x
        # mask = torch.cat([], dim=1)
        text_emb = self.clip.encode_text(mask)
        # if self.cond_pe:
        #     m = self.cond_pos_embedding(m)
        x = self.init_conv(x)
        r = x.clone()

        t_emb = self.time_mlp(c_noise)

        h = []
        hm = []
        for i, layer in enumerate(self.projects):
            hm.append(layer(text_emb))

        for i, ((block1, block2, attn, downsample), relation_layer) \
                in enumerate(zip(self.downs, self.relation_layers_down)):
            x = block1(x, t_emb)
            h.append(x)
            # h2.append(x.clone())
            # m = m_block(m, t)
            # hm.append(m)
            # hm2.append(m.clone())

            x = relation_layer(x, hm[i])

            x = block2(x, t_emb)
            x = attn(x)
            h.append(x)
            # h2.append(x.clone())

            x = downsample(x)
            # m = m_downsample(m)


        # x = x + F.interpolate(hm[-1], size=x.shape[2:], mode="bilinear", align_corners=True)
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)
        x1 = x + self.decouple1(x)

        x = x1
        for (block1, block2, attn, upsample), relation_layer in zip(self.ups, self.relation_layers_up):
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t_emb)
            x = relation_layer(x, hm.pop())
            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t_emb)
            x = attn(x)
            x = upsample(x)

        x1 = torch.cat((x, r), dim=1)
        x1 = self.final_res_block(x1, t_emb)
        x1 = self.final_conv(x1)

        if self.precondition:
            x1 = c_skip1 * x_clone + c_out1 * x1
            x2 = (x_clone - (t - 1) * x1) / t.sqrt()
        else:
            x2 = (x_clone - (t - 1) * x1) / t.sqrt()
        return x1, x2


if __name__ == "__main__":
    # resnet = resnet101(weights=ResNet101_Weights)
    # effnet = efficientnet_b7(weights=EfficientNet_B7_Weights)
    # effnet = efficientnet_b7(weights=None)
    # x = torch.rand(1, 3, 320, 320)
    # y = effnet(x)
    model = Unet(dim=128, dim_mults=(1, 2, 4, 4),
                 cond_dim=128,
                 cond_dim_mults=(2, 4, ),
                 channels=1,
                 window_sizes1=[[20, 20], [20, 20], [10, 10], [10, 10]],
                 window_sizes2=[[20, 20], [20, 20], [10, 10], [10, 10]],
                 cfg=fvcore.common.config.CfgNode({'cond_pe': False,
                      'cond_feature_size': (32, 128), 'cond_net': 'ViT-B/32',
                      'num_pos_feats': 96})
                 )
    x = torch.rand(1, 1, 80, 80)
    mask = torch.randint(49406, (1, 77))
    time = torch.tensor([0.5124])
    with torch.no_grad():
        y = model(x, time, mask)
    pass