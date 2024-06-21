#!/usr/bin/python3
import os
import sys

HOME_DIR = os.environ.get('HOME', '/root')
MVANET_SOURCE_DIR = HOME_DIR + '/GITHUB/qianyu-dlut/MVANet'
finetuned_MVANet_model_path = MVANET_SOURCE_DIR + '/model/Model_80.pth'
pretrained_SwinB_model_path = MVANET_SOURCE_DIR + '/model/swin_base_patch4_window12_384_22kto1k.pth'

import math
import numpy as np
import time
import cv2
import wget

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.autograd import Variable
from torch import nn
from torchvision import transforms

from einops import rearrange

from timm.models import load_checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

torch_device = 'cuda'
torch_dtype = torch.float16


def check_mkdir(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def SwinT(pretrained=True):
    model = SwinTransformer(embed_dim=96,
                            depths=[2, 2, 6, 2],
                            num_heads=[3, 6, 12, 24],
                            window_size=7)
    if pretrained is True:
        model.load_state_dict(torch.load(
            'data/backbone_ckpt/swin_tiny_patch4_window7_224.pth',
            map_location='cpu')['model'],
                              strict=False)

    return model


def SwinS(pretrained=True):
    model = SwinTransformer(embed_dim=96,
                            depths=[2, 2, 18, 2],
                            num_heads=[3, 6, 12, 24],
                            window_size=7)
    if pretrained is True:
        model.load_state_dict(torch.load(
            'data/backbone_ckpt/swin_small_patch4_window7_224.pth',
            map_location='cpu')['model'],
                              strict=False)

    return model


def SwinB(pretrained=True):
    model = SwinTransformer(embed_dim=128,
                            depths=[2, 2, 18, 2],
                            num_heads=[4, 8, 16, 32],
                            window_size=12)
    if pretrained is True:
        import os
        model.load_state_dict(torch.load(pretrained_SwinB_model_path,
                                         map_location='cpu')['model'],
                              strict=False)
    return model


def SwinL(pretrained=True):
    model = SwinTransformer(embed_dim=192,
                            depths=[2, 2, 18, 2],
                            num_heads=[6, 12, 24, 48],
                            window_size=12)
    if pretrained is True:
        model.load_state_dict(torch.load(
            'data/backbone_ckpt/swin_large_patch4_window12_384_22kto1k.pth',
            map_location='cpu')['model'],
                              strict=False)

    return model


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def make_cbr(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
                         nn.BatchNorm2d(out_dim), nn.PReLU())


def make_cbg(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
                         nn.BatchNorm2d(out_dim), nn.GELU())


def rescale_to(x, scale_factor: float = 2, interpolation='nearest'):
    return F.interpolate(x, scale_factor=scale_factor, mode=interpolation)


def resize_as(x, y, interpolation='bilinear'):
    return F.interpolate(x, size=y.shape[-2:], mode=interpolation)


def image2patches(x):
    """b c (hg h) (wg w) -> (hg wg b) c h w"""
    x = rearrange(x, 'b c (hg h) (wg w) -> (hg wg b) c h w', hg=2, wg=2)
    return x


def patches2image(x):
    """(hg wg b) c h w -> b c (hg h) (wg w)"""
    x = rearrange(x, '(hg wg b) c h w -> b c (hg h) (wg w)', hg=2, wg=2)
    return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size,
               C)
    windows = x.permute(0, 1, 3, 2, 4,
                        5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size,
                     window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def mkdir_safe(out_path):
    if type(out_path) == str:
        if len(out_path) > 0:
            if not os.path.exists(out_path):
                os.mkdir(out_path)


def get_model_path():
    import folder_paths
    from folder_paths import models_dir

    path_file_model = models_dir
    mkdir_safe(out_path=path_file_model)

    path_file_model = os.path.join(path_file_model, 'MVANet')
    mkdir_safe(out_path=path_file_model)

    path_file_model = os.path.join(path_file_model, 'Model_80.pth')

    return path_file_model


def download_model(path):
    if not os.path.exists(path):
        wget.download(
            'https://huggingface.co/aravindhv10/Self-Correction-Human-Parsing/resolve/main/checkpoints/Model_80.pth',
            out=path)


def load_model(model_checkpoint_path):

    download_model(path=model_checkpoint_path)

    torch.cuda.set_device(0)

    net = inf_MVANet().to(dtype=torch_dtype, device=torch_device)

    pretrained_dict = torch.load(model_checkpoint_path,
                                 map_location=torch_device)

    model_dict = net.state_dict()

    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items() if k in model_dict
    }

    model_dict.update(pretrained_dict)

    net.load_state_dict(model_dict)
    net = net.to(dtype=torch_dtype, device=torch_device)
    net.eval()

    return net


def do_infer_tensor2tensor(img, net):

    img_transform = transforms.Compose(
        [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    h_, w_ = img.shape[1], img.shape[2]

    with torch.no_grad():

        img = rearrange(img, 'B H W C -> B C H W')

        img_resize = torch.nn.functional.interpolate(input=img,
                                                     size=(1024, 1024),
                                                     mode='bicubic',
                                                     antialias=True)

        img_var = img_transform(img_resize)
        img_var = Variable(img_var)
        img_var = img_var.to(dtype=torch_dtype, device=torch_device)

        mask = []

        mask.append(net(img_var))

        prediction = torch.mean(torch.stack(mask, dim=0), dim=0)
        prediction = prediction.sigmoid()

        prediction = torch.nn.functional.interpolate(input=prediction,
                                                     size=(h_, w_),
                                                     mode='bicubic',
                                                     antialias=True)

        prediction = prediction.squeeze(0)
        prediction = prediction.clamp(0, 1)
        prediction = prediction.detach()
        prediction = prediction.to(dtype=torch.float32, device='cpu')

        return prediction


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:,
                                                                None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :,
                        0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        x = x.to(dtype=torch_dtype, device=torch_device)
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        attn = attn.to(dtype=torch_dtype, device=torch_device)
        v = v.to(dtype=torch_dtype, device=torch_device)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim,
                                    window_size=to_2tuple(self.window_size),
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attn_drop=attn_drop,
                                    proj_drop=drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x,
                                   shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size,
                                   C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp,
                                   Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x,
                           shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim,
                                 num_heads=num_heads,
                                 window_size=window_size,
                                 shift_size=0 if
                                 (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 qk_scale=qk_scale,
                                 drop=drop,
                                 attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(
                                     drop_path, list) else drop_path,
                                 norm_layer=norm_layer) for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size,
                          -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size,
                          -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                          float(-100.0)).masked_fill(
                                              attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x,
                      (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class SwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [
                pretrain_img_size[0] // patch_size[0],
                pretrain_img_size[1] // patch_size[1]
            ]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0],
                            patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if
                (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            load_checkpoint(self, pretrained, strict=False, logger=None)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed,
                                               size=(Wh, Ww),
                                               mode='bicubic')
            x = (x + absolute_pos_embed)  # B Wh*Ww C

        outs = [x.contiguous()]
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W,
                                 self.num_features[i]).permute(0, 3, 1,
                                                               2).contiguous()
                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()


class PositionEmbeddingSine:

    def __init__(self,
                 num_pos_feats=64,
                 temperature=10000,
                 normalize=False,
                 scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.dim_t = torch.arange(0,
                                  self.num_pos_feats,
                                  dtype=torch_dtype,
                                  device=torch_device)

    def __call__(self, b, h, w):
        mask = torch.zeros([b, h, w], dtype=torch.bool, device=torch_device)
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(dim=1, dtype=torch_dtype)
        x_embed = not_mask.cumsum(dim=2, dtype=torch_dtype)
        if self.normalize:
            eps = 1e-6
            y_embed = ((y_embed - 0.5) / (y_embed[:, -1:, :] + eps) *
                       self.scale).to(device=torch_device, dtype=torch_dtype)
            x_embed = ((x_embed - 0.5) / (x_embed[:, :, -1:] + eps) *
                       self.scale).to(device=torch_device, dtype=torch_dtype)

        dim_t = self.temperature**(2 * (self.dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class MCLM(nn.Module):

    def __init__(self, d_model, num_heads, pool_ratios=[1, 4, 8]):
        super(MCLM, self).__init__()
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1)
        ])

        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.linear2 = nn.Linear(d_model * 2, d_model)
        self.linear3 = nn.Linear(d_model, d_model * 2)
        self.linear4 = nn.Linear(d_model * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.activation = get_activation_fn('relu')
        self.pool_ratios = pool_ratios
        self.p_poses = []
        self.g_pos = None
        self.positional_encoding = PositionEmbeddingSine(
            num_pos_feats=d_model // 2, normalize=True)

    def forward(self, l, g):
        """
        l: 4,c,h,w
        g: 1,c,h,w
        """
        b, c, h, w = l.size()
        # 4,c,h,w -> 1,c,2h,2w
        concated_locs = rearrange(l,
                                  '(hg wg b) c h w -> b c (hg h) (wg w)',
                                  hg=2,
                                  wg=2)

        pools = []
        for pool_ratio in self.pool_ratios:
            # b,c,h,w
            tgt_hw = (round(h / pool_ratio), round(w / pool_ratio))
            pool = F.adaptive_avg_pool2d(concated_locs, tgt_hw)
            pools.append(rearrange(pool, 'b c h w -> (h w) b c'))
            if self.g_pos is None:
                pos_emb = self.positional_encoding(pool.shape[0],
                                                   pool.shape[2],
                                                   pool.shape[3])
                pos_emb = rearrange(pos_emb, 'b c h w -> (h w) b c')
                self.p_poses.append(pos_emb)
        pools = torch.cat(pools, 0)
        if self.g_pos is None:
            self.p_poses = torch.cat(self.p_poses, dim=0)
            pos_emb = self.positional_encoding(g.shape[0], g.shape[2],
                                               g.shape[3])
            self.g_pos = rearrange(pos_emb, 'b c h w -> (h w) b c')

        # attention between glb (q) & multisensory concated-locs (k,v)
        g_hw_b_c = rearrange(g, 'b c h w -> (h w) b c')
        g_hw_b_c = g_hw_b_c + self.dropout1(self.attention[0](
            g_hw_b_c + self.g_pos, pools + self.p_poses, pools)[0])
        g_hw_b_c = self.norm1(g_hw_b_c)
        g_hw_b_c = g_hw_b_c + self.dropout2(
            self.linear2(
                self.dropout(self.activation(self.linear1(g_hw_b_c)).clone())))
        g_hw_b_c = self.norm2(g_hw_b_c)

        # attention between origin locs (q) & freashed glb (k,v)
        l_hw_b_c = rearrange(l, "b c h w -> (h w) b c")
        _g_hw_b_c = rearrange(g_hw_b_c, '(h w) b c -> h w b c', h=h, w=w)
        _g_hw_b_c = rearrange(_g_hw_b_c,
                              "(ng h) (nw w) b c -> (h w) (ng nw b) c",
                              ng=2,
                              nw=2)
        outputs_re = []
        for i, (_l, _g) in enumerate(
                zip(l_hw_b_c.chunk(4, dim=1), _g_hw_b_c.chunk(4, dim=1))):
            outputs_re.append(self.attention[i + 1](_l, _g,
                                                    _g)[0])  # (h w) 1 c
        outputs_re = torch.cat(outputs_re, 1)  # (h w) 4 c

        l_hw_b_c = l_hw_b_c + self.dropout1(outputs_re)
        l_hw_b_c = self.norm1(l_hw_b_c)
        l_hw_b_c = l_hw_b_c + self.dropout2(
            self.linear4(
                self.dropout(self.activation(self.linear3(l_hw_b_c)).clone())))
        l_hw_b_c = self.norm2(l_hw_b_c)

        l = torch.cat((l_hw_b_c, g_hw_b_c), 1)  # hw,b(5),c
        return rearrange(l, "(h w) b c -> b c h w", h=h, w=w)  ## (5,c,h*w)


class inf_MCLM(nn.Module):

    def __init__(self, d_model, num_heads, pool_ratios=[1, 4, 8]):
        super(inf_MCLM, self).__init__()
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1)
        ])

        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.linear2 = nn.Linear(d_model * 2, d_model)
        self.linear3 = nn.Linear(d_model, d_model * 2)
        self.linear4 = nn.Linear(d_model * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.activation = get_activation_fn('relu')
        self.pool_ratios = pool_ratios
        self.p_poses = []
        self.g_pos = None
        self.positional_encoding = PositionEmbeddingSine(
            num_pos_feats=d_model // 2, normalize=True)

    def forward(self, l, g):
        """
        l: 4,c,h,w
        g: 1,c,h,w
        """
        b, c, h, w = l.size()
        # 4,c,h,w -> 1,c,2h,2w
        concated_locs = rearrange(l,
                                  '(hg wg b) c h w -> b c (hg h) (wg w)',
                                  hg=2,
                                  wg=2)
        self.p_poses = []
        pools = []
        for pool_ratio in self.pool_ratios:
            # b,c,h,w
            tgt_hw = (round(h / pool_ratio), round(w / pool_ratio))
            pool = F.adaptive_avg_pool2d(concated_locs, tgt_hw)
            pools.append(rearrange(pool, 'b c h w -> (h w) b c'))
            # if self.g_pos is None:
            pos_emb = self.positional_encoding(pool.shape[0], pool.shape[2],
                                               pool.shape[3])
            pos_emb = rearrange(pos_emb, 'b c h w -> (h w) b c')
            self.p_poses.append(pos_emb)
        pools = torch.cat(pools, 0)
        # if self.g_pos is None:
        self.p_poses = torch.cat(self.p_poses, dim=0)
        pos_emb = self.positional_encoding(g.shape[0], g.shape[2], g.shape[3])
        self.g_pos = rearrange(pos_emb, 'b c h w -> (h w) b c')

        # attention between glb (q) & multisensory concated-locs (k,v)
        g_hw_b_c = rearrange(g, 'b c h w -> (h w) b c')
        g_hw_b_c = g_hw_b_c + self.dropout1(self.attention[0](
            g_hw_b_c + self.g_pos, pools + self.p_poses, pools)[0])
        g_hw_b_c = self.norm1(g_hw_b_c)
        g_hw_b_c = g_hw_b_c + self.dropout2(
            self.linear2(
                self.dropout(self.activation(self.linear1(g_hw_b_c)).clone())))
        g_hw_b_c = self.norm2(g_hw_b_c)

        # attention between origin locs (q) & freashed glb (k,v)
        l_hw_b_c = rearrange(l, "b c h w -> (h w) b c")
        _g_hw_b_c = rearrange(g_hw_b_c, '(h w) b c -> h w b c', h=h, w=w)
        _g_hw_b_c = rearrange(_g_hw_b_c,
                              "(ng h) (nw w) b c -> (h w) (ng nw b) c",
                              ng=2,
                              nw=2)
        outputs_re = []
        for i, (_l, _g) in enumerate(
                zip(l_hw_b_c.chunk(4, dim=1), _g_hw_b_c.chunk(4, dim=1))):
            outputs_re.append(self.attention[i + 1](_l, _g,
                                                    _g)[0])  # (h w) 1 c
        outputs_re = torch.cat(outputs_re, 1)  # (h w) 4 c

        l_hw_b_c = l_hw_b_c + self.dropout1(outputs_re)
        l_hw_b_c = self.norm1(l_hw_b_c)
        l_hw_b_c = l_hw_b_c + self.dropout2(
            self.linear4(
                self.dropout(self.activation(self.linear3(l_hw_b_c)).clone())))
        l_hw_b_c = self.norm2(l_hw_b_c)

        l = torch.cat((l_hw_b_c, g_hw_b_c), 1)  # hw,b(5),c
        return rearrange(l, "(h w) b c -> b c h w", h=h, w=w)  ## (5,c,h*w)


class MCRM(nn.Module):

    def __init__(self, d_model, num_heads, pool_ratios=[4, 8, 16], h=None):
        super(MCRM, self).__init__()
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1)
        ])

        self.linear3 = nn.Linear(d_model, d_model * 2)
        self.linear4 = nn.Linear(d_model * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.activation = get_activation_fn('relu')
        self.sal_conv = nn.Conv2d(d_model, 1, 1)
        self.pool_ratios = pool_ratios
        self.positional_encoding = PositionEmbeddingSine(
            num_pos_feats=d_model // 2, normalize=True)

    def forward(self, x):
        b, c, h, w = x.size()
        loc, glb = x.split([4, 1], dim=0)  # 4,c,h,w; 1,c,h,w
        # b(4),c,h,w
        patched_glb = rearrange(glb,
                                'b c (hg h) (wg w) -> (hg wg b) c h w',
                                hg=2,
                                wg=2)

        # generate token attention map
        token_attention_map = self.sigmoid(self.sal_conv(glb))
        token_attention_map = F.interpolate(token_attention_map,
                                            size=patches2image(loc).shape[-2:],
                                            mode='nearest')
        loc = loc * rearrange(token_attention_map,
                              'b c (hg h) (wg w) -> (hg wg b) c h w',
                              hg=2,
                              wg=2)
        pools = []
        for pool_ratio in self.pool_ratios:
            tgt_hw = (round(h / pool_ratio), round(w / pool_ratio))
            pool = F.adaptive_avg_pool2d(patched_glb, tgt_hw)
            pools.append(rearrange(pool,
                                   'nl c h w -> nl c (h w)'))  # nl(4),c,hw
        # nl(4),c,nphw -> nl(4),nphw,1,c
        pools = rearrange(torch.cat(pools, 2), "nl c nphw -> nl nphw 1 c")
        loc_ = rearrange(loc, 'nl c h w -> nl (h w) 1 c')
        outputs = []
        for i, q in enumerate(
                loc_.unbind(dim=0)):  # traverse all local patches
            # np*hw,1,c
            v = pools[i]
            k = v
            outputs.append(self.attention[i](q, k, v)[0])
        outputs = torch.cat(outputs, 1)
        src = loc.view(4, c, -1).permute(2, 0, 1) + self.dropout1(outputs)
        src = self.norm1(src)
        src = src + self.dropout2(
            self.linear4(
                self.dropout(self.activation(self.linear3(src)).clone())))
        src = self.norm2(src)

        src = src.permute(1, 2, 0).reshape(4, c, h, w)  # freshed loc
        glb = glb + F.interpolate(patches2image(src),
                                  size=glb.shape[-2:],
                                  mode='nearest')  # freshed glb
        return torch.cat((src, glb), 0), token_attention_map


class inf_MCRM(nn.Module):

    def __init__(self, d_model, num_heads, pool_ratios=[4, 8, 16], h=None):
        super(inf_MCRM, self).__init__()
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1)
        ])

        self.linear3 = nn.Linear(d_model, d_model * 2)
        self.linear4 = nn.Linear(d_model * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.activation = get_activation_fn('relu')
        self.sal_conv = nn.Conv2d(d_model, 1, 1)
        self.pool_ratios = pool_ratios
        self.positional_encoding = PositionEmbeddingSine(
            num_pos_feats=d_model // 2, normalize=True)

    def forward(self, x):
        b, c, h, w = x.size()
        loc, glb = x.split([4, 1], dim=0)  # 4,c,h,w; 1,c,h,w
        # b(4),c,h,w
        patched_glb = rearrange(glb,
                                'b c (hg h) (wg w) -> (hg wg b) c h w',
                                hg=2,
                                wg=2)

        # generate token attention map
        token_attention_map = self.sigmoid(self.sal_conv(glb))
        token_attention_map = F.interpolate(token_attention_map,
                                            size=patches2image(loc).shape[-2:],
                                            mode='nearest')
        loc = loc * rearrange(token_attention_map,
                              'b c (hg h) (wg w) -> (hg wg b) c h w',
                              hg=2,
                              wg=2)
        pools = []
        for pool_ratio in self.pool_ratios:
            tgt_hw = (round(h / pool_ratio), round(w / pool_ratio))
            pool = F.adaptive_avg_pool2d(patched_glb, tgt_hw)
            pools.append(rearrange(pool,
                                   'nl c h w -> nl c (h w)'))  # nl(4),c,hw
        # nl(4),c,nphw -> nl(4),nphw,1,c
        pools = rearrange(torch.cat(pools, 2), "nl c nphw -> nl nphw 1 c")
        loc_ = rearrange(loc, 'nl c h w -> nl (h w) 1 c')
        outputs = []
        for i, q in enumerate(
                loc_.unbind(dim=0)):  # traverse all local patches
            # np*hw,1,c
            v = pools[i]
            k = v
            outputs.append(self.attention[i](q, k, v)[0])
        outputs = torch.cat(outputs, 1)
        src = loc.view(4, c, -1).permute(2, 0, 1) + self.dropout1(outputs)
        src = self.norm1(src)
        src = src + self.dropout2(
            self.linear4(
                self.dropout(self.activation(self.linear3(src)).clone())))
        src = self.norm2(src)

        src = src.permute(1, 2, 0).reshape(4, c, h, w)  # freshed loc
        glb = glb + F.interpolate(patches2image(src),
                                  size=glb.shape[-2:],
                                  mode='nearest')  # freshed glb
        return torch.cat((src, glb), 0)


# model for single-scale training
class MVANet(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = SwinB(pretrained=True)
        emb_dim = 128
        self.sideout5 = nn.Sequential(
            nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.sideout4 = nn.Sequential(
            nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.sideout3 = nn.Sequential(
            nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.sideout2 = nn.Sequential(
            nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.sideout1 = nn.Sequential(
            nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))

        self.output5 = make_cbr(1024, emb_dim)
        self.output4 = make_cbr(512, emb_dim)
        self.output3 = make_cbr(256, emb_dim)
        self.output2 = make_cbr(128, emb_dim)
        self.output1 = make_cbr(128, emb_dim)

        self.multifieldcrossatt = MCLM(emb_dim, 1, [1, 4, 8])
        self.conv1 = make_cbr(emb_dim, emb_dim)
        self.conv2 = make_cbr(emb_dim, emb_dim)
        self.conv3 = make_cbr(emb_dim, emb_dim)
        self.conv4 = make_cbr(emb_dim, emb_dim)
        self.dec_blk1 = MCRM(emb_dim, 1, [2, 4, 8])
        self.dec_blk2 = MCRM(emb_dim, 1, [2, 4, 8])
        self.dec_blk3 = MCRM(emb_dim, 1, [2, 4, 8])
        self.dec_blk4 = MCRM(emb_dim, 1, [2, 4, 8])

        self.insmask_head = nn.Sequential(
            nn.Conv2d(emb_dim, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384), nn.PReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.BatchNorm2d(384),
            nn.PReLU(), nn.Conv2d(384, emb_dim, kernel_size=3, padding=1))

        self.shallow = nn.Sequential(
            nn.Conv2d(3, emb_dim, kernel_size=3, padding=1))
        self.upsample1 = make_cbg(emb_dim, emb_dim)
        self.upsample2 = make_cbg(emb_dim, emb_dim)
        self.output = nn.Sequential(
            nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))

        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, x):
        x = x.to(dtype=torch_dtype, device=torch_device)
        shallow = self.shallow(x)
        glb = rescale_to(x, scale_factor=0.5, interpolation='bilinear')
        loc = image2patches(x)
        input = torch.cat((loc, glb), dim=0)
        feature = self.backbone(input)
        e5 = self.output5(feature[4])  # (5,128,16,16)
        e4 = self.output4(feature[3])  # (5,128,32,32)
        e3 = self.output3(feature[2])  # (5,128,64,64)
        e2 = self.output2(feature[1])  # (5,128,128,128)
        e1 = self.output1(feature[0])  # (5,128,128,128)
        loc_e5, glb_e5 = e5.split([4, 1], dim=0)
        e5 = self.multifieldcrossatt(loc_e5, glb_e5)  # (4,128,16,16)

        e4, tokenattmap4 = self.dec_blk4(e4 + resize_as(e5, e4))
        e4 = self.conv4(e4)
        e3, tokenattmap3 = self.dec_blk3(e3 + resize_as(e4, e3))
        e3 = self.conv3(e3)
        e2, tokenattmap2 = self.dec_blk2(e2 + resize_as(e3, e2))
        e2 = self.conv2(e2)
        e1, tokenattmap1 = self.dec_blk1(e1 + resize_as(e2, e1))
        e1 = self.conv1(e1)
        loc_e1, glb_e1 = e1.split([4, 1], dim=0)
        output1_cat = patches2image(loc_e1)  # (1,128,256,256)
        # add glb feat in
        output1_cat = output1_cat + resize_as(glb_e1, output1_cat)
        # merge
        final_output = self.insmask_head(output1_cat)  # (1,128,256,256)
        # shallow feature merge
        final_output = final_output + resize_as(shallow, final_output)
        final_output = self.upsample1(rescale_to(final_output))
        final_output = rescale_to(final_output +
                                  resize_as(shallow, final_output))
        final_output = self.upsample2(final_output)
        final_output = self.output(final_output)
        ####
        sideout5 = self.sideout5(e5).to(dtype=torch_dtype, device=torch_device)
        sideout4 = self.sideout4(e4)
        sideout3 = self.sideout3(e3)
        sideout2 = self.sideout2(e2)
        sideout1 = self.sideout1(e1)
        #######glb_sideouts ######
        glb5 = self.sideout5(glb_e5)
        glb4 = sideout4[-1, :, :, :].unsqueeze(0)
        glb3 = sideout3[-1, :, :, :].unsqueeze(0)
        glb2 = sideout2[-1, :, :, :].unsqueeze(0)
        glb1 = sideout1[-1, :, :, :].unsqueeze(0)
        ####### concat 4 to 1 #######
        sideout1 = patches2image(sideout1[:-1]).to(dtype=torch_dtype,
                                                   device=torch_device)
        sideout2 = patches2image(sideout2[:-1]).to(
            dtype=torch_dtype,
            device=torch_device)  ####(5,c,h,w) -> (1 c 2h,2w)
        sideout3 = patches2image(sideout3[:-1]).to(dtype=torch_dtype,
                                                   device=torch_device)
        sideout4 = patches2image(sideout4[:-1]).to(dtype=torch_dtype,
                                                   device=torch_device)
        sideout5 = patches2image(sideout5[:-1]).to(dtype=torch_dtype,
                                                   device=torch_device)
        if self.training:
            return sideout5, sideout4, sideout3, sideout2, sideout1, final_output, glb5, glb4, glb3, glb2, glb1, tokenattmap4, tokenattmap3, tokenattmap2, tokenattmap1
        else:
            return final_output


# model for multi-scale testing
class inf_MVANet(nn.Module):

    def __init__(self):
        super().__init__()
        # self.backbone = SwinB(pretrained=True)
        self.backbone = SwinB(pretrained=False)

        emb_dim = 128
        self.output5 = make_cbr(1024, emb_dim)
        self.output4 = make_cbr(512, emb_dim)
        self.output3 = make_cbr(256, emb_dim)
        self.output2 = make_cbr(128, emb_dim)
        self.output1 = make_cbr(128, emb_dim)

        self.multifieldcrossatt = inf_MCLM(emb_dim, 1, [1, 4, 8])
        self.conv1 = make_cbr(emb_dim, emb_dim)
        self.conv2 = make_cbr(emb_dim, emb_dim)
        self.conv3 = make_cbr(emb_dim, emb_dim)
        self.conv4 = make_cbr(emb_dim, emb_dim)
        self.dec_blk1 = inf_MCRM(emb_dim, 1, [2, 4, 8])
        self.dec_blk2 = inf_MCRM(emb_dim, 1, [2, 4, 8])
        self.dec_blk3 = inf_MCRM(emb_dim, 1, [2, 4, 8])
        self.dec_blk4 = inf_MCRM(emb_dim, 1, [2, 4, 8])

        self.insmask_head = nn.Sequential(
            nn.Conv2d(emb_dim, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384), nn.PReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.BatchNorm2d(384),
            nn.PReLU(), nn.Conv2d(384, emb_dim, kernel_size=3, padding=1))

        self.shallow = nn.Sequential(
            nn.Conv2d(3, emb_dim, kernel_size=3, padding=1))
        self.upsample1 = make_cbg(emb_dim, emb_dim)
        self.upsample2 = make_cbg(emb_dim, emb_dim)
        self.output = nn.Sequential(
            nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))

        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, x):
        shallow = self.shallow(x)
        glb = rescale_to(x, scale_factor=0.5, interpolation='bilinear')
        loc = image2patches(x)
        input = torch.cat((loc, glb), dim=0)
        feature = self.backbone(input)
        e5 = self.output5(feature[4])
        e4 = self.output4(feature[3])
        e3 = self.output3(feature[2])
        e2 = self.output2(feature[1])
        e1 = self.output1(feature[0])
        loc_e5, glb_e5 = e5.split([4, 1], dim=0)
        e5_cat = self.multifieldcrossatt(loc_e5, glb_e5)

        e4 = self.conv4(self.dec_blk4(e4 + resize_as(e5_cat, e4)))
        e3 = self.conv3(self.dec_blk3(e3 + resize_as(e4, e3)))
        e2 = self.conv2(self.dec_blk2(e2 + resize_as(e3, e2)))
        e1 = self.conv1(self.dec_blk1(e1 + resize_as(e2, e1)))
        loc_e1, glb_e1 = e1.split([4, 1], dim=0)
        # after decoder, concat loc features to a whole one, and merge
        output1_cat = patches2image(loc_e1)
        # add glb feat in
        output1_cat = output1_cat + resize_as(glb_e1, output1_cat)
        # merge
        final_output = self.insmask_head(output1_cat)
        # shallow feature merge
        final_output = final_output + resize_as(shallow, final_output)
        final_output = self.upsample1(rescale_to(final_output))
        final_output = rescale_to(final_output +
                                  resize_as(shallow, final_output))
        final_output = self.upsample2(final_output)
        final_output = self.output(final_output)
        return final_output


class load_MVANet_Model:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
        }

    RETURN_TYPES = ("MVANet_Model", )
    FUNCTION = "test"
    CATEGORY = "MVANet"

    def test(self):
        return (load_model(get_model_path()), )


class run_MVANet_inference:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "MVANet_Model": ("MVANet_Model", ),
            },
        }

    RETURN_TYPES = ("MASK", )
    FUNCTION = "test"
    CATEGORY = "MVANet"

    def test(
        self,
        image,
        MVANet_Model,
    ):
        ret = do_infer_tensor2tensor(img=image, net=MVANet_Model)

        return (ret, )


# NODE_CLASS_MAPPINGS = {
#     "load_MVANet_Model": load_MVANet_Model,
#     "run_MVANet_inference": run_MVANet_inference
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "load_MVANet_Model": "load MVANet Model",
#     "run_MVANet_inference": "run MVANet inference"
# }
