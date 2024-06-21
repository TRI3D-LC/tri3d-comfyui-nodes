import cv2
import math
import numpy as np
import os
import random
import wget

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from collections import OrderedDict
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import folder_paths
from folder_paths import models_dir


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

    path_file_model = os.path.join(path_file_model, 'AEMatter')
    mkdir_safe(out_path=path_file_model)

    path_file_model = os.path.join(path_file_model, 'AEM_RWA.ckpt')

    return path_file_model


def download_model(path):
    if not os.path.exists(path):
        wget.download(
            'https://huggingface.co/aravindhv10/Self-Correction-Human-Parsing/resolve/main/checkpoints/AEMatter/AEM_RWA.ckpt?download=true',
            out=path)


def from_torch_image(image):
    image = image.cpu().numpy() * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def to_torch_image(image):
    image = image.astype(dtype=np.float32)
    image /= 255.0
    image = torch.from_numpy(image)
    return image


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


def get_AEMatter_model(path_model_checkpoint):

    download_model(path=path_model_checkpoint)

    matmodel = AEMatter()
    matmodel.load_state_dict(
        torch.load(path_model_checkpoint, map_location='cpu')['model'])

    matmodel = matmodel.cuda()
    matmodel.eval()

    return matmodel


def do_infer(rawimg, trimap, matmodel):
    trimap_nonp = trimap.copy()
    h, w, c = rawimg.shape
    nonph, nonpw, _ = rawimg.shape
    newh = (((h - 1) // 32) + 1) * 32
    neww = (((w - 1) // 32) + 1) * 32
    padh = newh - h
    padh1 = int(padh / 2)
    padh2 = padh - padh1
    padw = neww - w
    padw1 = int(padw / 2)
    padw2 = padw - padw1

    rawimg_pad = cv2.copyMakeBorder(rawimg, padh1, padh2, padw1, padw2,
                                    cv2.BORDER_REFLECT)

    trimap_pad = cv2.copyMakeBorder(trimap, padh1, padh2, padw1, padw2,
                                    cv2.BORDER_REFLECT)

    h_pad, w_pad, _ = rawimg_pad.shape
    tritemp = np.zeros([*trimap_pad.shape, 3], np.float32)
    tritemp[:, :, 0] = (trimap_pad == 0)
    tritemp[:, :, 1] = (trimap_pad == 128)
    tritemp[:, :, 2] = (trimap_pad == 255)
    tritempimgs = np.transpose(tritemp, (2, 0, 1))
    tritempimgs = tritempimgs[np.newaxis, :, :, :]
    img = np.transpose(rawimg_pad, (2, 0, 1))[np.newaxis, ::-1, :, :]
    img = np.array(img, np.float32)
    img = img / 255.
    img = torch.from_numpy(img).cuda()
    tritempimgs = torch.from_numpy(tritempimgs).cuda()
    with torch.no_grad():
        pred = matmodel(img, tritempimgs)
        pred = pred.detach().cpu().numpy()[0]
        pred = pred[:, padh1:padh1 + h, padw1:padw1 + w]
        preda = pred[
            0:1,
        ] * 255
        preda = np.transpose(preda, (1, 2, 0))
        preda = preda * (trimap_nonp[:, :, None]
                         == 128) + (trimap_nonp[:, :, None] == 255) * 255
    preda = np.array(preda, np.uint8)
    return preda


def main():
    ptrimap = '/home/asd/Desktop/demo/retriever_trimap.png'
    pimgs = '/home/asd/Desktop/demo/retriever_rgb.png'
    p_outs = 'alpha.png'

    matmodel = get_AEMatter_model(
        path_model_checkpoint='/home/asd/Desktop/AEM_RWA.ckpt')

    # matmodel = AEMatter()
    # matmodel.load_state_dict(
    #     torch.load('/home/asd/Desktop/AEM_RWA.ckpt',
    #                map_location='cpu')['model'])

    # matmodel = matmodel.cuda()
    # matmodel.eval()

    rawimg = pimgs
    trimap = ptrimap
    rawimg = cv2.imread(rawimg, cv2.IMREAD_COLOR)
    trimap = cv2.imread(trimap, cv2.IMREAD_GRAYSCALE)
    trimap_nonp = trimap.copy()
    h, w, c = rawimg.shape
    nonph, nonpw, _ = rawimg.shape
    newh = (((h - 1) // 32) + 1) * 32
    neww = (((w - 1) // 32) + 1) * 32
    padh = newh - h
    padh1 = int(padh / 2)
    padh2 = padh - padh1
    padw = neww - w
    padw1 = int(padw / 2)
    padw2 = padw - padw1
    rawimg_pad = cv2.copyMakeBorder(rawimg, padh1, padh2, padw1, padw2,
                                    cv2.BORDER_REFLECT)
    trimap_pad = cv2.copyMakeBorder(trimap, padh1, padh2, padw1, padw2,
                                    cv2.BORDER_REFLECT)
    h_pad, w_pad, _ = rawimg_pad.shape
    tritemp = np.zeros([*trimap_pad.shape, 3], np.float32)
    tritemp[:, :, 0] = (trimap_pad == 0)
    tritemp[:, :, 1] = (trimap_pad == 128)
    tritemp[:, :, 2] = (trimap_pad == 255)
    tritempimgs = np.transpose(tritemp, (2, 0, 1))
    tritempimgs = tritempimgs[np.newaxis, :, :, :]
    img = np.transpose(rawimg_pad, (2, 0, 1))[np.newaxis, ::-1, :, :]
    img = np.array(img, np.float32)
    img = img / 255.
    img = torch.from_numpy(img).cuda()
    tritempimgs = torch.from_numpy(tritempimgs).cuda()
    with torch.no_grad():
        pred = matmodel(img, tritempimgs)
        pred = pred.detach().cpu().numpy()[0]
        pred = pred[:, padh1:padh1 + h, padw1:padw1 + w]
        preda = pred[
            0:1,
        ] * 255
        preda = np.transpose(preda, (1, 2, 0))
        preda = preda * (trimap_nonp[:, :, None]
                         == 128) + (trimap_nonp[:, :, None] == 255) * 255
    preda = np.array(preda, np.uint8)
    cv2.imwrite(p_outs, preda)


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
        # print(x.shape,H,W)
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
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(
            2)  # nW, ww window_size*window_size
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

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed,
                                               size=(Wh, Ww),
                                               mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1,
                                                              2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
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


class ResBlock(nn.Module):

    def __init__(self, inc, midc):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inc,
                               midc,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=True)
        self.gn1 = nn.GroupNorm(16, midc)
        self.conv2 = nn.Conv2d(midc,
                               midc,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.gn2 = nn.GroupNorm(16, midc)
        self.conv3 = nn.Conv2d(midc,
                               inc,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=True)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x_ = x
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = x + x_
        x = self.relu(x)
        return x


class AEALblock(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=512,
                 dropout=0.0,
                 layer_norm_eps=1e-5,
                 batch_first=True,
                 norm_first=False,
                 width=5):
        super(AEALblock, self).__init__()
        self.self_attn2 = nn.MultiheadAttention(d_model // 2,
                                                nhead // 2,
                                                dropout=dropout,
                                                batch_first=batch_first)
        self.self_attn1 = nn.MultiheadAttention(d_model // 2,
                                                nhead // 2,
                                                dropout=dropout,
                                                batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.width = width
        self.trans = nn.Sequential(
            nn.Conv2d(d_model + 512, d_model // 2, 1, 1, 0),
            ResBlock(d_model // 2, d_model // 4),
            nn.Conv2d(d_model // 2, d_model, 1, 1, 0))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        src,
        feats,
    ):
        src = self.gamma * self.trans(torch.cat([src, feats], 1)) + src
        b, c, h, w = src.shape
        x1 = src[:, 0:c // 2]
        x1_ = rearrange(x1, 'b c (h1 h2) w -> b c h1 h2 w', h2=self.width)
        x1_ = rearrange(x1_, 'b c h1 h2 w -> (b h1) (h2 w) c')
        x2 = src[:, c // 2:]
        x2_ = rearrange(x2, 'b c h (w1 w2) -> b c h w1 w2', w2=self.width)
        x2_ = rearrange(x2_, 'b c h w1 w2 -> (b w1) (h w2) c')
        x = rearrange(src, 'b c h w-> b (h w) c')
        x = self.norm1(x + self._sa_block(x1_, x2_, h, w))
        x = self.norm2(x + self._ff_block(x))
        x = rearrange(x, 'b (h w) c->b c h w', h=h, w=w)
        return x

    def _sa_block(self, x1, x2, h, w):
        x1 = self.self_attn1(x1,
                             x1,
                             x1,
                             attn_mask=None,
                             key_padding_mask=None,
                             need_weights=False)[0]

        x2 = self.self_attn2(x2,
                             x2,
                             x2,
                             attn_mask=None,
                             key_padding_mask=None,
                             need_weights=False)[0]

        x1 = rearrange(x1,
                       '(b h1) (h2 w) c-> b (h1 h2 w) c',
                       h2=self.width,
                       h1=h // self.width)
        x2 = rearrange(x2,
                       ' (b w1) (h w2) c-> b (h w1 w2) c',
                       w2=self.width,
                       w1=w // self.width)
        x = torch.cat([x1, x2], dim=2)
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class AEMatter(nn.Module):

    def __init__(self):
        super(AEMatter, self).__init__()
        trans = SwinTransformer(pretrain_img_size=224,
                                embed_dim=96,
                                depths=[2, 2, 6, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=7,
                                ape=False,
                                drop_path_rate=0.2,
                                patch_norm=True,
                                use_checkpoint=False)

        # trans.load_state_dict(torch.load(
        #     '/home/asd/Desktop/swin_tiny_patch4_window7_224.pth',
        #     map_location="cpu")["model"],
        #                       strict=False)

        trans.patch_embed.proj = nn.Conv2d(64, 96, 3, 2, 1)

        self.start_conv0 = nn.Sequential(nn.Conv2d(6, 48, 3, 1, 1),
                                         nn.PReLU(48))

        self.start_conv = nn.Sequential(nn.Conv2d(48, 64, 3, 2,
                                                  1), nn.PReLU(64),
                                        nn.Conv2d(64, 64, 3, 1, 1),
                                        nn.PReLU(64))

        self.trans = trans
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=640 + 768,
                      out_channels=256,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256 + 384,
                      out_channels=256,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True), )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256 + 192,
                      out_channels=192,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True), )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=192 + 96,
                      out_channels=128,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True), )
        self.ctran0 = BasicLayer(256, 3, 8, 7, drop_path=0.09)
        self.ctran1 = BasicLayer(256, 3, 8, 7, drop_path=0.07)
        self.ctran2 = BasicLayer(192, 3, 6, 7, drop_path=0.05)
        self.ctran3 = BasicLayer(128, 3, 4, 7, drop_path=0.03)
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=192,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.PReLU(64),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.PReLU(64),
            nn.Conv2d(in_channels=64,
                      out_channels=48,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.PReLU(48))
        self.convo = nn.Sequential(
            nn.Conv2d(in_channels=48 + 48 + 6,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.PReLU(32),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.PReLU(32),
            nn.Conv2d(in_channels=32,
                      out_channels=1,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True))
        self.up = nn.Upsample(scale_factor=2,
                              mode='bilinear',
                              align_corners=False)
        self.upn = nn.Upsample(scale_factor=2, mode='nearest')
        self.apptrans = nn.Sequential(
            nn.Conv2d(256 + 384, 256, 1, 1, bias=True), ResBlock(256, 128),
            ResBlock(256, 128), nn.Conv2d(256, 512, 2, 2, bias=True),
            ResBlock(512, 128))
        self.emb = nn.Sequential(nn.Conv2d(768, 640, 1, 1, 0),
                                 ResBlock(640, 160))
        self.embdp = nn.Sequential(nn.Conv2d(640, 640, 1, 1, 0))
        self.h2l = nn.Conv2d(768, 256, 1, 1, 0)
        self.width = 5
        self.trans1 = AEALblock(d_model=640,
                                nhead=20,
                                dim_feedforward=2048,
                                dropout=0.2,
                                width=self.width)
        self.trans2 = AEALblock(d_model=640,
                                nhead=20,
                                dim_feedforward=2048,
                                dropout=0.2,
                                width=self.width)
        self.trans3 = AEALblock(d_model=640,
                                nhead=20,
                                dim_feedforward=2048,
                                dropout=0.2,
                                width=self.width)

    def aeal(self, x, sem):
        xe = self.emb(x)
        x_ = xe
        x_ = self.embdp(x_)
        b, c, h1, w1 = x_.shape
        bnew_ph = int(np.ceil(h1 / self.width) * self.width) - h1
        bnew_pw = int(np.ceil(w1 / self.width) * self.width) - w1
        newph1 = bnew_ph // 2
        newph2 = bnew_ph - newph1
        newpw1 = bnew_pw // 2
        newpw2 = bnew_pw - newpw1
        x_ = F.pad(x_, (newpw1, newpw2, newph1, newph2))
        sem = F.pad(sem, (newpw1, newpw2, newph1, newph2))
        x_ = self.trans1(x_, sem)
        x_ = self.trans2(x_, sem)
        x_ = self.trans3(x_, sem)
        x_ = x_[:, :, newph1:h1 + newph1, newpw1:w1 + newpw1]
        return x_

    def forward(self, x, y):
        inputs = torch.cat((x, y), 1)
        x = self.start_conv0(inputs)
        x_ = self.start_conv(x)
        x1, x2, x3, x4 = self.trans(x_)
        x4h = self.h2l(x4)
        x3s = self.apptrans(torch.cat([x3, self.upn(x4h)], 1))
        x4_ = self.aeal(x4, x3s)
        x4 = torch.cat((x4, x4_), 1)
        X4 = self.conv1(x4)
        wh, ww = X4.shape[2], X4.shape[3]
        X4 = rearrange(X4, 'b c h w -> b (h w) c')
        X4, _, _, _, _, _ = self.ctran0(X4, wh, ww)
        X4 = rearrange(X4, 'b (h w) c -> b c h w', h=wh, w=ww)
        X3 = self.up(X4)
        X3 = torch.cat((x3, X3), 1)
        X3 = self.conv2(X3)
        wh, ww = X3.shape[2], X3.shape[3]
        X3 = rearrange(X3, 'b c h w -> b (h w) c')
        X3, _, _, _, _, _ = self.ctran1(X3, wh, ww)
        X3 = rearrange(X3, 'b (h w) c -> b c h w', h=wh, w=ww)
        X2 = self.up(X3)
        X2 = torch.cat((x2, X2), 1)
        X2 = self.conv3(X2)
        wh, ww = X2.shape[2], X2.shape[3]
        X2 = rearrange(X2, 'b c h w -> b (h w) c')
        X2, _, _, _, _, _ = self.ctran2(X2, wh, ww)
        X2 = rearrange(X2, 'b (h w) c -> b c h w', h=wh, w=ww)
        X1 = self.up(X2)
        X1 = torch.cat((x1, X1), 1)
        X1 = self.conv4(X1)
        wh, ww = X1.shape[2], X1.shape[3]
        X1 = rearrange(X1, 'b c h w -> b (h w) c')
        X1, _, _, _, _, _ = self.ctran3(X1, wh, ww)
        X1 = rearrange(X1, 'b (h w) c -> b c h w', h=wh, w=ww)
        X0 = self.up(X1)
        X0 = torch.cat((x_, X0), 1)
        X0 = self.conv5(X0)
        X = self.up(X0)
        X = torch.cat((inputs, x, X), 1)
        alpha = self.convo(X)
        alpha = torch.clamp(alpha, min=0, max=1)
        return alpha


class load_AEMatter_Model:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
        }

    RETURN_TYPES = ("AEMatter_Model", )
    FUNCTION = "test"
    CATEGORY = "AEMatter"

    def test(self):
        return (get_AEMatter_model(get_model_path()), )


class run_AEMatter_inference:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "trimap": ("MASK", ),
                "AEMatter_Model": ("AEMatter_Model", ),
            },
        }

    RETURN_TYPES = ("MASK", )
    FUNCTION = "test"
    CATEGORY = "AEMatter"

    def test(
        self,
        image,
        trimap,
        AEMatter_Model,
    ):

        ret = []
        batch_size = image.shape[0]

        for i in range(batch_size):
            tmp_i = from_torch_image(image[i])
            tmp_m = from_torch_image(trimap[i])
            tmp = do_infer(tmp_i, tmp_m, AEMatter_Model)
            ret.append(tmp)

        ret = to_torch_image(np.array(ret))
        ret = ret.squeeze(-1)
        print(ret.shape)

        return ret


# NODE_CLASS_MAPPINGS = {
#     'load_AEMatter_Model': load_AEMatter_Model,
#     'run_AEMatter_inference': run_AEMatter_inference,
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     'load_AEMatter_Model': 'load_AEMatter_Model',
#     'run_AEMatter_inference': 'run_AEMatter_inference',
# }
