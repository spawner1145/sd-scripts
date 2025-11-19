import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from library import utils, custom_offloading_utils
from library.device_utils import clean_memory_on_device, init_ipex

init_ipex()

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from torch.autograd import Function
from einops import rearrange
import logging

logger = logging.getLogger(__name__)

# ==========================================
# Flux AutoEncoder (Ported from flux_models.py)
# ==========================================

@dataclass
class AutoEncoderParams:
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: Tensor) -> Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        else:
            return mean


class AutoEncoder(nn.Module):
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.reg = DiagonalGaussian()

        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def encode(self, x: Tensor) -> Tensor:
        z = self.reg(self.encoder(x))
        z = self.scale_factor * (z - self.shift_factor)
        return z

    def decode(self, z: Tensor) -> Tensor:
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))

# ==========================================
# SKA (Selective Kernel Attention) from lsnet-test/model/ska.py
# ==========================================

try:
    import triton
    import triton.language as tl
    from torch.amp import custom_bwd, custom_fwd

    def _grid(numel: int, bs: int) -> tuple:
        return (triton.cdiv(numel, bs),)

    @triton.jit
    def _idx(i, n: int, c: int, h: int, w: int):
        ni = i // (c * h * w)
        ci = (i // (h * w)) % c
        hi = (i // w) % h
        wi = i % w
        m = i < (n * c * h * w)
        return ni, ci, hi, wi, m

    @triton.jit
    def ska_fwd(
        x_ptr, w_ptr, o_ptr,
        n, ic, h, w, ks, pad, wc,
        BS: tl.constexpr,
        CT: tl.constexpr, AT: tl.constexpr
    ):
        pid = tl.program_id(0)
        start = pid * BS
        offs = start + tl.arange(0, BS)

        ni, ci, hi, wi, m = _idx(offs, n, ic, h, w)
        val = tl.zeros((BS,), dtype=AT)

        for kh in range(ks):
            hin = hi - pad + kh
            hb = (hin >= 0) & (hin < h)
            for kw in range(ks):
                win = wi - pad + kw
                b = hb & (win >= 0) & (win < w)

                x_off = ((ni * ic + ci) * h + hin) * w + win
                w_off = ((ni * wc + ci % wc) * ks * ks + (kh * ks + kw)) * h * w + hi * w + wi

                x_val = tl.load(x_ptr + x_off, mask=m & b, other=0.0).to(CT)
                w_val = tl.load(w_ptr + w_off, mask=m, other=0.0).to(CT)
                val += tl.where(b & m, x_val * w_val, 0.0).to(AT)

        tl.store(o_ptr + offs, val.to(CT), mask=m)

    @triton.jit
    def ska_bwd_x(
        go_ptr, w_ptr, gi_ptr,
        n, ic, h, w, ks, pad, wc,
        BS: tl.constexpr,
        CT: tl.constexpr, AT: tl.constexpr
    ):
        pid = tl.program_id(0)
        start = pid * BS
        offs = start + tl.arange(0, BS)

        ni, ci, hi, wi, m = _idx(offs, n, ic, h, w)
        val = tl.zeros((BS,), dtype=AT)

        for kh in range(ks):
            ho = hi + pad - kh
            hb = (ho >= 0) & (ho < h)
            for kw in range(ks):
                wo = wi + pad - kw
                b = hb & (wo >= 0) & (wo < w)

                go_off = ((ni * ic + ci) * h + ho) * w + wo
                w_off = ((ni * wc + ci % wc) * ks * ks + (kh * ks + kw)) * h * w + ho * w + wo

                go_val = tl.load(go_ptr + go_off, mask=m & b, other=0.0).to(CT)
                w_val = tl.load(w_ptr + w_off, mask=m, other=0.0).to(CT)
                val += tl.where(b & m, go_val * w_val, 0.0).to(AT)

        tl.store(gi_ptr + offs, val.to(CT), mask=m)

    @triton.jit
    def ska_bwd_w(
        go_ptr, x_ptr, gw_ptr,
        n, wc, h, w, ic, ks, pad,
        BS: tl.constexpr,
        CT: tl.constexpr, AT: tl.constexpr
    ):
        pid = tl.program_id(0)
        start = pid * BS
        offs = start + tl.arange(0, BS)

        ni, ci, hi, wi, m = _idx(offs, n, wc, h, w)

        for kh in range(ks):
            hin = hi - pad + kh
            hb = (hin >= 0) & (hin < h)
            for kw in range(ks):
                win = wi - pad + kw
                b = hb & (win >= 0) & (win < w)
                w_off = ((ni * wc + ci) * ks * ks + (kh * ks + kw)) * h * w + hi * w + wi

                val = tl.zeros((BS,), dtype=AT)
                steps = (ic - ci + wc - 1) // wc
                for s in range(tl.max(steps, axis=0)):
                    cc = ci + s * wc
                    cm = (cc < ic) & m & b

                    x_off = ((ni * ic + cc) * h + hin) * w + win
                    go_off = ((ni * ic + cc) * h + hi) * w + wi

                    x_val = tl.load(x_ptr + x_off, mask=cm, other=0.0).to(CT)
                    go_val = tl.load(go_ptr + go_off, mask=cm, other=0.0).to(CT)
                    val += tl.where(cm, x_val * go_val, 0.0).to(AT)

                tl.store(gw_ptr + w_off, val.to(CT), mask=m)

    class SkaFn(Function):
        @staticmethod
        @custom_fwd(device_type='cuda')
        def forward(ctx, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            ks = int(math.sqrt(w.shape[2]))
            pad = (ks - 1) // 2
            ctx.ks, ctx.pad = ks, pad
            n, ic, h, width = x.shape
            wc = w.shape[1]
            o = torch.empty(n, ic, h, width, device=x.device, dtype=x.dtype)
            numel = o.numel()

            x = x.contiguous()
            w = w.contiguous()

            grid = lambda meta: _grid(numel, meta["BS"])

            if x.dtype == torch.float16:
                ct = tl.float16
                at = tl.float32
            elif x.dtype == torch.bfloat16:
                ct = tl.bfloat16
                at = tl.float32
            elif x.dtype == torch.float32:
                ct = tl.float32
                at = tl.float32
            elif hasattr(torch, "float8_e4m3fn") and x.dtype == torch.float8_e4m3fn:
                ct = tl.float8e4nv
                at = tl.float32
            elif hasattr(torch, "float8_e5m2") and x.dtype == torch.float8_e5m2:
                ct = tl.float8e5
                at = tl.float32
            else:
                # Fallback for other types (e.g. float64), though typically not used in this context
                ct = tl.float32
                at = tl.float32

            ska_fwd[grid](x, w, o, n, ic, h, width, ks, pad, wc, BS=1024, CT=ct, AT=at)

            ctx.save_for_backward(x, w)
            ctx.ct, ctx.at = ct, at
            return o

        @staticmethod
        @custom_bwd(device_type='cuda')
        def backward(ctx, go: torch.Tensor) -> tuple:
            ks, pad = ctx.ks, ctx.pad
            x, w = ctx.saved_tensors
            n, ic, h, width = x.shape
            wc = w.shape[1]

            go = go.contiguous()
            gx = gw = None
            ct, at = ctx.ct, ctx.at

            if ctx.needs_input_grad[0]:
                gx = torch.empty_like(x)
                numel = gx.numel()
                ska_bwd_x[lambda meta: _grid(numel, meta["BS"])](go, w, gx, n, ic, h, width, ks, pad, wc, BS=1024, CT=ct, AT=at)

            if ctx.needs_input_grad[1]:
                gw = torch.empty_like(w)
                numel = gw.numel() // w.shape[2]
                ska_bwd_w[lambda meta: _grid(numel, meta["BS"])](go, x, gw, n, wc, h, width, ic, ks, pad, BS=1024, CT=ct, AT=at)

            return gx, gw, None, None

except ImportError:
    SkaFn = None

def pytorch_ska(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    ks = int(math.sqrt(w.shape[2]))
    pad = (ks - 1) // 2

    n, ic, h, width = x.shape
    wc = w.shape[1]
    x_unfolded = F.unfold(x, kernel_size=ks, padding=pad)

    x_unfolded = x_unfolded.view(n, ic, ks * ks, h * width)

    w = w.view(n, wc, ks * ks, h * width)

    if ic != wc:
        repeats = ic // wc
        w = w.repeat(1, repeats, 1, 1)

    output = (x_unfolded * w).sum(dim=2)

    output = output.view(n, ic, h, width)

    return output


class SKA(torch.nn.Module):
    _fallback_logged = False

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # Fallback for unsupported types to BF16/FP16
        supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        if hasattr(torch, "float8_e4m3fn"): supported_dtypes.append(torch.float8_e4m3fn)
        if hasattr(torch, "float8_e5m2"): supported_dtypes.append(torch.float8_e5m2)

        if x.dtype not in supported_dtypes:
            target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            x = x.to(target_dtype)
            w = w.to(target_dtype)

        if SkaFn is not None:
            try:
                return SkaFn.apply(x, w)  # type: ignore
            except Exception as e:
                msg = str(e)
                if "Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)" in msg:
                    if not SKA._fallback_logged:
                        logger.info(
                            "SkaFn requires GPU due to Triton; CPU input detected. "
                            f"Using pytorch_ska fallback. (Error message: {msg})",
                            exc_info=False,
                        )
                        SKA._fallback_logged = True
                else:
                    if not SKA._fallback_logged:
                        logger.info(
                            "SkaFn failed; falling back to pytorch_ska. "
                            f"(This issue is likely due to a bad Triton setup but can usually be ignored) Error: {msg}",
                            exc_info=False,
                        )
                        SKA._fallback_logged = True
                return pytorch_ska(x, w)
        else:
            return pytorch_ska(x, w)

# ==========================================
# LSNet Components from lsnet-test/model/lsnet.py
# ==========================================

from timm.layers import SqueezeExcite
from timm.models.vision_transformer import trunc_normal_

# for cpu_offload_checkpointing
def to_cuda(x):
    if isinstance(x, torch.Tensor):
        return x.cuda()
    elif isinstance(x, (list, tuple)):
        return [to_cuda(elem) for elem in x]
    elif isinstance(x, dict):
        return {k: to_cuda(v) for k, v in x.items()}
    else:
        return x

def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.cpu()
    elif isinstance(x, (list, tuple)):
        return [to_cpu(elem) for elem in x]
    elif isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    else:
        return x

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x

class LKP(nn.Module):
    def __init__(self, dim, lks, sks, groups):
        super().__init__()
        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = nn.ReLU()
        self.cv2 = Conv2d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv2d_BN(dim // 2, dim // 2)
        self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)
        
        self.sks = sks
        self.groups = groups
        self.dim = dim
        
    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))
        b, _, h, width = w.size()
        w = w.view(b, self.dim // self.groups, self.sks ** 2, h, width)
        return w

class LSConv(nn.Module):
    def __init__(self, dim):
        super(LSConv, self).__init__()
        self.lkp = LKP(dim, lks=7, sks=3, groups=8)
        self.ska = SKA()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(self.ska(x, self.lkp(x))) + x

# ==========================================
# LSUNet (Modified SDXL UNet)
# ==========================================

IN_CHANNELS: int = 16  # Flux VAE uses 16 channels
OUT_CHANNELS: int = 16
ADM_IN_CHANNELS: int = 2560 # 1024 (Qwen) + 1536 (Time IDs)
CONTEXT_DIM: int = 1024 # Qwen 0.6B (0.5B) uses 1024
MODEL_CHANNELS: int = 320
TIME_EMBED_DIM = 320 * 4
USE_REENTRANT = True

def get_timestep_embedding(timesteps, embedding_dim, downscale_freq_shift=1, max_period=10000):
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

    return emb

def get_parameter_dtype(parameter: torch.nn.Module):
    try:
        return next(parameter.parameters()).dtype
    except StopIteration:
        return torch.float32

def get_parameter_device(parameter: torch.nn.Module):
    try:
        return next(parameter.parameters()).device
    except StopIteration:
        return torch.device("cpu")

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        if x.dtype == torch.float32:
            return super().forward(x)
        return F.group_norm(
            x.float(), 
            self.num_groups, 
            self.weight.float() if self.weight is not None else None, 
            self.bias.float() if self.bias is not None else None, 
            self.eps
        ).type(x.dtype)

class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.in_layers = nn.Sequential(
            GroupNorm32(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(TIME_EMBED_DIM, out_channels))

        self.out_layers = nn.Sequential(
            GroupNorm32(32, out_channels),
            nn.SiLU(),
            nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.skip_connection = nn.Identity()

        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

    def forward_body(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        h = h + emb_out[:, :, None, None]
        h = self.out_layers(h)
        x = self.skip_connection(x)
        return x + h

    def forward(self, x, emb):
        if self.training and self.gradient_checkpointing:
            if self.cpu_offload_checkpointing:
                def create_custom_forward(func):
                    def custom_forward(*inputs):
                        cuda_inputs = to_cuda(inputs)
                        outputs = func(*cuda_inputs)
                        return to_cpu(outputs)
                    return custom_forward
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.forward_body), x, emb, use_reentrant=False)
            else:
                def create_custom_forward(func):
                    def custom_forward(*inputs):
                        return func(*inputs)
                    return custom_forward
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.forward_body), x, emb, use_reentrant=USE_REENTRANT)
        else:
            x = self.forward_body(x, emb)
        return x

class Downsample2D(nn.Module):
    def __init__(self, channels, out_channels):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.op = nn.Conv2d(self.channels, self.out_channels, 3, stride=2, padding=1)
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

    def forward_body(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        hidden_states = self.op(hidden_states)
        return hidden_states

    def forward(self, hidden_states):
        if self.training and self.gradient_checkpointing:
            if self.cpu_offload_checkpointing:
                def create_custom_forward(func):
                    def custom_forward(*inputs):
                        cuda_inputs = to_cuda(inputs)
                        outputs = func(*cuda_inputs)
                        return to_cpu(outputs)
                    return custom_forward
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.forward_body), hidden_states, use_reentrant=False
                )
            else:
                def create_custom_forward(func):
                    def custom_forward(*inputs):
                        return func(*inputs)
                    return custom_forward
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.forward_body), hidden_states, use_reentrant=USE_REENTRANT
                )
        else:
            hidden_states = self.forward_body(hidden_states)
        return hidden_states

class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        upcast_attention: bool = False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))

        self.use_memory_efficient_attention_xformers = False
        self.use_memory_efficient_attention_mem_eff = False
        self.use_sdpa = False

    def set_use_memory_efficient_attention(self, xformers, mem_eff):
        self.use_memory_efficient_attention_xformers = xformers
        self.use_memory_efficient_attention_mem_eff = mem_eff

    def set_use_sdpa(self, sdpa):
        self.use_sdpa = sdpa

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def forward(self, hidden_states, context=None, mask=None):
        # Prepare mask if present
        # Assuming mask is (B, L_kv) with 1 for keep, 0 for discard
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1) # (B, 1, 1, L_kv)
            
            # Convert to additive mask if it looks like a binary mask (0/1)
            # We check if values are > -1 (typical additive mask has -inf)
            # This is a heuristic. If the user passes an additive mask, we shouldn't break it.
            # But here we know we are passing tokenizer mask (0/1).
            if mask.dtype != torch.bool and mask.min() > -1:
                 new_mask = torch.zeros_like(mask, dtype=hidden_states.dtype)
                 new_mask.masked_fill_(mask == 0, float("-inf"))
                 mask = new_mask
            elif mask.dtype == torch.bool:
                 new_mask = torch.zeros_like(mask, dtype=hidden_states.dtype)
                 new_mask.masked_fill_(~mask, float("-inf"))
                 mask = new_mask

        if self.use_memory_efficient_attention_xformers:
            return self.forward_memory_efficient_xformers(hidden_states, context, mask)
        if self.use_memory_efficient_attention_mem_eff:
            return self.forward_memory_efficient_mem_eff(hidden_states, context, mask)
        if self.use_sdpa:
            return self.forward_sdpa(hidden_states, context, mask)

        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        # Reshape mask for manual attention: (B, 1, 1, L_kv) -> (B*H, 1, L_kv)
        mask_for_attn = None
        if mask is not None:
            batch_size = hidden_states.shape[0]
            mask_for_attn = mask.repeat(1, self.heads, 1, 1)
            mask_for_attn = mask_for_attn.view(batch_size * self.heads, 1, mask.shape[-1])

        hidden_states = self._attention(query, key, value, mask_for_attn)
        hidden_states = self.to_out[0](hidden_states)
        return hidden_states

    def _attention(self, query, key, value, mask=None):
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        
        if mask is not None:
            attention_scores = attention_scores + mask

        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(value.dtype)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def forward_memory_efficient_xformers(self, x, context=None, mask=None):
        import xformers.ops
        h = self.heads
        q_in = self.to_q(x)
        context = context if context is not None else x
        context = context.to(x.dtype)
        k_in = self.to_k(context)
        v_in = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b n h d", h=h), (q_in, k_in, v_in))
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        # xformers expects attn_bias to be broadcastable
        # mask is (B, 1, 1, L_kv)
        # xformers might want (B, 1, L_q, L_kv) or similar.
        # If mask is additive, we can pass it directly as attn_bias
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask)
        out = rearrange(out, "b n h d -> b n (h d)", h=h)
        out = self.to_out[0](out)
        return out

    def forward_memory_efficient_mem_eff(self, x, context=None, mask=None):
        # Placeholder for FlashAttentionFunction if needed, but xformers is preferred
        return self.forward_sdpa(x, context, mask)

    def forward_sdpa(self, x, context=None, mask=None):
        h = self.heads
        q_in = self.to_q(x)
        context = context if context is not None else x
        context = context.to(x.dtype)
        k_in = self.to_k(context)
        v_in = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q_in, k_in, v_in))
        
        # SDPA expects attn_mask
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.to_out[0](out)
        return out

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def gelu(self, gate):
        if gate.device.type != "mps":
            return F.gelu(gate)
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        inner_dim = int(dim * 4)
        self.net = nn.ModuleList([])
        self.net.append(GEGLU(dim, inner_dim))
        self.net.append(nn.Identity())
        self.net.append(nn.Linear(inner_dim, dim))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states

class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim, channels, eps=1e-6):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, channels * 2)
        self.norm = RMSNorm(channels, eps=eps)

    def forward(self, x, timestep_emb):
        emb = self.linear(self.silu(timestep_emb))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x

class LSUNetBlock(nn.Module):
    def __init__(
        self, dim: int, num_attention_heads: int, attention_head_dim: int, cross_attention_dim: int, upcast_attention: bool = False, time_embed_dim: Optional[int] = None
    ):
        super().__init__()
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

        # 1. LSConv (Replacing Self-Attention)
        self.ls_conv = LSConv(dim)
        
        # 2. Cross-Attn
        self.attn2 = CrossAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            upcast_attention=upcast_attention,
        )

        self.norm1 = RMSNorm(dim)
        
        if time_embed_dim is not None:
            self.norm2 = AdaLayerNorm(time_embed_dim, dim)
            self.norm3 = AdaLayerNorm(time_embed_dim, dim)
        else:
            self.norm2 = RMSNorm(dim)
            self.norm3 = RMSNorm(dim)

        # 3. Feed-forward
        self.ff = FeedForward(dim)

    def set_use_memory_efficient_attention(self, xformers: bool, mem_eff: bool):
        self.attn2.set_use_memory_efficient_attention(xformers, mem_eff)

    def set_use_sdpa(self, sdpa: bool):
        self.attn2.set_use_sdpa(sdpa)

    def forward_body(self, hidden_states, context=None, context_mask=None, timestep=None):
        # hidden_states is (B, N, C)
        B, N, C = hidden_states.shape
        H = int(math.sqrt(N))
        W = H
        
        # 1. LSConv (Spatial Mixing)
        # Reshape to (B, C, H, W)
        x = hidden_states.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.ls_conv(x)
        # Reshape back to (B, N, C)
        x = x.reshape(B, C, N).permute(0, 2, 1)
        
        hidden_states = x

        # 2. Cross-Attention
        if isinstance(self.norm2, AdaLayerNorm):
             norm_hidden_states = self.norm2(hidden_states, timestep)
        else:
             norm_hidden_states = self.norm2(hidden_states)
             
        hidden_states = self.attn2(norm_hidden_states, context=context, mask=context_mask) + hidden_states

        # 3. Feed-forward
        if isinstance(self.norm3, AdaLayerNorm):
            norm_hidden_states = self.norm3(hidden_states, timestep)
        else:
            norm_hidden_states = self.norm3(hidden_states)
            
        hidden_states = self.ff(norm_hidden_states) + hidden_states

        return hidden_states

    def forward(self, hidden_states, context=None, context_mask=None, timestep=None):
        if self.training and self.gradient_checkpointing:
            if self.cpu_offload_checkpointing:
                def create_custom_forward(func):
                    def custom_forward(*inputs):
                        cuda_inputs = to_cuda(inputs)
                        outputs = func(*cuda_inputs)
                        return to_cpu(outputs)
                    return custom_forward
                output = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.forward_body), hidden_states, context, context_mask, timestep, use_reentrant=False
                )
            else:
                def create_custom_forward(func):
                    def custom_forward(*inputs):
                        return func(*inputs)
                    return custom_forward
                output = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.forward_body), hidden_states, context, context_mask, timestep, use_reentrant=USE_REENTRANT
                )
        else:
            output = self.forward_body(hidden_states, context, context_mask, timestep)
        return output

class LSTransformer2DModel(nn.Module):
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        use_linear_projection: bool = False,
        upcast_attention: bool = False,
        num_transformer_layers: int = 1,
        time_embed_dim: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.use_linear_projection = use_linear_projection

        self.norm = GroupNorm32(32, in_channels, eps=1e-6, affine=True)

        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        blocks = []
        for _ in range(num_transformer_layers):
            blocks.append(
                LSUNetBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                    time_embed_dim=time_embed_dim,
                )
            )

        self.transformer_blocks = nn.ModuleList(blocks)

        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

        self.gradient_checkpointing = False

    def set_use_memory_efficient_attention(self, xformers, mem_eff):
        for transformer in self.transformer_blocks:
            transformer.set_use_memory_efficient_attention(xformers, mem_eff)

    def set_use_sdpa(self, sdpa):
        for transformer in self.transformer_blocks:
            transformer.set_use_sdpa(sdpa)

    def forward(self, hidden_states, encoder_hidden_states=None, context_mask=None, timestep=None):
        batch, _, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, context=encoder_hidden_states, context_mask=context_mask, timestep=timestep)

        if not self.use_linear_projection:
            hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual
        return output

class Upsample2D(nn.Module):
    def __init__(self, channels, out_channels):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

    def forward_body(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)
        hidden_states = self.conv(hidden_states)
        return hidden_states

    def forward(self, hidden_states, output_size=None):
        if self.training and self.gradient_checkpointing:
            if self.cpu_offload_checkpointing:
                def create_custom_forward(func):
                    def custom_forward(*inputs):
                        cuda_inputs = to_cuda(inputs)
                        outputs = func(*cuda_inputs)
                        return to_cpu(outputs)
                    return custom_forward
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.forward_body), hidden_states, output_size, use_reentrant=False
                )
            else:
                def create_custom_forward(func):
                    def custom_forward(*inputs):
                        return func(*inputs)
                    return custom_forward
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.forward_body), hidden_states, output_size, use_reentrant=USE_REENTRANT
                )
        else:
            hidden_states = self.forward_body(hidden_states, output_size)
        return hidden_states

class LSUNet(nn.Module):
    _supports_gradient_checkpointing = True

    def __init__(self, **kwargs):
        super().__init__()
        self.in_channels = IN_CHANNELS
        self.out_channels = OUT_CHANNELS
        self.model_channels = MODEL_CHANNELS
        self.time_embed_dim = TIME_EMBED_DIM
        self.adm_in_channels = ADM_IN_CHANNELS
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        # self.label_emb removed as we don't use pooled output anymore

        self.input_blocks = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(self.in_channels, self.model_channels, kernel_size=3, padding=(1, 1)))]
        )

        # level 0
        for i in range(2):
            layers = [ResnetBlock2D(in_channels=1 * self.model_channels, out_channels=1 * self.model_channels)]
            self.input_blocks.append(nn.ModuleList(layers))

        self.input_blocks.append(nn.Sequential(Downsample2D(channels=1 * self.model_channels, out_channels=1 * self.model_channels)))

        # level 1
        for i in range(2):
            layers = [
                ResnetBlock2D(in_channels=(1 if i == 0 else 2) * self.model_channels, out_channels=2 * self.model_channels),
                LSTransformer2DModel(
                    num_attention_heads=2 * self.model_channels // 64,
                    attention_head_dim=64,
                    in_channels=2 * self.model_channels,
                    num_transformer_layers=2,
                    use_linear_projection=True,
                    cross_attention_dim=CONTEXT_DIM,
                    time_embed_dim=self.time_embed_dim,
                ),
            ]
            self.input_blocks.append(nn.ModuleList(layers))

        self.input_blocks.append(nn.Sequential(Downsample2D(channels=2 * self.model_channels, out_channels=2 * self.model_channels)))

        # level 2
        for i in range(2):
            layers = [
                ResnetBlock2D(in_channels=(2 if i == 0 else 4) * self.model_channels, out_channels=4 * self.model_channels),
                LSTransformer2DModel(
                    num_attention_heads=4 * self.model_channels // 64,
                    attention_head_dim=64,
                    in_channels=4 * self.model_channels,
                    num_transformer_layers=10,
                    use_linear_projection=True,
                    cross_attention_dim=CONTEXT_DIM,
                    time_embed_dim=self.time_embed_dim,
                ),
            ]
            self.input_blocks.append(nn.ModuleList(layers))

        # mid
        self.middle_block = nn.ModuleList(
            [
                ResnetBlock2D(in_channels=4 * self.model_channels, out_channels=4 * self.model_channels),
                LSTransformer2DModel(
                    num_attention_heads=4 * self.model_channels // 64,
                    attention_head_dim=64,
                    in_channels=4 * self.model_channels,
                    num_transformer_layers=10,
                    use_linear_projection=True,
                    cross_attention_dim=CONTEXT_DIM,
                    time_embed_dim=self.time_embed_dim,
                ),
                ResnetBlock2D(in_channels=4 * self.model_channels, out_channels=4 * self.model_channels),
            ]
        )

        # output
        self.output_blocks = nn.ModuleList([])

        # level 2
        for i in range(3):
            layers = [
                ResnetBlock2D(
                    in_channels=4 * self.model_channels + (4 if i <= 1 else 2) * self.model_channels,
                    out_channels=4 * self.model_channels,
                ),
                LSTransformer2DModel(
                    num_attention_heads=4 * self.model_channels // 64,
                    attention_head_dim=64,
                    in_channels=4 * self.model_channels,
                    num_transformer_layers=10,
                    use_linear_projection=True,
                    cross_attention_dim=CONTEXT_DIM,
                    time_embed_dim=self.time_embed_dim,
                ),
            ]
            if i == 2:
                layers.append(Upsample2D(channels=4 * self.model_channels, out_channels=4 * self.model_channels))
            self.output_blocks.append(nn.ModuleList(layers))

        # level 1
        for i in range(3):
            layers = [
                ResnetBlock2D(
                    in_channels=2 * self.model_channels + (4 if i == 0 else (2 if i == 1 else 1)) * self.model_channels,
                    out_channels=2 * self.model_channels,
                ),
                LSTransformer2DModel(
                    num_attention_heads=2 * self.model_channels // 64,
                    attention_head_dim=64,
                    in_channels=2 * self.model_channels,
                    num_transformer_layers=2,
                    use_linear_projection=True,
                    cross_attention_dim=CONTEXT_DIM,
                    time_embed_dim=self.time_embed_dim,
                ),
            ]
            if i == 2:
                layers.append(Upsample2D(channels=2 * self.model_channels, out_channels=2 * self.model_channels))
            self.output_blocks.append(nn.ModuleList(layers))

        # level 0
        for i in range(3):
            layers = [
                ResnetBlock2D(
                    in_channels=1 * self.model_channels + (2 if i == 0 else 1) * self.model_channels,
                    out_channels=1 * self.model_channels,
                ),
            ]
            self.output_blocks.append(nn.ModuleList(layers))

        self.out = nn.ModuleList(
            [GroupNorm32(32, self.model_channels), nn.SiLU(), nn.Conv2d(self.model_channels, self.out_channels, 3, padding=1)]
        )

    def prepare_config(self):
        from types import SimpleNamespace
        self.config = SimpleNamespace()

    @property
    def dtype(self) -> torch.dtype:
        return get_parameter_dtype(self)

    @property
    def device(self) -> torch.device:
        return get_parameter_device(self)

    def is_gradient_checkpointing(self) -> bool:
        return any(hasattr(m, "gradient_checkpointing") and m.gradient_checkpointing for m in self.modules())

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload
        self.set_gradient_checkpointing(value=True, cpu_offload=cpu_offload)

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        self.set_gradient_checkpointing(value=False, cpu_offload=False)

    def set_use_memory_efficient_attention(self, xformers: bool, mem_eff: bool) -> None:
        blocks = self.input_blocks + [self.middle_block] + self.output_blocks
        for block in blocks:
            for module in block:
                if hasattr(module, "set_use_memory_efficient_attention"):
                    module.set_use_memory_efficient_attention(xformers, mem_eff)

    def set_use_sdpa(self, sdpa: bool) -> None:
        blocks = self.input_blocks + [self.middle_block] + self.output_blocks
        for block in blocks:
            for module in block:
                if hasattr(module, "set_use_sdpa"):
                    module.set_use_sdpa(sdpa)

    def set_gradient_checkpointing(self, value=False, cpu_offload=False):
        blocks = self.input_blocks + [self.middle_block] + self.output_blocks
        for block in blocks:
            for module in block.modules():
                if hasattr(module, "gradient_checkpointing"):
                    module.gradient_checkpointing = value
                    if hasattr(module, "cpu_offload_checkpointing"):
                        module.cpu_offload_checkpointing = cpu_offload

    def forward(self, x, timesteps=None, context=None, context_mask=None, **kwargs):
        timesteps = timesteps.expand(x.shape[0])
        hs = []
        t_emb = get_timestep_embedding(timesteps, self.model_channels, downscale_freq_shift=0)
        t_emb = t_emb.to(x.dtype)
        emb = self.time_embed(t_emb)
        
        def call_module(module, h, emb, context, context_mask):
            x = h
            for layer in module:
                if isinstance(layer, ResnetBlock2D):
                    x = layer(x, emb)
                elif isinstance(layer, LSTransformer2DModel):
                    x = layer(x, context, context_mask=context_mask, timestep=emb)
                else:
                    x = layer(x)
            return x

        h = x
        for module in self.input_blocks:
            h = call_module(module, h, emb, context, context_mask)
            hs.append(h)

        h = call_module(self.middle_block, h, emb, context, context_mask)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = call_module(module, h, emb, context, context_mask)

        h = h.type(x.dtype)
        h = call_module(self.out, h, emb, context, context_mask)
        return h

class TextProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        
        # Initialize with fallback to float32 for precision
        orig_dtype = self.linear.weight.dtype
        self.linear.weight.data = self.linear.weight.data.to(torch.float32)
        if in_dim == out_dim:
            self.linear.weight.data.copy_(torch.eye(in_dim))
        else:
            torch.nn.init.orthogonal_(self.linear.weight)
        self.linear.weight.data = self.linear.weight.data.to(orig_dtype)

    def forward(self, x):
        return self.linear(x)
