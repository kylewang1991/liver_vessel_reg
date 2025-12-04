import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
import math
# from hilbert import decode, encode
# from pyzorder import ZOrderIndexer

import torch
import torch.nn as nn
from typing import Optional, Union, Type, List, Tuple, Callable, Dict
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

import numpy as np
import torch
import torch.nn as nn

class HSCANS(nn.Module):
    def __init__(self, size_list, dim=3, scan_type='scan'):
        super().__init__()
        assert len(size_list) == dim, "size_list length must match dim"
        self.size_list = size_list
        max_num = np.prod(size_list)
        indexes = np.arange(max_num)
        self.dim = dim
        if scan_type == 'sweep':  # ['sweep', 'scan']
            locs_flat = indexes
        elif scan_type == 'scan':
            if dim == 2:
                indexes = indexes.reshape(size_list)
                for i in np.arange(1, size_list[0], step=2):
                    indexes[i, :] = indexes[i, :][::-1]
                locs_flat = indexes.reshape(-1)
            elif dim == 3:
                indexes = indexes.reshape(size_list)
                for i in np.arange(1, size_list[1], step=2):
                    indexes[:, i, :] = np.flip(indexes[:, i, :], axis=1)  # Flipping along depth dimension
                for j in np.arange(1, size_list[0], step=2):
                    indexes[j, :, :] = np.flip(indexes[j, :, :], axis=(0, 1))  # Flipping along height and depth
                locs_flat = indexes.reshape(-1)
            else:
                raise NotImplementedError("Dimension not supported for scan_type 'scan'")
        else:
            raise Exception('invalid encoder mode')

        locs_flat_inv = np.argsort(locs_flat)
        index_flat = torch.LongTensor(locs_flat.astype(np.int64)).unsqueeze(0).unsqueeze(1)
        index_flat_inv = torch.LongTensor(locs_flat_inv.astype(np.int64)).unsqueeze(0).unsqueeze(1)
        self.index_flat = nn.Parameter(index_flat, requires_grad=False)
        self.index_flat_inv = nn.Parameter(index_flat_inv, requires_grad=False)

    def __call__(self, img):
        img_encode = self.encode(img)
        return img_encode

    def encode(self, img):
        img_encode = torch.zeros_like(img).scatter_(
            2, self.index_flat_inv.expand(img.shape), img
        )
        return img_encode

    def decode(self, img):
        img_decode = torch.zeros_like(img).scatter_(
            2, self.index_flat.expand(img.shape), img
        )
        return img_decode
class SS3D(nn.Module): #for the original Vanilla VSS block, worse as described in VMamba paper
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=1, #2
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device='cuda',
            dtype=None,
            **kwargs,
            ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model #channel dim, 512 or 1024, gets expanded
        self.d_state = d_state
        
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=8, N, inner) = (K=8, new_c = self.dt_rank + self.d_state * 2, C)
        del self.x_proj

        self.dt_projs = (
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=8, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=8, merge=True) # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
                torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=8, device=None, merge=True):
        # S4D real initialization
        A = repeat(
                torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=d_inner,
        ).contiguous()
        #('A', A.shape)
        A_log = torch.log(A)    # Keep A_log in fp32

        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=8, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)    # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        #0,1, 2, 3, 4
        B, C, H, W, D = x.shape
        L = H * W * D
        K = 8

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L), torch.transpose(x, dim0=2, dim1=4).contiguous().view(B, -1, L), torch.transpose(x, dim0=3, dim1=4).contiguous().view(B, -1, L)], dim=1).view(B, 4, -1, L)
        # hwd, whd, dwh, hdw; reversed
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, c, l)
        # hwd b, 1, c, l >
        # whd b, 1, c, l >
        # dwh b, 1, c, l >
        # hdw b, 1, c, l >
        # hwd reversed l
        # whd reversed l
        # dwh reversed l
        # hdw reversed l

        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        
        xs = xs.float().view(B, -1, L) # (b, k * d, l)

        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)    # (k * d, d_state)
        
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        

        out_y = self.selective_scan(
                xs, dts,
                As, Bs, Cs, Ds, z=None,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
                return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        # hwd b, 1, c, l >
        # whd b, 1, c, l >
        # dwh b, 1, c, l >
        # hdw b, 1, c, l >
        # hwd reversed l
        # whd reversed l
        # dwh reversed l
        # hdw reversed l

        #revert back to all hwd forward l

        #out1 = out_y[:,0,:,:]
        out2 = torch.transpose(out_y[:, 1].view(B, -1, W, H, D), dim0=2, dim1=3).contiguous().view(B, -1, L)
        out3 = torch.transpose(out_y[:, 2].view(B, -1, W, H, D), dim0=2, dim1=4).contiguous().view(B, -1, L)
        out4 = torch.transpose(out_y[:, 3].view(B, -1, W, H, D), dim0=3, dim1=4).contiguous().view(B, -1, L)

        out5 = torch.flip(out_y[:, 0], dims=[-1]).view(B, -1, L)
        out6 = torch.flip(out2, dims=[-1]).view(B, -1, L)
        out7 = torch.flip(out3, dims=[-1]).view(B, -1, L)
        out8 = torch.flip(out4, dims=[-1]).view(B, -1, L)

        return out_y[:, 0], out2, out3, out4, out5, out6, out7, out8

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, D, C = x.shape #!!!
        #d_model = C
        
        xz = self.in_proj(x) # (b, h, w, d, d_model) -> (b, h, w, d, d_inner * 2)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d, d_inner), z for the multiplicative path
        
        x = x.permute(0, 4, 1, 2, 3).contiguous()    
        x = self.act(self.conv3d(x)) # (b, d, h, w)
        
        y1, y2, y3, y4, y5, y6, y7, y8 = self.forward_core(x) # 1 1024 1728
        
        assert y1.dtype == torch.float32
        
        y = y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8
        
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, D, -1) #bcl > blc > bhwdc
        y = self.out_norm(y)
        y = y * F.silu(z) #multiplicative path, ignored in v2 because ssm is inherently selective, described in VMamba
        
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        
        return out
    
class SS3D_v5(nn.Module): #no multiplicative path, the better version described in VMamba
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=1, #2
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device='cuda',
            dtype=None,
            einsum=True,
            size_list=(16, 16, 8), 
            scan_type='scan',#size needs to be a power of 2 to use hilbert
            num_direction = 8,
            orientation = 0, #0, 1, 2
            **kwargs,
            ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.orientation = orientation
        self.d_model = d_model #channel dim, 512 or 1024, gets expanded
        self.d_state = d_state
        
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        if einsum:
            self.x_proj = (
                    nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs) for i in range(num_direction)
            )
            self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=8, N, inner) = (K=8, new_c = self.dt_rank + self.d_state * 2, C)
            del self.x_proj
        else:
            #print('no einsum for x_proj')
            raise Exception('have to use einsum for now lol')
        # figure out how to do dts without einsum
        self.dt_projs = [
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs) for i in range(num_direction)
                ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=num_direction, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=num_direction, merge=True) # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        
        self.scans = HSCANS(size_list, scan_type=scan_type)

        # self.scans.encode = lambda x: x
        # self.scans.decode = lambda x: x

        self.num_direction = num_direction

        if (orientation%3) == 0:
            self.transp = lambda x: x
        elif (orientation%3) == 1:
            self.transp = lambda x: torch.transpose(x, dim0=2, dim1=3) # change to 3 4 if hilbert
        elif (orientation%3) == 2:
            self.transp = lambda x: torch.transpose(x, dim0=2, dim1=4) # scan goes across first dim
        self.transp2 = lambda x: x
        if (orientation%6) > 2: # 3, 4, 5
            self.transp2 = lambda x: torch.transpose(x, dim0=3, dim1=4)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
                torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=8, device=None, merge=True):
        # S4D real initialization
        A = repeat(
                torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=d_inner,
        ).contiguous()
        #('A', A.shape)
        A_log = torch.log(A)    # Keep A_log in fp32

        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=8, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)    # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        #0,1, 2, 3, 4
        B, C, H, W, D = x.shape
        L = H * W * D
        K = self.num_direction
        xs = []

        xs.append(self.scans.encode(self.transp2(self.transp(x)).contiguous().view(B, -1, L)))
        
        xs.append(self.scans.encode(self.transp2(self.transp(torch.rot90(torch.rot90(x, k=1, dims=(3,4)), k=1, dims=(2,4)))).contiguous().view(B, -1, L)))
        xs.append(self.scans.encode(self.transp2(self.transp(torch.rot90(x, k=2, dims=(2,4)))).contiguous().view(B, -1, L)))
        xs.append(self.scans.encode(self.transp2(self.transp(torch.rot90(torch.rot90(x, k=-1, dims=(2,4)), k=1, dims=(2,3)))).contiguous().view(B, -1, L)))
        
        # xs.append(self.scans.encode(self.transp2(self.transp(torch.rot90(x, k=2, dims=(2,3)))).contiguous().view(B, -1, L)))
        # xs.append(self.scans.encode(self.transp2(self.transp(torch.rot90(x, k=2, dims=(2,4)))).contiguous().view(B, -1, L)))
        # xs.append(self.scans.encode(self.transp2(self.transp(torch.rot90(x, k=2, dims=(3,4)))).contiguous().view(B, -1, L)))
        
        xs = torch.stack(xs,dim=1).view(B, K // 2, -1, L)
        xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1) # (b, k, c, l)
                
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        
        xs = xs.float().view(B, -1, L) # (b, k * d, l)

        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)    # (k * d, d_state)
        
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
                xs, dts,
                As, Bs, Cs, Ds, z=None,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
                return_last_state=False,
        ).view(B, K, -1, L)
        
        assert out_y.dtype == torch.float

        # out_y = xs.view(B, K, -1, L) # for testing

        inv_y = torch.flip(out_y[:, K // 2:K], dims=[-1]).view(B, K // 2, -1, L)
        ys = []

        # xs.append(self.scans.encode(self.transp2(self.transp(x)).contiguous().view(B, -1, L)))        
        ys.append(self.transp(self.transp2(self.scans.decode(out_y[:, 0]).view(B, -1, H, W, D))).contiguous().view(B, -1, L))
        ys.append(self.transp(self.transp2(self.scans.decode(inv_y[:, 0]).view(B, -1, H, W, D))).contiguous().view(B, -1, L))

        # xs.append(self.scans.encode(self.transp2(self.transp(torch.rot90(torch.rot90(x, k=1, dims=(3,4)), k=1, dims=(2,4)))).contiguous().view(B, -1, L)))
        ys.append(torch.rot90(torch.rot90(self.transp(self.transp2(self.scans.decode(out_y[:, 1]).view(B, -1, H, W, D))), k=-1, dims=(2,4)), k=-1, dims=(3,4)).contiguous().view(B, -1, L))
        ys.append(torch.rot90(torch.rot90(self.transp(self.transp2(self.scans.decode(inv_y[:, 1]).view(B, -1, H, W, D))), k=-1, dims=(2,4)), k=-1, dims=(3,4)).contiguous().view(B, -1, L))

        # xs.append(self.scans.encode(self.transp2(self.transp(torch.rot90(x, k=2, dims=(2,4)))).contiguous().view(B, -1, L)))
        ys.append(torch.rot90(self.transp(self.transp2(self.scans.decode(out_y[:, 2]).view(B, -1, H, W, D))), k=2, dims=(2,4)).contiguous().view(B, -1, L))
        ys.append(torch.rot90(self.transp(self.transp2(self.scans.decode(inv_y[:, 2]).view(B, -1, H, W, D))), k=2, dims=(2,4)).contiguous().view(B, -1, L))

        # xs.append(self.scans.encode(self.transp2(self.transp(torch.rot90(torch.rot90(x, k=3, dims=(2,4)), k=1, dims=(2,3)))).contiguous().view(B, -1, L)))
        ys.append(torch.rot90(torch.rot90(self.transp(self.transp2(self.scans.decode(out_y[:, 3]).view(B, -1, H, W, D))), k=-1, dims=(2,3)), k=1, dims=(2,4)).contiguous().view(B, -1, L))
        ys.append(torch.rot90(torch.rot90(self.transp(self.transp2(self.scans.decode(inv_y[:, 3]).view(B, -1, H, W, D))), k=-1, dims=(2,3)), k=1, dims=(2,4)).contiguous().view(B, -1, L))
        
        # ys.append(torch.rot90(self.transp(self.transp2(self.scans.decode(out_y[:, 1]).view(B, -1, H, W, D))), k=2, dims=(2,3)).contiguous().view(B, -1, L))
        # ys.append(torch.rot90(self.transp(self.transp2(self.scans.decode(inv_y[:, 1]).view(B, -1, H, W, D))), k=2, dims=(2,3)).contiguous().view(B, -1, L))

        # ys.append(torch.rot90(self.transp(self.transp2(self.scans.decode(out_y[:, 2]).view(B, -1, H, W, D))), k=2, dims=(2,4)).contiguous().view(B, -1, L))
        # ys.append(torch.rot90(self.transp(self.transp2(self.scans.decode(inv_y[:, 2]).view(B, -1, H, W, D))), k=2, dims=(2,4)).contiguous().view(B, -1, L))

        # ys.append(torch.rot90(self.transp(self.transp2(self.scans.decode(out_y[:, 3]).view(B, -1, H, W, D))), k=2, dims=(3,4)).contiguous().view(B, -1, L))
        # ys.append(torch.rot90(self.transp(self.transp2(self.scans.decode(inv_y[:, 3]).view(B, -1, H, W, D))), k=2, dims=(3,4)).contiguous().view(B, -1, L))
        
        # for y in ys:
        #     print(torch.all(y==x.view(B, -1, L)))
        return sum(ys)

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, D, C = x.shape #!!!

        x = self.in_proj(x)

        x = x.permute(0, 4, 1, 2, 3).contiguous()        
        x = self.act(self.conv3d(x)) # (b, d, h, w)
        y = self.forward_core(x) # 1 1024 1728
        
        assert y.dtype == torch.float32
                
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, D, -1) #bcl > blc > bhwdc
        
        y = self.out_norm(y)
        
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        
        return out

class SS3D_v6(nn.Module): #no multiplicative path, the better version described in VMamba
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=1, #2
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device='cuda',
            dtype=None,
            einsum=True,
            size_list=(16, 16, 8), 
            scan_type='scan',#size needs to be a power of 2 to use hilbert
            num_direction = 6,
            orientation = 0, #0, 1, 2, 4, 5, 6, 7
            **kwargs,
            ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.orientation = orientation
        self.d_model = d_model #channel dim, 512 or 1024, gets expanded
        self.d_state = d_state
        
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        if einsum:
            self.x_proj = (
                    nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs) for i in range(num_direction)
            )
            self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=8, N, inner) = (K=8, new_c = self.dt_rank + self.d_state * 2, C)
            del self.x_proj
        else:
            #print('no einsum for x_proj')
            raise Exception('have to use einsum for now lol')
        # figure out how to do dts without einsum
        self.dt_projs = [
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs) for i in range(num_direction)
                ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=num_direction, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=num_direction, merge=True) # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        
        # self.scans = HSCANS(size=size, scan_type=scan_type)

        # self.scans.encode = lambda x: x
        # self.scans.decode = lambda x: x

        self.num_direction = num_direction

        if (orientation%8) ==0:
            self.rot = lambda x: x
            self.unrot = lambda x: x
        elif (orientation%8) ==1:
            self.rot = lambda x: torch.rot90(x, 1, (2,3))
            self.unrot = lambda x: torch.rot90(x, -1, (2,3))
        elif (orientation%8) ==2:
            self.rot = lambda x: torch.rot90(x, 1, (3,4))
            self.unrot = lambda x: torch.rot90(x, -1, (3,4))
        elif (orientation%8) ==3:
            self.rot = lambda x: torch.rot90(x, -1, (2,4))
            self.unrot = lambda x: torch.rot90(x, 1, (2,4))
        elif (orientation%8) ==4:
            self.rot = lambda x: torch.transpose(torch.transpose(torch.rot90(torch.rot90(x, 2, (2,3)), 1, (2,4)), 2, 4), 2, 3)
            self.unrot = lambda x: torch.rot90(torch.rot90(torch.transpose(torch.transpose(x, 3, 4), 2, 3), -1, (2,4)), 2, (2,3))
        elif (orientation%8) ==5:
            self.rot = lambda x: torch.rot90(x, 2, (2,4))
            self.unrot = lambda x: torch.rot90(x, 2, (2,4))
        elif (orientation%8) ==6:
            self.rot = lambda x: torch.transpose(torch.transpose(torch.rot90(x, 2, (2,3)), 3,4), 2,3)
            self.unrot = lambda x: torch.rot90(torch.transpose(torch.transpose(x, 2,3), 3,4), 2, (2,3))
        elif (orientation%8) ==7:
            self.rot = lambda x: torch.transpose(torch.transpose(torch.rot90(x, 2, (3,4)), 2,3), 3,4)
            self.unrot = lambda x: torch.rot90(torch.transpose(torch.transpose(x, 3,4), 2,3), 2, (3,4))
        
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
                torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=8, device=None, merge=True):
        # S4D real initialization
        A = repeat(
                torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=d_inner,
        ).contiguous()
        #('A', A.shape)
        A_log = torch.log(A)    # Keep A_log in fp32

        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=8, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)    # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        #0,1, 2, 3, 4
        B, C, H, W, D = x.shape
        L = H * W * D
        K = self.num_direction
        xs = []

        xs.append(self.rot(x).contiguous().view(B, -1, L))
        xs.append(torch.transpose(self.rot(x), 2, 4).contiguous().view(B, -1, L))
        xs.append(torch.transpose(self.rot(x), 3, 4).contiguous().view(B, -1, L))
        
        xs = torch.stack(xs,dim=1).view(B, K // 2, -1, L)
        xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1) # (b, k, c, l)
                
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        
        xs = xs.float().view(B, -1, L) # (b, k * d, l)

        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)    # (k * d, d_state)
        
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
                xs, dts,
                As, Bs, Cs, Ds, z=None,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
                return_last_state=False,
        ).view(B, K, -1, L)
        
        assert out_y.dtype == torch.float

        # out_y = xs.view(B, K, -1, L) # for testing

        inv_y = torch.flip(out_y[:, K // 2:K], dims=[-1]).view(B, K // 2, -1, L)
        ys = []

        # xs.append(self.rot(x).contiguous().view(B, -1, L))
        # xs.append(torch.transpose(self.rot(x), 2, 4).contiguous().view(B, -1, L)))
        # xs.append(torch.transpose(self.rot(x), 3, 4).contiguous().view(B, -1, L)))
        
        ys.append(self.unrot(out_y[:, 0].view(B, -1, H, W, D)).contiguous().view(B, -1, L))
        ys.append(self.unrot(inv_y[:, 0].view(B, -1, H, W, D)).contiguous().view(B, -1, L))

        ys.append(self.unrot(torch.transpose(out_y[:, 1].view(B, -1, H, W, D), 2, 4)).contiguous().view(B, -1, L))
        ys.append(self.unrot(torch.transpose(inv_y[:, 1].view(B, -1, H, W, D), 2, 4)).contiguous().view(B, -1, L))
            
        ys.append(self.unrot(torch.transpose(out_y[:, 2].view(B, -1, H, W, D), 3, 4)).contiguous().view(B, -1, L))
        ys.append(self.unrot(torch.transpose(inv_y[:, 2].view(B, -1, H, W, D), 3, 4)).contiguous().view(B, -1, L))
        
        # for y in ys:
        #     print(torch.all(y==x.view(B, -1, L)))
        return sum(ys)

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, D, C = x.shape #!!!

        x = self.in_proj(x)

        x = x.permute(0, 4, 1, 2, 3).contiguous()        
        x = self.act(self.conv3d(x)) # (b, d, h, w)
        y = self.forward_core(x) # 1 1024 1728
        
        assert y.dtype == torch.float32
                
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, D, -1) #bcl > blc > bhwdc
        
        y = self.out_norm(y)
        
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        
        return out
    

class FeedForward(nn.Module):
    def __init__(self, dim, dropout_rate, hidden_dim = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim=dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)
class VSSBlock3D(nn.Module):
  def __init__(
      self,
      hidden_dim: int = 0,
      drop_path: float = 0,
      norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
      attn_drop_rate: float = 0,
      d_state: int = 16,
      expansion_factor = 1,
      **kwargs,
      ):
    super().__init__()
    self.ln_1 = norm_layer(hidden_dim)
    self.self_attention = SS3D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, expand=expansion_factor, **kwargs)
    self.drop_path = DropPath(drop_path)

  def forward(self, input: torch.Tensor):
    x = input + self.drop_path(self.self_attention(self.ln_1(input)))
    return x
  
class VSSBlock3D_v5(nn.Module): #no multiplicative path, added MLP. more like transformer block used in TABSurfer now
  def __init__(
      self,
      hidden_dim: int = 0,
      drop_path: float = 0,
      norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
      attn_drop_rate: float = 0,
      d_state: int = 16,
      expansion_factor = 1, # can only be 1 for v3, no linear projection to increase channels
      mlp_drop_rate=0.,
      orientation = 0,
      scan_type = 'scan',
      size_list = (16, 16, 8),
      **kwargs,
      ):
    super().__init__()
    print(orientation, end='')
    self.ln_1 = norm_layer(hidden_dim)
    self.self_attention = SS3D_v5(d_model=hidden_dim, 
                                  dropout=attn_drop_rate, 
                                  d_state=d_state, 
                                  expand=expansion_factor, 
                                  orientation=orientation, 
                                  scan_type=scan_type, 
                                  size_list=size_list,
                                  **kwargs)

    self.ln_2 = norm_layer(hidden_dim)
    self.mlp = FeedForward(dim = hidden_dim, hidden_dim=expansion_factor*hidden_dim, dropout_rate = mlp_drop_rate)

    self.drop_path = DropPath(drop_path)

  def forward(self, input: torch.Tensor):
    x = input + self.drop_path(self.self_attention(self.ln_1(input)))
    x = x + self.drop_path(self.mlp(self.ln_2(x)))
    return x

class VSSBlock3D_v6(nn.Module): #no multiplicative path, added MLP. more like transformer block used in TABSurfer now
  def __init__(
      self,
      hidden_dim: int = 0,
      drop_path: float = 0,
      norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
      attn_drop_rate: float = 0,
      d_state: int = 16,
      expansion_factor = 1, # can only be 1 for v3, no linear projection to increase channels
      mlp_drop_rate=0.,
      orientation = 0,
      scan_type = 'scan',
      size_list = (16, 16, 8),
      **kwargs,
      ):
    super().__init__()
    print(orientation, end='')
    self.ln_1 = norm_layer(hidden_dim)
    self.self_attention = SS3D_v6(d_model=hidden_dim, 
                                  dropout=attn_drop_rate, 
                                  d_state=d_state, 
                                  expand=expansion_factor, 
                                  orientation=orientation, 
                                  scan_type=scan_type, 
                                  size_list=size_list,
                                  **kwargs)

    self.ln_2 = norm_layer(hidden_dim)
    self.mlp = FeedForward(dim = hidden_dim, hidden_dim=expansion_factor*hidden_dim, dropout_rate = mlp_drop_rate)

    self.drop_path = DropPath(drop_path)

  def forward(self, input: torch.Tensor):
    x = input + self.drop_path(self.self_attention(self.ln_1(input)))
    x = x + self.drop_path(self.mlp(self.ln_2(x)))
    return x
  
class VSSLayer3D(nn.Module):
    """ A basic layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        depth,
        attn_drop=0.,
        mlp_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        d_state=64,
        version = 'v5', #None, v5, v6
        expansion_factor = 1,
        scan_type = 'scan',
        orientation_order = None,
        size_list = (16, 16, 8),
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        if version is None:
            print('Vanilla VSS')
            self.blocks = nn.ModuleList([
                VSSBlock3D(
                    hidden_dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                    expansion_factor=expansion_factor,
                )
                for i in range(depth)])
        elif version =='v5':
            print('VSS version 5:')
            if orientation_order is None:
                self.blocks = nn.ModuleList([
                    VSSBlock3D_v5(
                        hidden_dim=dim,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer,
                        attn_drop_rate=attn_drop,
                        d_state=d_state,
                        expansion_factor=expansion_factor,
                        mlp_drop_rate=mlp_drop,
                        scan_type=scan_type,
                        size_list = size_list,
                        orientation=i%6, # 0 1 2 3 4 5 6 7 8 > 0 1 2 3 4 5 0 1 2
                    )
                    for i in range(depth)])
            else:
                self.blocks = nn.ModuleList([
                    VSSBlock3D_v5(
                        hidden_dim=dim,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer,
                        attn_drop_rate=attn_drop,
                        d_state=d_state,
                        expansion_factor=expansion_factor,
                        mlp_drop_rate=mlp_drop,
                        scan_type=scan_type,
                        size_list=size_list,
                        orientation=i%6, # 0 1 2 3 4 5 6 7 8 > 0 1 2 3 4 5 0 1 2
                    )
                    for i in orientation_order])
            print()
        elif version =='v6':
            print('VSS version 6:')
            if orientation_order is None:
                self.blocks = nn.ModuleList([
                    VSSBlock3D_v6(
                        hidden_dim=dim,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer,
                        attn_drop_rate=attn_drop,
                        d_state=d_state,
                        expansion_factor=expansion_factor,
                        mlp_drop_rate=mlp_drop,
                        scan_type=scan_type,
                        size_list = size_list,
                        orientation=i%8, # 0 1 2 3 4 5 6 7 8 > 0 1 2 3 4 5 0 1 2
                    )
                    for i in range(depth)])
            else:
                self.blocks = nn.ModuleList([
                    VSSBlock3D_v6(
                        hidden_dim=dim,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer,
                        attn_drop_rate=attn_drop,
                        d_state=d_state,
                        expansion_factor=expansion_factor,
                        mlp_drop_rate=mlp_drop,
                        scan_type=scan_type,
                        size_list=size_list,
                        orientation=i%8, # 0 1 2 3 4 5 6 7 8 > 0 1 2 3 4 5 0 1 2
                    )
                    for i in orientation_order])
            print()
        else:
            raise Exception('define a valid VSS version')
            

        if True:
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x