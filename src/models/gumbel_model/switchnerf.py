import torch
import torch.nn as nn
import sys
import copy
from .moe import MoE
from .common import RGBHead

def PE(x, degree):
    y = torch.cat([2.**i * x for i in range(degree)], -1)
    w = 1
    return torch.cat([x] + [torch.sin(y) * w, torch.cos(y) * w], -1)


class SwitchNeRF(nn.Module):
    def __init__(self, 
                 enc_xyz_size = 256,
                 W = 256, 
                 num_xyz_freq = 10, 
                 num_dir_freq = 4,
                 latent = None,
                 moe=None):
        super().__init__()
        self.num_xyz_freq = num_xyz_freq
        self.num_dir_freq = num_dir_freq
        d_xyz, d_viewdir = 3 + 6 * num_xyz_freq, 3 + 6 * num_dir_freq
        self.latent_dim  = latent["size"] if latent is not None else 0

        self.num_experts = moe["num_experts"]
        
        self.encoding_xyz = nn.Sequential(nn.Linear(d_xyz, enc_xyz_size))
        moe.update({"latent_dim":self.latent_dim, "enc_xyz_size": enc_xyz_size})
        self.moe_model = MoE(**moe)
        self.sigma = nn.Sequential(nn.Linear(W,1), nn.Softplus())
        
        self.rgb = RGBHead(W, d_viewdir, self.latent_dim)
        
        
    def forward(self, xyz, viewdir, shape_latent, texture_latent, temperature):
        nrays, nsamples, xyz_dim = xyz.shape
        xyz = xyz.view([-1, xyz_dim])
        xyz = PE(xyz, self.num_xyz_freq)
        
        y = self.encoding_xyz(xyz)
        shape_out, aux_loss, gates_soft, num_pts = \
            self.moe_model(y, shape_latent)
        sigmas = self.sigma(shape_out)
        mean_sigma = sigmas.mean(0)
        
        index = gates_soft.max(dim=-1, keepdim=True)[1]
        gates_hard = torch.zeros_like(gates_soft, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
        gates_hard = gates_hard - gates_soft.detach() + gates_soft
                
        _, _, viewdir_dim = viewdir.shape
        viewdir = viewdir.view([-1, viewdir_dim])
        viewdir = PE(viewdir, self.num_dir_freq)
        
        rgbs = self.rgb(shape_out, viewdir, texture_latent if self.latent_dim else None)
        
        ## reshape results
        rgbs = rgbs.view([nrays, nsamples, -1])
        sigmas = sigmas.view([nrays, nsamples, -1])
        gates_soft = gates_soft.view([nrays, nsamples, -1])
        gates_hard = gates_hard.view([nrays, nsamples, -1])
        
        extras = {
            "mean_sigma": mean_sigma,
            "num_pts"    : num_pts,
            "aux_loss"   : aux_loss
        }
        return sigmas, rgbs, gates_soft, gates_hard, extras