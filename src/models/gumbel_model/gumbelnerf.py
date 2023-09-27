import torch
import torch.nn as nn
import sys
import copy

def PE(x, degree):
    y = torch.cat([2.**i * x for i in range(degree)], -1)
    w = 1
    return torch.cat([x] + [torch.sin(y) * w, torch.cos(y) * w], -1)

class GumbelNeRFCommon(nn.Module):
    def __init__(self):
        super().__init__()
           
    def forward(self, x: torch.Tensor, sigma_noise = None, temperature = 0.166667):
        
        xyz = x[:, :3]
        viewdir = x[:, 3:6]
        instance_ids = None if self.latent_dim == 0 else x[:, 6].long()
            
        xyz = PE(xyz, self.num_xyz_freq)
        viewdir = PE(viewdir, self.num_dir_freq)
        
        ## encode xyz
        y = self.encoding_xyz(xyz)
        
        ## instance latent -> part latents
        if instance_ids is not None:
            shape_latent, color_latent = self.latent_net()
            shape_latent = shape_latent[instance_ids]
            color_latent = color_latent[instance_ids]
        
        ## shape block
        sigma_out, shape_out = [], []
        for i in range(self.num_experts):
            if instance_ids is not None:
                _shape_out = getattr(self, f"shapeExpert_{i}")(y, shape_latent[:, i, :])
            else:
                _shape_out = getattr(self, f"shapeExpert_{i}")(y)
            shape_out.append(_shape_out)
            _sigma = self.sigma(_shape_out)
            sigma_out.append(_sigma)
        
        ## gumbel-maxpool
        # add gumbel noise
        sigmas = torch.cat(sigma_out, 1)
        # logits = self.sigmoid(sigmas) / temperature
        ## 0628
        # logits = nn.functional.normalize(sigmas, p=2, dim=1) / temperature
        ## 0820
        # logits = nn.functional.normalize((sigmas)**(1/temperature), p=1, dim=1)
        # log_logits = torch.log(logits+1e-16)
        ## 0820_v2
        log_logits = torch.log_softmax(torch.log(sigmas + 1e-10)/temperature, dim=-1)
        logits = torch.exp(log_logits)
        
        gates_soft = nn.functional.gumbel_softmax(log_logits)
        index = gates_soft.max(dim=-1, keepdim=True)[1]
        gates_hard = torch.zeros_like(sigmas, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
        gates_hard = gates_hard - gates_soft.detach() + gates_soft
        # maxpooling and retrieve latent
        sigmas_pooled = torch.gather(sigmas, -1, index)
        shape_out_comb = torch.zeros_like(shape_out[0])
        for i in range(self.num_experts):
            shape_out_comb[gates_hard[:, i]==1] = shape_out[i][gates_hard[:, i]==1]
            
        ## rgb block
        if self.unified_head:
            rgbs = self.rgbExpert(shape_out_comb, viewdir, color_latent if instance_ids is not None else None)
        else:
            rgbs = torch.zeros([sigmas.shape[0], 3]).to(sigmas.device)
            for i in range(self.num_experts):
                if instance_ids is not None:
                    rgbs += getattr(self, f"rgbExpert_{i}")(shape_out[i], viewdir, color_latent[:, i, :]) * gates_hard[:, [i]]
                else:
                    rgbs += getattr(self, f"rgbExpert_{i}")(shape_out[i], viewdir) * gates_hard[:, [i]]
        
        num_pts = [torch.sum(index==i) for i in range(self.num_experts)]
        orig_num_pts = [torch.sum(logits.max(dim=-1, keepdim=True)[1]==i) for i in range(self.num_experts)]
        
        extras = {
            "num_pts"    : num_pts
        }
        return {
            "outputs": torch.cat([rgbs, sigmas_pooled], -1),
            "extras": {
                # "logits": nn.functional.normalize(sigmas, p=1, dim=1),
                "logits": logits,
                "gates_hard": gates_hard,
                "gates_soft": gates_soft,
                "expert_sigma": sigma_out  # [(N_pts)] * #of experts
            }
        }


class GumbelNeRFBig(GumbelNeRFCommon):
    def __init__(self, 
                 enc_xyz_size = 256,
                 W = 256, 
                 num_xyz_freq = 10, 
                 num_dir_freq = 4, 
                 latent = None,
                 num_experts = 4,
                 uni_head=False,
                 moe=None,
                 rgb=None,
                 codenerfstyle=True,
                 latent_fc=True
                 ):
        super().__init__()
        
        if latent_fc:
            from .common import LatentNet
        else:
            from .common import LatentNet_nofc as LatentNet
        if codenerfstyle:
            from .common import CodeNeRFMLP as MLP, CodeNeRFRGBHead as RGBHead
        else:
            from .common import MLP, RGBHead
        
        
        self.num_xyz_freq = num_xyz_freq
        self.num_dir_freq = num_dir_freq
        self.num_experts = num_experts
        self.latent_dim  = latent["len"] if latent is not None else 0
        
        ## encode xyz
        d_xyz, d_viewdir = 3 + 6 * num_xyz_freq, 3 + 6 * num_dir_freq
        self.encoding_xyz = nn.Sequential(nn.Linear(d_xyz, enc_xyz_size)) 
        
        ## instance code -> part codes
        if latent is not None:
            if "latent_map_type" in latent:
                self.latent_net = LatentNet(num_experts, latent["size"], self.latent_dim, uni_head, latent["init"], latent["latent_map_type"])
            else:
                self.latent_net = LatentNet(num_experts, latent["size"], self.latent_dim, uni_head, latent["init"])
        
        ## shape mlps
        moe.update({"latent_dim": (self.latent_net.shapefc.weight.shape[0] // num_experts) if latent_fc \
                                    else self.latent_net.part_code_len}) ## part_code_len
        for i in range(self.num_experts):
            setattr(self, f"shapeExpert_{i}", MLP(**moe))
        
        ## gumbel-maxpool
        self.maxpool = torch.nn.MaxPool1d(self.num_experts, return_indices=True)
        self.sigma = nn.Sequential(nn.Linear(W,1), nn.Softplus())
        
        ## rgb mlp
        rgb.update({"d_viewdir": d_viewdir,
                    "latent_dim": self.latent_dim})
        self.unified_head = uni_head
        if self.unified_head:
            self.rgbExpert = RGBHead(**rgb)
        else:
            for i in range(self.num_experts):
                setattr(self, f"rgbExpert_{i}", RGBHead(**rgb))
        