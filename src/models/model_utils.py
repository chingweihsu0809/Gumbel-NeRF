from argparse import Namespace

import torch
from torch import nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from src.models.switch_model.nerf import ShiftedSoftplus
from src.models.switch_model.nerf_moe import get_nerf_moe_inner
from src.models.gumbel_model.gumbelnerf import GumbelNeRFBig

def convert_to_seqexperts(state_dict):
    keys = list(state_dict.keys())
    for key in keys:
        if "layers.0.experts.0." in key:
            if "weight" in key or "bias" in key:
                para_type = "weight" if "weight" in key else "bias"
                layer_id = int(key[-1])
                v = state_dict.pop(key)
                v = torch.unbind(v, dim=0)
                for expert_id, expert_v in enumerate(v):
                    new_key = f'module.layers.0.experts.0.experts.{expert_id}.layers.{layer_id}.{para_type}'
                    if para_type == "weight":
                        new_v = expert_v.t().contiguous()
                    if para_type == "bias":
                        new_v = expert_v.squeeze(0)
                    state_dict[new_key] = new_v
    return state_dict

def convert_to_seqexperts1(state_dict, moe_layer_num):
    keys = list(state_dict.keys())
    for key in keys:
        for moe_layer_id in range(moe_layer_num):
            if f"layers.{moe_layer_id}.experts.0." in key:
                if "weight" in key or "bias" in key:
                    para_type = "weight" if "weight" in key else "bias"
                    layer_id = int(key[-1])
                    v = state_dict.pop(key)
                    v = torch.unbind(v, dim=0)
                    for expert_id, expert_v in enumerate(v):
                        new_key = f'module.layers.{moe_layer_id}.experts.0.experts.{expert_id}.layers.{layer_id}.{para_type}'
                        if para_type == "weight":
                            new_v = expert_v.t().contiguous()
                        if para_type == "bias":
                            new_v = expert_v.squeeze(0)
                        state_dict[new_key] = new_v
    return state_dict


def convert_to_seqexperts2(state_dict, moe_layer_ids):
    keys = list(state_dict.keys())
    for key in keys:
        for moe_layer_id in moe_layer_ids:
            if f"layers.{moe_layer_id}.experts.0." in key:
                if "weight" in key or "bias" in key:
                    para_type = "weight" if "weight" in key else "bias"
                    layer_id = int(key[-1])
                    v = state_dict.pop(key)
                    v = torch.unbind(v, dim=0)
                    for expert_id, expert_v in enumerate(v):
                        new_key = f'module.layers.{moe_layer_id}.experts.0.experts.{expert_id}.layers.{layer_id}.{para_type}'
                        if para_type == "weight":
                            new_v = expert_v.t().contiguous()
                        if para_type == "bias":
                            new_v = expert_v.squeeze(0)
                        state_dict[new_key] = new_v
    return state_dict

def get_nerf(hparams: Namespace, appearance_count: int) -> nn.Module:
    return _get_nerf_inner(hparams, appearance_count, 3, 'model_state_dict')


def _get_nerf_inner(hparams: Namespace, appearance_count: int, xyz_dim: int, weight_key: str) -> nn.Module:
    if hparams.use_gumbel:
        import json
        from pathlib import Path
        with open(hparams.gumbel_config) as f:
            modelargs = json.load(f)['model_hparams']
        if modelargs["latent"] is not None:
            cars_id = sorted(list((Path(hparams.dataset_path) / hparams.latent_src).iterdir()))
            modelargs["latent"].update({"size": len(cars_id),
                                        "init": hparams.latent_init})    
        nerf = GumbelNeRFBig(**modelargs)
    elif hparams.use_moe:
        if weight_key == "model_state_dict":
            model_cfg_name = "switch_model"
        else:
            model_cfg_name = None
            raise NotImplementedError
        nerf = get_nerf_moe_inner(hparams, appearance_count, xyz_dim, model_cfg_name=model_cfg_name)

    if hparams.ckpt_path is not None:
        state_dict = torch.load(hparams.ckpt_path, map_location='cpu')[weight_key]
        if hparams.latent_init == "train_avg":
            print(torch.mean(state_dict['module.latent_net.shape_codes.weight'], dim=0).shape)
            print(torch.mean(state_dict['module.latent_net.shape_codes.weight'], dim=0).reshape(1,-1).shape)
            print(torch.mean(state_dict['module.latent_net.shape_codes.weight'], dim=0).reshape(1,-1).repeat(modelargs["latent"]["size"], 1).shape)
            nerf.latent_net.shape_codes.weight = nn.Parameter(torch.mean(state_dict['module.latent_net.shape_codes.weight'], dim=0).reshape(1,-1).repeat(modelargs["latent"]["size"], 1))
            nerf.latent_net.texture_codes.weight = nn.Parameter(torch.mean(state_dict['module.latent_net.texture_codes.weight'], dim=0).reshape(1,-1).repeat(modelargs["latent"]["size"], 1))

        if hparams.expertmlp2seqexperts and hparams.use_moe:
            if getattr(hparams, "moe_layer_num", 1) > 1:
                state_dict = convert_to_seqexperts1(state_dict, hparams.moe_layer_num)
            elif getattr(hparams, "moe_layer_ids", None) is not None:
                state_dict = convert_to_seqexperts2(state_dict, hparams.moe_layer_ids)
            else:
                state_dict = convert_to_seqexperts(state_dict)

        consume_prefix_in_state_dict_if_present(state_dict, prefix='module.')

        model_dict = nerf.state_dict()
        if not hparams.load_latent:
            state_dict = {k:v for k, v in state_dict.items() if "latent_net" not in k}
        model_dict.update(state_dict)
        nerf.load_state_dict(model_dict)
        
    return nerf


from math import cos, pi
class CosineScheduler:
    def __init__(self, eta_max, eta_min, T_max, T_init = 0) -> None:
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.T_max   = T_max
        self.T_init = T_init
        self.t_cur = 0
        self.eta_t = eta_max
        
    def step(self):
        if self.t_cur < self.T_max:
            self.eta_t = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1 + cos(pi*(self.t_cur-self.T_init)/(self.T_max - self.T_init)))
        else:
            self.eta_t = self.eta_min 
        self.t_cur += 1
        