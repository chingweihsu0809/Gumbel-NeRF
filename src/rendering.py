import os
from argparse import Namespace
from typing import Optional, Dict, Callable, Tuple

import torch
import torch.nn.functional as F
from torch import nn

TO_COMPOSITE = {'rgb', 'depth'}
INTERMEDIATE_KEYS = {'zvals_coarse', 'raw_rgb_coarse', 'raw_sigma_coarse', 'depth_real_coarse'}


def render_rays(nerf: nn.Module,
                rays: torch.Tensor,
                image_indices: Optional[torch.Tensor],
                instance_ids: Optional[torch.Tensor],
                hparams: Namespace,
                get_depth: bool,
                get_depth_variance: bool,
                get_bg_fg_rgb: bool,
                nerf_kwargs: Optional[Dict]=None) -> Tuple[Dict[str, torch.Tensor], bool]:
    N_rays = rays.shape[0]

    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)
    if image_indices is not None:
        image_indices = image_indices.unsqueeze(-1).unsqueeze(-1)
    if instance_ids is not None:
        instance_ids = instance_ids.unsqueeze(-1).unsqueeze(-1)

    perturb = hparams.perturb if nerf.training else 0
    last_delta = 1e10 * torch.ones(N_rays, 1, device=rays.device)

    rays_o = rays_o.view(rays_o.shape[0], 1, rays_o.shape[1])
    rays_d = rays_d.view(rays_d.shape[0], 1, rays_d.shape[1])

    # Sample depth points
    z_steps = torch.linspace(0, 1, hparams.coarse_samples, device=rays.device)  # (N_samples)
    z_vals = near * (1 - z_steps) + far * z_steps

    z_vals = _expand_and_perturb_z_vals(z_vals, hparams.coarse_samples, perturb, N_rays)

    xyz_coarse = rays_o + rays_d * z_vals.unsqueeze(-1)
    
    results = _get_results(nerf=nerf,
                           rays_d=rays_d,
                           image_indices=image_indices,
                           instance_ids=instance_ids,
                           hparams=hparams,
                           xyz_coarse=xyz_coarse,
                           z_vals=z_vals,
                           last_delta=last_delta,
                           get_depth=get_depth,
                           get_depth_variance=get_depth_variance,
                           flip=False,
                           depth_real=None,
                           xyz_fine_fn=lambda fine_z_vals: (rays_o + rays_d * fine_z_vals.unsqueeze(-1), None),
                           nerf_kwargs=nerf_kwargs)

    return results


def _get_results(nerf: nn.Module,
                 rays_d: torch.Tensor,
                 image_indices: Optional[torch.Tensor],
                 instance_ids: Optional[torch.Tensor],
                 hparams: Namespace,
                 xyz_coarse: torch.Tensor,
                 z_vals: torch.Tensor,
                 last_delta: torch.Tensor,
                 get_depth: bool,
                 get_depth_variance: bool,
                 flip: bool,
                 depth_real: Optional[torch.Tensor],
                 xyz_fine_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]],
                 nerf_kwargs: Optional[Dict]=None) \
        -> Dict[str, torch.Tensor]:
    results = {}

    last_delta_diff = torch.zeros_like(last_delta)
    last_delta_diff[last_delta.squeeze() < 1e10, 0] = z_vals[last_delta.squeeze() < 1e10].max(dim=-1)[0]

    _inference(results=results,
               typ='coarse',
               nerf=nerf,
               rays_d=rays_d,
               image_indices=image_indices,
               instance_ids=instance_ids,
               hparams=hparams,
               xyz=xyz_coarse,
               z_vals=z_vals,
               last_delta=last_delta - last_delta_diff,
               composite_rgb=hparams.fine_samples == 0,
               get_depth=hparams.fine_samples == 0 and get_depth,
               get_depth_variance=hparams.fine_samples == 0 and get_depth_variance,
               get_weights=hparams.fine_samples > 0,
               flip=flip,
               depth_real=depth_real,
               white_bkgd=hparams.white_bkgd,
               nerf_kwargs=nerf_kwargs)

    if hparams.fine_samples > 0:  # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
        perturb = hparams.perturb if nerf.training else 0
        fine_z_vals = _sample_pdf(z_vals_mid, results['weights_coarse'][:, 1:-1].detach(),
                                  hparams.fine_samples // 2 if flip else hparams.fine_samples, det=(perturb == 0))

        del results['weights_coarse']

        xyz_fine, depth_real_fine = xyz_fine_fn(fine_z_vals)
        last_delta_diff = torch.zeros_like(last_delta)
        last_delta_diff[last_delta.squeeze() < 1e10, 0] = fine_z_vals[last_delta.squeeze() < 1e10].max(dim=-1)[0]

        _inference(results=results,
                   typ='fine',
                   nerf=nerf,
                   rays_d=rays_d,
                   image_indices=image_indices,
                   instance_ids=instance_ids,
                   hparams=hparams,
                   xyz=xyz_fine,
                   z_vals=fine_z_vals,
                   last_delta=last_delta - last_delta_diff,
                   composite_rgb=True,
                   get_depth=get_depth,
                   get_depth_variance=get_depth_variance,
                   get_weights=False,
                   flip=flip,
                   depth_real=depth_real_fine,
                   white_bkgd=hparams.white_bkgd,
                   nerf_kwargs=nerf_kwargs)

        for key in INTERMEDIATE_KEYS:
            if key in results:
                del results[key]

    return results


def _inference(results: Dict[str, torch.Tensor],
               typ: str,
               nerf: nn.Module,
               rays_d: torch.Tensor,
               image_indices: Optional[torch.Tensor],
               instance_ids: Optional[torch.Tensor],
               hparams: Namespace,
               xyz: torch.Tensor,
               z_vals: torch.Tensor,
               last_delta: torch.Tensor,
               composite_rgb: bool,
               get_depth: bool,
               get_depth_variance: bool,
               get_weights: bool,
               flip: bool,
               depth_real: Optional[torch.Tensor],
               white_bkgd=False,
               nerf_kwargs=None):
    N_rays_ = xyz.shape[0]
    N_samples_ = xyz.shape[1]

    if hparams.return_pts:
        results[f"pts_{typ}"] = xyz

    # Otherwise this will get sorted in the proper order anyways
    if flip and 'zvals_coarse' not in results:
        xyz = torch.flip(xyz, dims=[-2, ])
        z_vals = torch.flip(z_vals, dims=[-1, ])

    xyz_ = xyz.view(-1, xyz.shape[-1])

    # Perform model inference to get rgb and raw sigma
    B = xyz_.shape[0]
    out_chunks = []
    rays_d_ = rays_d.repeat(1, N_samples_, 1).view(-1, rays_d.shape[-1])

    assert (image_indices is None) or (instance_ids is None)
    if image_indices is not None:
        image_indices_ = image_indices.repeat(1, N_samples_, 1).view(-1, 1)
    if instance_ids is not None:
        instance_ids_ = instance_ids.repeat(1, N_samples_, 1).view(-1, 1)


    # (N_rays*N_samples_, embed_dir_channels)
    for i in range(0, B, hparams.model_chunk_size):
        xyz_chunk = xyz_[i:i + hparams.model_chunk_size]

        if image_indices is not None:
            xyz_chunk = torch.cat([xyz_chunk,
                                    rays_d_[i:i + hparams.model_chunk_size],
                                    image_indices_[i:i + hparams.model_chunk_size]], 1)
        elif instance_ids is not None:
            xyz_chunk = torch.cat([xyz_chunk,
                                    rays_d_[i:i + hparams.model_chunk_size],
                                    instance_ids_[i:i + hparams.model_chunk_size]], 1)
        else:
            xyz_chunk = torch.cat([xyz_chunk, rays_d_[i:i + hparams.model_chunk_size]], 1)

        if hparams.use_sigma_noise:
            if hparams.sigma_noise_std > 0.0:
                sigma_noise = torch.randn(len(xyz_chunk), 1, device=xyz_chunk.device) * hparams.sigma_noise_std if nerf.training else None
            else:
                sigma_noise = None
        else:
            sigma_noise = None

        if nerf_kwargs is not None:
            model_chunk = nerf(xyz_chunk, sigma_noise=sigma_noise, **nerf_kwargs)
        else:
            model_chunk = nerf(xyz_chunk, sigma_noise=sigma_noise)

        out_chunks += [model_chunk]
    
    if isinstance(out_chunks[0], dict):
        assert "extras" in out_chunks[0]
        
        if hparams.use_moe:
            # assert "moe_loss" in out_chunks[0]["extras"]
            if "moe_loss" in out_chunks[0]["extras"]:
                gate_loss = [i["extras"]["moe_loss"] for i in out_chunks]
                results[f'gate_loss_{typ}'] = torch.cat(gate_loss, 0)

            if hparams.moe_return_gates:
                moe_gates = [torch.stack(i["extras"]["moe_gates"], dim=1) for i in out_chunks]
                moe_gates = torch.cat(moe_gates, 0) # points, layer_num, top k
                results[f'moe_gates_{typ}'] = moe_gates.view(N_rays_, N_samples_, moe_gates.shape[1], moe_gates.shape[2])

            if hparams.use_load_importance_loss and hparams.compute_balance_loss:
                balance_loss = [i["extras"]["balance_loss"] for i in out_chunks]
                results[f'balance_loss_{typ}'] = torch.cat(balance_loss, 0)
                
        out = [i["outputs"] for i in out_chunks]
    else:
        # out = [i["outputs"] for i in out_chunks]
        out = out_chunks
    
    out = torch.cat(out, 0)
    out = out.view(N_rays_, N_samples_, out.shape[-1])

    if "gates_hard" in out_chunks[0]["extras"]:
        gates_hard = [i["extras"]["gates_hard"] for i in out_chunks]
        gates_hard = torch.cat(gates_hard, 0)
        gates_hard = gates_hard.view(N_rays_, N_samples_, gates_hard.shape[-1])
        results[f"gates_hard_{typ}"] = gates_hard
    if "gates_soft" in out_chunks[0]["extras"]:
        gates_soft = [i["extras"]["gates_soft"] for i in out_chunks]
        gates_soft = torch.cat(gates_soft, 0)
        gates_soft = gates_soft.view(N_rays_, N_samples_, gates_soft.shape[-1])
        results[f"gates_soft_{typ}"] = gates_soft
    if "logits" in out_chunks[0]["extras"]:
        logits = [i["extras"]["logits"] for i in out_chunks]
        logits = torch.cat(logits, 0)
        logits = logits.view(N_rays_, N_samples_, logits.shape[-1])
        results[f"logits_{typ}"] = logits

    rgbs = out[..., :3]  # (N_rays, N_samples_, 3)
    sigmas = out[..., 3]  # (N_rays, N_samples_)
    if hparams.return_sigma:
        results[f'sigma_{typ}'] = sigmas
        
    if hparams.return_pts_rgb:
        results[f"pts_rgb_{typ}"] = rgbs

    if 'zvals_coarse' in results:
        # combine coarse and fine samples
        z_vals, ordering = torch.sort(torch.cat([z_vals, results['zvals_coarse']], -1), -1, descending=flip)
        rgbs = torch.cat((
            torch.gather(torch.cat((rgbs[..., 0], results['raw_rgb_coarse'][..., 0]), 1), 1, ordering).unsqueeze(
                -1),
            torch.gather(torch.cat((rgbs[..., 1], results['raw_rgb_coarse'][..., 1]), 1), 1, ordering).unsqueeze(
                -1),
            torch.gather(torch.cat((rgbs[..., 2], results['raw_rgb_coarse'][..., 2]), 1), 1, ordering).unsqueeze(-1)
        ), -1)
        sigmas = torch.gather(torch.cat((sigmas, results['raw_sigma_coarse']), 1), 1, ordering)
        
        
        if 'gates_hard_coarse' in results:
            gates_hard = torch.cat(tuple(
                torch.gather(torch.cat((gates_hard[..., k], results['gates_hard_coarse'][..., k]), 1), 1, ordering).unsqueeze(-1) for k in range(hparams.expert_num)
            ), -1)
            
        if 'gates_soft_coarse' in results:
            gates_soft = torch.cat(tuple(
                torch.gather(torch.cat((gates_soft[..., k], results['gates_soft_coarse'][..., k]), 1), 1, ordering).unsqueeze(-1) for k in range(hparams.expert_num)
            ), -1)

        if 'logits_coarse' in results:
            logits = torch.cat(tuple(
                torch.gather(torch.cat((logits[..., k], results['logits_coarse'][..., k]), 1), 1, ordering).unsqueeze(-1) for k in range(hparams.expert_num)
            ), -1)
            
        if depth_real is not None:
            depth_real = torch.gather(torch.cat((depth_real, results['depth_real_coarse']), 1), 1,
                                      ordering)

    # Convert these values using volume rendering (Section 4)
    if flip:
        deltas = z_vals[..., :-1] - z_vals[..., 1:]
    else:
        deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)

    deltas = torch.cat([deltas, last_delta], -1)  # (N_rays, N_samples_)
    alphas = 1 - torch.exp(-deltas * sigmas)  # (N_rays, N_samples_)
    if hparams.return_alpha:
        results[f'alpha_{typ}'] = alphas

    if hparams.return_pts_alpha:
        if 'zvals_coarse' in results:
            # recover the alpha of fine points
            pts_alpha = torch.zeros_like(alphas)
            pts_alpha.scatter_(dim=1, index=ordering, src=alphas)
            results[f"pts_alpha_{typ}"] = pts_alpha[:, 0:N_samples_]
        else:
            results[f"pts_alpha_{typ}"] = alphas

    T = torch.cumprod(1 - alphas + 1e-8, -1)
    T = torch.cat((torch.ones_like(T[..., 0:1]), T[..., :-1]), dim=-1)  # [..., N_samples]

    weights = alphas * T  # (N_rays, N_samples_)
    
    if get_weights:
        results[f'weights_{typ}'] = weights

    if composite_rgb:
        results[f'rgb_{typ}'] = (weights.unsqueeze(-1) * rgbs).sum(dim=1)  # n1 n2 c -> n1 c
        if white_bkgd:
            acc_map = torch.sum(weights, -1)
            results[f'rgb_{typ}'] = results[f'rgb_{typ}'] + (1.-acc_map[...,None])
            results[f'acc_map_{typ}'] = acc_map
        elif hparams.use_random_background_color:
            acc_map = torch.sum(weights, -1) # N_rays
            background_color = torch.rand([3], dtype=acc_map.dtype, device=acc_map.device)
            results[f'rgb_{typ}'] = results[f'rgb_{typ}'] + (1.-acc_map[...,None]) * background_color
            
        if f"gates_hard_{typ}" in results:
            for k in range(gates_hard.shape[-1]):
                alpha_exp = alphas * gates_hard[:, :, k]
                T_exp = torch.cumprod(1 - alpha_exp + 1e-8, -1)
                T_exp = torch.cat((torch.ones_like(T_exp[..., 0:1]), T_exp[..., :-1]), dim=-1)  # [..., N_samples]
                weights_exp = alpha_exp * T_exp
                
                results[f'rgb_{typ}_exp{k}'] = (weights_exp.unsqueeze(-1) * rgbs).sum(dim=1)
                if white_bkgd:
                    acc_map = torch.sum(weights_exp * gates_hard[:, :, k], -1)
                    results[f'rgb_{typ}_exp{k}'] = results[f'rgb_{typ}_exp{k}'] + (1.-acc_map[...,None])
    else:
        results[f'zvals_{typ}'] = z_vals
        results[f'raw_rgb_{typ}'] = rgbs
        results[f'raw_sigma_{typ}'] = sigmas
        if depth_real is not None:
            results[f'depth_real_{typ}'] = depth_real

    with torch.no_grad():
        if get_depth or get_depth_variance:
            if depth_real is not None:
                depth_map = (weights * depth_real).sum(dim=1)  # n1 n2 -> n1
            else:
                depth_map = (weights * z_vals).sum(dim=1)  # n1 n2 -> n1

        if get_depth:
            results[f'depth_{typ}'] = depth_map

        if get_depth_variance:
            results[f'depth_variance_{typ}'] = (weights * (z_vals - depth_map.unsqueeze(1)).square()).sum(
                axis=-1)


def _intersect_sphere(rays_o: torch.Tensor, rays_d: torch.Tensor, sphere_center: torch.Tensor,
                      sphere_radius: torch.Tensor) -> torch.Tensor:
    if sphere_radius is not None:
        rays_o = (rays_o - sphere_center) / sphere_radius
        rays_d = rays_d / sphere_radius

    '''
    rays_o, rays_d: [..., 3]
    compute the depth of the intersection point between this ray and unit sphere
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(rays_d * rays_o, dim=-1) / torch.sum(rays_d * rays_d, dim=-1)
    p = rays_o + d1.unsqueeze(-1) * rays_d
    # consider the case where the ray does not intersect the sphere
    ray_d_cos = 1. / torch.norm(rays_d, dim=-1)
    p_norm_sq = torch.sum(p * p, dim=-1)
    if (p_norm_sq >= 1.).any():
        raise Exception(
            'Not all your cameras are bounded by the unit sphere; please make sure the cameras are normalized properly!')
    d2 = torch.sqrt(1. - p_norm_sq) * ray_d_cos

    return d1 + d2


def _depth2pts_outside(rays_o: torch.Tensor, rays_d: torch.Tensor, depth: torch.Tensor, sphere_center: torch.Tensor,
                       sphere_radius: torch.Tensor, include_xyz_real: bool):
    '''
    rays_o, rays_d: [..., 3]
    depth: [...]; inverse of distance to sphere origin
    '''
    rays_o_orig = rays_o
    rays_d_orig = rays_d
    if sphere_radius is not None:
        rays_o = (rays_o - sphere_center) / sphere_radius
        rays_d = rays_d / sphere_radius

    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(rays_d * rays_o, dim=-1) / torch.sum(rays_d * rays_d, dim=-1)
    p_mid = rays_o + d1.unsqueeze(-1) * rays_d
    p_mid_norm = torch.norm(p_mid, dim=-1)
    ray_d_norm = rays_d.norm(dim=-1)
    ray_d_cos = 1. / ray_d_norm
    d2 = torch.sqrt(1. - p_mid_norm * p_mid_norm) * ray_d_cos
    p_sphere = rays_o + (d1 + d2).unsqueeze(-1) * rays_d

    rot_axis = torch.cross(rays_o, p_sphere, dim=-1)
    rot_axis = rot_axis / (torch.norm(rot_axis, dim=-1, keepdim=True) + 1e-8)
    phi = torch.asin(p_mid_norm)
    theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
    rot_angle = (phi - theta).unsqueeze(-1)  # [..., 1]

    # now rotate p_sphere
    # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                   torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                   rot_axis * torch.sum(rot_axis * p_sphere, dim=-1, keepdim=True) * (1. - torch.cos(rot_angle))
    p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)

    # now calculate conventional depth
    depth_real = 1. / (depth + 1e-8) * torch.cos(theta) + d1

    if include_xyz_real:
        boundary = rays_o_orig + rays_d_orig * (d1 + d2).unsqueeze(-1)
        pts = torch.cat((boundary.repeat(1, p_sphere_new.shape[1], 1), p_sphere_new, depth.unsqueeze(-1)), dim=-1)

    else:
        pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

    return pts, depth_real


def _expand_and_perturb_z_vals(z_vals, samples, perturb, N_rays):
    z_vals = z_vals.expand(N_rays, samples)
    if perturb > 0:  # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    return z_vals


def _sample_pdf(bins: torch.Tensor, weights: torch.Tensor, fine_samples: int, det: bool) -> torch.Tensor:
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        fine_samples: the number of samples to draw from the distribution
        det: deterministic or not
    Outputs:
        samples: the sampled samples
    """
    weights = weights + 1e-8  # prevent division by zero (don't do inplace op!)

    pdf = weights / weights.sum(-1).unsqueeze(-1)  # (N_rays, N_samples_)

    cdf = torch.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
    return _sample_cdf(bins, cdf, fine_samples, det)


def _sample_cdf(bins: torch.Tensor, cdf: torch.Tensor, fine_samples: int, det: bool) -> torch.Tensor:
    N_rays, N_samples_ = cdf.shape

    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
    # padded to 0~1 inclusive
    if det:
        u = torch.linspace(0, 1, fine_samples, device=bins.device)
        u = u.expand(N_rays, fine_samples)
    else:
        u = torch.rand(N_rays, fine_samples, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1)
    inds_sampled = inds_sampled.view(inds_sampled.shape[0], -1)  # n1 n2 2 -> n1 (n2 2)

    cdf_g = torch.gather(cdf, 1, inds_sampled)
    cdf_g = cdf_g.view(cdf_g.shape[0], -1, 2)  # n1 (n2 2) -> n1 n2 2

    bins_g = torch.gather(bins, 1, inds_sampled)
    bins_g = bins_g.view(bins_g.shape[0], -1, 2)  # n1 (n2 2) -> n1 n2 2

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < 1e-8] = 1  # denom equals 0 means a bin has weight 0,
    # in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])
    return samples
