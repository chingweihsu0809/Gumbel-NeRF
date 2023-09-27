import datetime
import faulthandler
import math
import time
import os
import random
import shutil
import signal
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union, cast
from xml.dom.expatbuilder import parseString
import subprocess

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

from src.datasets.filesystem_dataset import FilesystemDataset
from src.image_metadata import ImageMetadata
from src.metrics import psnr, ssim, lpips, psnr_mask, ssim_mask
from src.misc_utils import main_print, main_tqdm, global_main_tqdm, main_log, count_parameters
from src.models.model_utils import get_nerf, CosineScheduler
from src.ray_utils import get_rays, get_ray_directions
from src.rendering import render_rays
from src.modules.tutel_moe_ext import tutel_system
from src.utils.logger import setup_logger
from src.utils.functions import DictAverageMeter, voc_palette, DictAverageMeter1
from src.modules.tutel_moe_ext.tutel_moe_layer_nobatch import MOELayer
from src.modules.tutel_moe_ext.tutel_fast_dispatch_nobatch import one_hot_with_dtype

from contextlib import nullcontext
from plyfile import PlyData, PlyElement

from torch.utils.data import Subset
import json


class Runner:
    def __init__(self, hparams: Namespace, set_experiment_path: bool = True):
        faulthandler.register(signal.SIGUSR1)

        print("WORLD_SIZE", int(os.environ['WORLD_SIZE']))
        print("LOCAL_RANK", int(os.environ['LOCAL_RANK']))
        print("WORLD_RANK", int(os.environ['RANK']))
        print("MASTER_ADDR", os.environ['MASTER_ADDR'])
        print("MASTER_PORT", os.environ['MASTER_PORT'])
        # setup tutel
        parallel_env = tutel_system.init_data_model_parallel(use_slurm=False)
        
        if hparams.no_expert_parallel:
            from tutel import net
            self.single_data_group = net.create_groups_from_world(group_count=1).data_group
        else:
            self.single_data_group = None

        hparams.single_data_group = self.single_data_group
        hparams.parallel_env = parallel_env
        hparams.local_rank = parallel_env.local_device.index
        hparams.dist_rank = parallel_env.global_rank
        self.device = parallel_env.local_device
        hparams.device = self.device
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        torch.set_default_dtype(torch.float32)

        self.is_master = (int(os.environ['RANK']) == 0)
        self.is_local_master = ('RANK' not in os.environ) or int(os.environ['LOCAL_RANK']) == 0

        world_size = parallel_env.global_size
        if hparams.no_expert_parallel:
            hparams.moe_local_expert_num = hparams.expert_num
        else:
            hparams.moe_local_expert_num = hparams.expert_num // world_size
            assert hparams.moe_local_expert_num * world_size == hparams.expert_num

        # dir
        self.hparams = hparams
        self.experiment_path = self._get_experiment_path() if self.is_master else None
        self.model_path = self.experiment_path / 'models' if self.is_master else None

        hparams.experiment_path = self.experiment_path

        self.train_items, self.val_items = self._get_srn_image_metadata()
        self._setup_experiment_dir()

        # logger
        if self.is_master:
            log_dir = self.experiment_path
            hparams.logdir = log_dir
            self.logger = setup_logger(None, log_dir, timestamp=False) # root logger
        else:
            self.logger = None
        hparams.logger = self.logger
        
        try:
            git_commit_id = \
                subprocess.check_output(["git", "rev-parse", "HEAD"]).decode('UTF-8')[0:-1]
            git_branch_name = \
                subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode('UTF-8')[0:-1]
        except:
            main_log("No git founded")
            git_commit_id = ""
            git_branch_name = ""

        self.hparams = hparams
        
        main_log("Branch " + git_branch_name)
        main_log("Commit ID " + git_commit_id)
        main_log(" ".join(sys.argv))
        main_log("Running with config:\n{}".format(hparams))     

        if hparams.ckpt_path is not None:
            checkpoint = torch.load(hparams.ckpt_path, map_location='cpu')
            np.random.set_state(checkpoint['np_random_state'])
            torch.set_rng_state(checkpoint['torch_random_state'])
            random.setstate(checkpoint['random_state'])
        else:
            np.random.seed(hparams.random_seed)
            torch.manual_seed(hparams.random_seed)
            torch.cuda.manual_seed_all(hparams.random_seed)
            random.seed(hparams.random_seed)

        self.near = hparams.near 
        self.far = hparams.far 
        camera_positions = torch.cat([x.c2w[:3, 3].unsqueeze(0) for x in self.train_items + self.val_items])
        min_position = camera_positions.min(dim=0)[0]
        max_position = camera_positions.max(dim=0)[0]

        main_log('Ray bounds: {}, {}'.format(self.near, self.far))
        main_log('Using {} train images and {} val images'.format(len(self.train_items), len(self.val_items)))
        main_log('Camera range in [-1, 1] space: {} {}'.format(min_position, max_position))

        self.nerf = get_nerf(hparams, len(self.train_items)).to(self.device)
            
        if self.is_master:
            print(self.nerf)
            num_params = {p: v.numel() for p, v in self.nerf.named_parameters()}
            for nump in num_params.items():
                print(nump)
            print("total:",  sum(list(num_params.values())))
        self.model_parameter_num = count_parameters(self.nerf)
        main_log(f"Total parameters number is {self.model_parameter_num / 1024.0 / 1024.0:.4f} M")

        if 'RANK' in os.environ:
            self.nerf = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.nerf)
            self.nerf = torch.nn.parallel.DistributedDataParallel(self.nerf, device_ids=[int(os.environ['LOCAL_RANK'])],
                                                                output_device=int(os.environ['LOCAL_RANK']),
                                                                find_unused_parameters=hparams.find_unused_parameters)

    def train(self):
        # self._setup_experiment_dir()
        
        if not self.hparams.moe_train_batch:
            self.set_no_batch(mode=True)
        else:
            self.set_no_batch(mode=False)
        scaler = torch.cuda.amp.GradScaler(enabled=self.hparams.amp)

        optimizers = {}
        if self.hparams.latent_lr_gain != 1:
            code_param_filter = ['module.latent_net.shape_codes.weight', 'module.latent_net.texture_codes.weight']
            code_params = [param for name, param in self.nerf.named_parameters() if name in code_param_filter]
            model_params = [param for name, param in self.nerf.named_parameters() if name not in code_param_filter]
            print(f"split {len(list(self.nerf.parameters()))} parameters into {len(code_params)} for code and {len(model_params)} for model")
            if len(model_params) != 0:
                optimizers['nerf'] = Adam(model_params, lr=self.hparams.lr)
            if len(code_params) != 0:
                optimizers['code'] = Adam(code_params, lr=self.hparams.lr * self.hparams.latent_lr_gain)
        else:
            optimizers['nerf'] = Adam(self.nerf.parameters(), lr=self.hparams.lr)
            
        if self.hparams.ckpt_path is not None:
            checkpoint = torch.load(self.hparams.ckpt_path, map_location='cpu')
            train_iterations = checkpoint['iteration']
            scaler_dict = scaler.state_dict()
            scaler_dict.update(checkpoint['scaler'])
            scaler.load_state_dict(scaler_dict)
            for key, optimizer in optimizers.items():
                optimizer_dict = optimizer.state_dict()
                optimizer_dict.update(checkpoint['optimizers'][key])
                optimizer.load_state_dict(optimizer_dict)
            discard_index = checkpoint['dataset_index'] if self.hparams.resume_ckpt_state else -1
        else:
            train_iterations = 0
            discard_index = -1

        schedulers = {}
        if self.hparams.no_optimizer_schedulers:
            pass
        else:
            for key, optimizer in optimizers.items():
                schedulers[key] = ExponentialLR(optimizer,
                                                gamma=self.hparams.lr_decay_factor ** (1 / (self.hparams.train_iterations if self.hparams.task == "train" else train_iterations)),
                                                last_epoch=train_iterations - 1)

        # Let the local master write data to disk first
        # We could further parallelize the disk writing process by having all of the ranks write data,
        # but it would make determinism trickier
        if 'RANK' in os.environ and (not self.is_local_master):
            dist.barrier()
        
        dataset = FilesystemDataset(self.train_items, self.near, self.far,
                                    self.hparams.center_pixels, self.device,
                                    [Path(x) for x in sorted(self.hparams.chunk_paths)], self.hparams.num_chunks, 
                                    self.hparams.disk_flush_size, self.hparams.shuffle_chunk)

        
        if self.hparams.ckpt_path is not None and self.hparams.resume_ckpt_state:
            dataset.set_state(checkpoint['dataset_state'])
        if 'RANK' in os.environ and self.is_local_master:
            dist.barrier()
            
                
        if self.hparams.generate_chunk:
            main_log(f"Chunk generated")
            return

        if self.is_master:
            pbar = tqdm(total=self.hparams.train_iterations)
            pbar.update(train_iterations)
        else:
            pbar = None
        
        if self.hparams.use_gumbel:
            if self.hparams.T_init == None:
                self.hparams.T_init = train_iterations
            self.temperature_scheduler = CosineScheduler(self.hparams.eta_max, self.hparams.eta_min, self.hparams.T_max, self.hparams.T_init)
            self.temperature_scheduler.t_cur = train_iterations
            
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)

        train_meter = DictAverageMeter1()
        while train_iterations < self.hparams.train_iterations:
            # If discard_index >= 0, we already set to the right chunk through set_state
            main_log(f"Loading chunk {dataset.get_state()}")
            chunk_time = time.time()
            dataset.load_chunk()
            chunk_time = time.time() - chunk_time
            main_log(f"Chunk {dataset.get_state()} loaded by {chunk_time:.2f} s")

            if 'RANK' in os.environ:
                world_size = int(os.environ['WORLD_SIZE'])
                sampler = DistributedSampler(dataset, world_size, int(os.environ['RANK']))
                assert self.hparams.batch_size % world_size == 0
                data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size // world_size, sampler=sampler,
                                         num_workers=self.hparams.data_loader_num_workers, pin_memory=True)
            else:
                data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.data_loader_num_workers,
                                         pin_memory=True)

            data_sample_time = time.time()
            for dataset_index, item in enumerate(data_loader):
                data_sample_time = time.time() - data_sample_time
                if dataset_index <= discard_index:
                    data_sample_time = time.time()
                    continue

                discard_index = -1
                fwd_bwd_time = time.time()
                fwd_time = time.time()
                amp_dtype = torch.bfloat16 if self.hparams.amp_use_bfloat16 else torch.float32
                self.hparams.amp_dtype = amp_dtype
                
                if self.hparams.compute_memory:
                    torch.cuda.reset_peak_memory_stats()
                with torch.cuda.amp.autocast(enabled=self.hparams.amp, dtype=amp_dtype):
                    if self.hparams.appearance_dim > 0:
                        image_indices = item['image_indices'].to(self.device, non_blocking=True)
                    else:
                        image_indices = None
                    if self.hparams.latent_dim > 0:
                        instance_ids = item['id'].to(self.device, non_blocking=True)
                    else:
                        instance_ids = None

                    metrics = self._training_step(
                        item['rgbs'].to(self.device, non_blocking=True),
                        item['rays'].to(self.device, non_blocking=True),
                        image_indices,
                        instance_ids)

                    if self.hparams.disable_check_finite:
                        pass
                    else:
                        check_finite = torch.tensor(1, device=self.device)
                        with torch.no_grad():
                            for key, val in metrics.items():
                                if key == 'psnr' and math.isinf(val):  # a perfect reproduction will give PSNR = infinity
                                    continue
                                if isinstance(val, torch.Tensor) and len(val.shape) > 0:
                                    continue

                                if not math.isfinite(val):
                                    check_finite -= 1
                                    main_log(f"{key} is infinite, val{val}")
                                    break
                        
                        check_finite_gather = [torch.tensor(0, device=self.device) for _ in range(dist.get_world_size())]
                        dist.all_gather(check_finite_gather, check_finite)
                        check_finite_gather = torch.stack(check_finite_gather)

                all_loss = metrics['loss']
                if self.hparams.use_moe and self.hparams.use_balance_loss:

                    moe_l_aux_wt = self.hparams.moe_l_aux_wt
                    if "gate_loss" in metrics:
                        all_loss = all_loss + moe_l_aux_wt * metrics['gate_loss']

                metrics['all_loss'] = all_loss

                if self.hparams.disable_check_finite:
                    pass
                else:
                    check_finite_flag = (check_finite_gather.sum().item() == dist.get_world_size())
                    if not check_finite_flag:
                        with self.nerf.no_sync():
                            with nullcontext():
                                scaler.scale(all_loss).backward()
                        for optimizer in optimizers.values():
                            optimizer.zero_grad(set_to_none=True)
                        
                        main_log(f"skip step {train_iterations} due to inf")
                        main_log(f"check_finite of GPUs {check_finite_gather}")
                        continue

                fwd_time = time.time() - fwd_time

                with nullcontext():
                    scaler.scale(all_loss).backward()

                for key, optimizer in optimizers.items():
                    scaler.step(optimizer)

                scaler.update()
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)

                for scheduler in schedulers.values():
                    scheduler.step()
                    
                if self.hparams.use_gumbel:
                    self.temperature_scheduler.step()

                fwd_bwd_time = time.time() - fwd_bwd_time

                if self.hparams.compute_memory:
                    forward_backward_memory = torch.cuda.max_memory_allocated() / (1024.0 ** 2)

                train_iterations += 1
                train_meter_dict = {}
                for k, v in metrics.items():
                    train_meter_dict[k] = v.item() if isinstance(v, torch.Tensor) and len(v.shape) == 0 else v

                train_meter.update(train_meter_dict)
                if self.is_master:
                    train_meter_mean = train_meter.mean()
                    pbar.set_postfix(psnr=f"{train_meter_dict['psnr']:.2f} ({train_meter_mean['psnr']:.2f})")
                    pbar.update(1)
                    
                    for key, value in optimizers.items():
                        self.writer.add_scalar(f'train/lr_{key}', value.param_groups[0]['lr'], train_iterations)
                    
                    for key, value in metrics.items():
                        if not isinstance(value, torch.Tensor):
                            self.writer.add_scalar('train/{}'.format(key), value, train_iterations)
                        else:
                            self.writer.add_scalar('train/{}'.format(key), value.item(), train_iterations)
                            
                    if self.hparams.use_gumbel:
                        self.writer.add_scalar('train/temperature', self.temperature_scheduler.eta_t, train_iterations)

                    if train_iterations > 0 and train_iterations % self.hparams.ckpt_interval == 0:
                        tmp_optimizers = optimizers
                        self._save_checkpoint(tmp_optimizers, scaler, train_iterations, dataset_index, dataset.get_state())
                        
                    if train_iterations % self.hparams.i_print==0 or train_iterations == self.hparams.train_iterations:
                        train_meter_mean = train_meter.mean()
                        train_print_str = [
                            f"[TRAIN] Iter: {train_iterations} all_loss: {train_meter_dict['all_loss']:.5f} ({train_meter_mean['all_loss']:.5f})",
                            f"PSNR: {train_meter_dict['psnr']:.2f} ({train_meter_mean['psnr']:.2f})",
                            f"img_loss: {train_meter_dict['loss']:.5f} ({train_meter_mean['loss']:.5f})",
                            f"lr: {optimizers['nerf'].param_groups[0]['lr']:.5f}",
                            f"data time: {data_sample_time:.5f}",
                            f"fwd_bwd time: {fwd_bwd_time:.5f}",
                            f"fwd_time: {fwd_time:.5f}",
                            f"bwd_time: {fwd_bwd_time - fwd_time:.5f}",
                            f"fwd_bwd memory: {forward_backward_memory:.2f}" if self.hparams.compute_memory else ""
                        ]

                        if self.hparams.use_balance_loss:
                            if "gate_loss" in metrics:
                                train_print_str.append(f"gate_loss: {self.hparams.moe_l_aux_wt * train_meter_dict['gate_loss']:.7f} ({self.hparams.moe_l_aux_wt * train_meter_mean['gate_loss']:.7f})")
                                train_print_str.append(f"real_gate_loss: {train_meter_dict['gate_loss']:.7f} ({train_meter_mean['gate_loss']:.7f})")
                        
                        train_print_str = " ".join(train_print_str)
                        main_log(train_print_str)


                data_sample_time = time.time()

                if train_iterations >= self.hparams.train_iterations:
                    break

        if 'RANK' in os.environ:
            dist.barrier()

        if self.is_master:
            pbar.close()
            tmp_optimizers = optimizers
            self._save_checkpoint(tmp_optimizers, scaler, train_iterations, dataset_index, dataset.get_state())

    def set_no_batch(self, mode=True):
        for net in self.nerf.modules():
            if hasattr(net, "moe_no_batch"):
                net.moe_no_batch = mode
  
    def eval_image(self):
        # self._setup_experiment_dir()
        if not self.hparams.moe_test_batch:
            self.set_no_batch(mode=True)
        else:
            self.set_no_batch(mode=False)
        val_metrics = self._run_validation_image(0)
        self._write_final_metrics(val_metrics)  
  
    def test_time_opt(self):
        # self._setup_experiment_dir()
        if not self.hparams.moe_test_batch:
            self.set_no_batch(mode=True)
        else:
            self.set_no_batch(mode=False)
        self._test_time_opt(0)

    def _write_final_metrics(self, val_metrics: Dict[str, float]) -> None:
        if self.is_master:
            with (self.experiment_path / 'metrics.txt').open('w') as f:
                for key in val_metrics:
                    avg_val = val_metrics[key] / len(self.val_items)
                    message = 'Average {}: {}'.format(key, avg_val)
                    main_log(message)
                    f.write('{}\n'.format(message))

            self.writer.flush()
            self.writer.close()

    def _setup_experiment_dir(self) -> None:
        if self.is_master:
            self.experiment_path.mkdir(exist_ok=True)
            with (self.experiment_path / 'hparams.txt').open('w') as f:
                for key in vars(self.hparams):
                    f.write('{}: {}\n'.format(key, vars(self.hparams)[key]))
                if 'WORLD_SIZE' in os.environ:
                    f.write('WORLD_SIZE: {}\n'.format(os.environ['WORLD_SIZE']))

            with (self.experiment_path / 'command.txt').open('w') as f:
                f.write(' '.join(sys.argv))
                f.write('\n')

            self.model_path.mkdir(parents=True, exist_ok=True)

            with (self.experiment_path / 'image_indices.txt').open('w') as f:
                for i, metadata_item in enumerate(self.train_items):
                    f.write('{},{}\n'.format(metadata_item.image_index, metadata_item.image_path.name))
        self.writer = SummaryWriter(str(self.experiment_path / 'tb')) if self.is_master else None

        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and os.environ['WORLD_SIZE'] != "1":
            dist.barrier()
    
    def _get_experiment_path(self) -> Path:
        exp_dir = Path(self.hparams.exp_name)
        exp_dir.mkdir(parents=True, exist_ok=True)
        existing_versions = [int(x.name) for x in exp_dir.iterdir() if x.name.isdigit()]
        if self.hparams.ckpt_path is not None and self.hparams.exp_name in self.hparams.ckpt_path:
            version = os.path.basename(os.path.dirname(os.path.dirname(self.hparams.ckpt_path)))
        else:
            version = 0 if len(existing_versions) == 0 else max(existing_versions) + 1
        experiment_path = exp_dir / str(version)
        return experiment_path   

    def _training_step(self, rgbs: torch.Tensor, rays: torch.Tensor, image_indices: Optional[torch.Tensor], instance_ids: Optional[torch.Tensor]) \
            -> Tuple[Dict[str, Union[torch.Tensor, float]], bool]:
        
        results = render_rays(  nerf=self.nerf,
                                rays=rays,
                                image_indices=image_indices,
                                instance_ids=instance_ids,
                                hparams=self.hparams,
                                get_depth=False,
                                get_depth_variance=True,
                                get_bg_fg_rgb=False,
                                nerf_kwargs={'temperature': self.temperature_scheduler.eta_t} if self.hparams.use_gumbel else None)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            depth_variance = results[f'depth_variance_{typ}'].mean()

        metrics = {
            'psnr': psnr_,
            'depth_variance': depth_variance,
        }

        photo_loss = F.mse_loss(results[f'rgb_{typ}'], rgbs, reduction='mean')
        metrics['photo_loss'] = photo_loss
        metrics['loss'] = photo_loss
        
        if self.hparams.reg_wt != 0:
            reg_loss = torch.norm(self.nerf.module.latent_net.shape_codes.weight, dim=-1) + \
                       torch.norm(self.nerf.module.latent_net.texture_codes.weight, dim=-1)
            metrics['reg_loss'] =  torch.mean(reg_loss)
            metrics['loss'] +=  metrics['reg_loss'] * self.hparams.reg_wt            

        if self.hparams.use_moe and self.hparams.use_balance_loss:
            gate_loss = torch.mean(results[f'gate_loss_{typ}'])
            metrics[f'{typ}_gate_loss'] = gate_loss
            metrics['gate_loss'] = gate_loss
            if typ == 'fine':
                coarse_gate_loss = torch.mean(results[f'gate_loss_coarse'])
                metrics['coarse_gate_loss'] = coarse_gate_loss
                metrics['gate_loss'] = (metrics['gate_loss'] + coarse_gate_loss) / 2.0
            
        return metrics

    def _run_validation_image(self, train_index: int) -> Dict[str, float]:
        with torch.inference_mode():
            self.nerf.eval()

            val_metrics = defaultdict(float)
            val_metrics_per_instance = {'psnr': defaultdict(list), 'ssim': defaultdict(list)}
            base_tmp_path = None
            try:
                if 'RANK' in os.environ:
                    base_tmp_path = Path(self.hparams.exp_name) / os.environ['TORCHELASTIC_RUN_ID']
                    metric_path = base_tmp_path / 'tmp_val_metrics'
                    image_path = base_tmp_path / 'tmp_val_images'

                    world_size = int(os.environ['WORLD_SIZE'])
                    indices_to_eval = np.arange(int(os.environ['RANK']), len(self.val_items), world_size)
                    if self.is_master:
                        base_tmp_path.mkdir()
                        metric_path.mkdir()
                        image_path.mkdir()
                        base_img_path = Path(self.experiment_path) / "images"
                        base_img_path.mkdir()
                        base_img_path_broadcast = [base_img_path]
                    if not self.is_master:
                        base_img_path_broadcast = [None]
                    torch.distributed.broadcast_object_list(base_img_path_broadcast, src=0)
                    dist.barrier()
                    base_img_path = base_img_path_broadcast[0]
                else:
                    indices_to_eval = np.arange(len(self.val_items))
                    base_img_path = Path(self.experiment_path) / "images"
                    base_img_path.mkdir()

                for i in global_main_tqdm(indices_to_eval):
                    metadata_item = self.val_items[i]
                    image_path = metadata_item.image_path
                    image_idx = int(os.path.splitext(os.path.basename(image_path))[0])
                    instance_name = os.path.basename(os.path.dirname(os.path.dirname(image_path)))

                    torch.cuda.reset_peak_memory_stats()
                    time_end = time.time()
                    viz_rgbs = metadata_item.load_image().float() / 255.
                    results, _ = self.render_image(metadata_item)                    
                    
                    forward_time = time.time() - time_end
                    forward_max_memory_allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 2)

                    val_metrics['val/time'] += forward_time                    
                    val_metrics['val/memory'] += forward_max_memory_allocated                    

                    typ = 'fine' if 'rgb_fine' in results else 'coarse'
                    viz_result_rgbs = results[f'rgb_{typ}'].view(*viz_rgbs.shape).cpu()

                    eval_rgbs = viz_rgbs.contiguous()
                    eval_result_rgbs = viz_result_rgbs.contiguous()

                    val_psnr = psnr(eval_result_rgbs.view(-1, 3), eval_rgbs.view(-1, 3))
                    val_metrics['val/psnr'] += val_psnr
                    val_metrics_per_instance["psnr"][instance_name] += [val_psnr]
                    val_ssim = ssim(eval_result_rgbs.view(*eval_rgbs.shape), eval_rgbs, 1)
                    val_metrics['val/ssim'] += val_ssim
                    val_metrics_per_instance["ssim"][instance_name] += [val_ssim]
                    val_lpips_metrics = lpips(eval_result_rgbs.view(*eval_rgbs.shape), eval_rgbs)
                    for network in val_lpips_metrics:
                        agg_key = 'val/lpips/{}'.format(network)
                        val_metrics[agg_key] += val_lpips_metrics[network]
                    
                    val_metrics_txt = {"psnr": val_psnr, "ssim": val_ssim}
                    for tmp_network in val_lpips_metrics:
                        val_metrics_txt['lpips-{}'.format(tmp_network)] = val_lpips_metrics[tmp_network]
                    val_metrics_txt["time"] = forward_time
                    val_metrics_txt["memory"] = forward_max_memory_allocated

                    os.makedirs(f'{base_img_path}/{instance_name}', exist_ok=True)
                    if int(image_idx) in self.hparams.save_img_index:
                        with (base_img_path / f'{instance_name}/{image_idx}_metrics.txt').open('w') as f:
                            for key in val_metrics_txt:
                                message = '{}: {}'.format(key, val_metrics_txt[key])
                                f.write('{}\n'.format(message))

                    viz_result_rgbs = viz_result_rgbs.view(viz_rgbs.shape[0], viz_rgbs.shape[1], 3).cpu()

                    img = Runner._create_result_image(viz_rgbs, viz_result_rgbs)

                    if int(image_idx) in self.hparams.save_img_index:                            
                        for img_i, img_suf in enumerate(["gt", "pred"]):
                            # left, upper, right, lower
                            img_w, img_h = img.size
                            img_box = [img_w // 2 * img_i, 0, img_w // 2 * (img_i + 1), img_h]
                            img.crop(img_box).save(f'{base_img_path}/{instance_name}/{image_idx}_{img_suf}.jpg')
                            
                        for k in range(self.hparams.expert_num):
                            viz_result_rgbs_exp = results[f'rgb_{typ}_exp{k}'].view(viz_rgbs.shape[0], viz_rgbs.shape[1], 3).cpu()
                            img_exp = Image.fromarray(np.array(viz_result_rgbs_exp * 255).astype(np.uint8))
                            img_exp.save(f'{base_img_path}/{instance_name}/{image_idx}_pred_exp{k}.jpg')


                    del results

                if 'RANK' in os.environ:
                    dist.barrier()
                    metric_gather = [None for _ in range(dist.get_world_size())]
                    dist.all_gather_object(metric_gather, val_metrics)
                    metric_gather_per_instance = [None for _ in range(dist.get_world_size())]
                    dist.all_gather_object(metric_gather_per_instance, val_metrics_per_instance)
        
                    if self.writer is not None:
                        for mk in val_metrics:
                            gathered_val = sum([met[mk] for met in metric_gather])     
                            val_metrics[mk] = gathered_val
                        for key in val_metrics:
                            avg_val = val_metrics[key] / len(self.val_items)
                            self.writer.add_scalar('{}/avg'.format(key), avg_val, 0)

                        for mk, mv in val_metrics_per_instance.items():
                            for instk in mv:
                                val_metrics_per_instance[mk][instk] = sum([met[mk][instk] for met in metric_gather_per_instance], [])
                            with (self.experiment_path / f'{mk}_per_instance.txt').open('w') as f:
                                for instk, instval in val_metrics_per_instance[mk].items():
                                    avg_val = sum(instval) / len(instval)
                                    f.write(f'{instk}: {avg_val}\n')

                    dist.barrier()

                self.nerf.train()
            finally:
                if self.is_master and base_tmp_path is not None:
                    shutil.rmtree(base_tmp_path)

            return val_metrics
    
    def _test_time_opt(self, train_index: int) -> Dict[str, float]:
        
        base_tmp_path = None
        try:
            ## prepare dir
            if 'RANK' in os.environ:
                base_tmp_path = Path(self.hparams.exp_name) / os.environ['TORCHELASTIC_RUN_ID']

                world_size = int(os.environ['WORLD_SIZE'])
                indices_to_eval = np.arange(int(os.environ['RANK']), len(self.val_items), world_size)
                if self.is_master:
                    base_tmp_path.mkdir()
                    base_img_path = Path(self.experiment_path) / "images"
                    base_img_path.mkdir()
                    base_img_path_broadcast = [base_img_path]
                if not self.is_master:
                    base_img_path_broadcast = [None]
                torch.distributed.broadcast_object_list(base_img_path_broadcast, src=0)
                dist.barrier()
                base_img_path = base_img_path_broadcast[0]
            else:
                indices_to_eval = np.arange(len(self.val_items))
                base_img_path = Path(self.experiment_path) / "images"
                base_img_path.mkdir()

            ## prepare scheduler and optimizer
            self.set_no_batch(mode=not self.hparams.moe_train_batch)
            scaler = torch.cuda.amp.GradScaler(enabled=self.hparams.amp)
            
            opt_params = []
            for name, param in self.nerf.named_parameters():
                if "shape_codes" in name or "texture_codes" in name: # or "shapefc" in name:
                    opt_params.append(param)
                else:
                    param.requires_grad = False
            
            ## optimize
            for i_obj, ind in enumerate(indices_to_eval):

                if self.is_master:
                    print(f"optimizing {i_obj} / {len(indices_to_eval)} th object")
                    print(f"opt_params_len: {len(opt_params)}")
                    
                optimizers = {'nerf': Adam(opt_params, lr=self.hparams.lr)}
                schedulers = {}
                if self.hparams.no_optimizer_schedulers:
                    pass
                else:
                    for key, optimizer in optimizers.items():
                        schedulers[key] = ExponentialLR(optimizer,
                                                        gamma=self.hparams.lr_decay_factor ** (1 / self.hparams.train_iterations))
                        
                self.nerf.train()
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
                    
                metadata_item = self.val_items[ind]
                viz_rgbs = metadata_item.load_image().float() / 255.
                viz_rgbs = viz_rgbs.view(-1, 3).to(self.device)
                
                for train_iterations in global_main_tqdm(range(self.hparams.train_iterations)):
                    
                    directions = get_ray_directions(metadata_item.W,
                                                    metadata_item.H,
                                                    metadata_item.intrinsics[0],
                                                    metadata_item.intrinsics[1],
                                                    metadata_item.intrinsics[2],
                                                    metadata_item.intrinsics[3],
                                                    self.hparams.center_pixels,
                                                    self.device)


                    amp_dtype = torch.bfloat16 if self.hparams.amp_use_bfloat16 else torch.float32
                    with torch.cuda.amp.autocast(enabled=self.hparams.amp, dtype=amp_dtype):
                        rays = get_rays(directions, metadata_item.c2w.to(self.device), self.near, self.far)

                        rays = rays.view(-1, 8).to(self.device, non_blocking=True)  # (H*W, 8)
                        instance_ids = metadata_item.instanceid * torch.ones(rays.shape[0], device=rays.device) \
                            if self.hparams.latent_dim > 0 else None

                        for i in range(0, rays.shape[0], self.hparams.image_pixel_batch_size):
                            result_batch = render_rays(nerf=self.nerf,
                                                        rays=rays[i:i + self.hparams.image_pixel_batch_size],
                                                        image_indices=None,
                                                        instance_ids=instance_ids[
                                                                        i:i + self.hparams.image_pixel_batch_size] if self.hparams.latent_dim > 0 else None,
                                                        hparams=self.hparams,
                                                        get_depth=False,
                                                        get_depth_variance=False,
                                                        get_bg_fg_rgb=True,
                                                        nerf_kwargs={'temperature': self.hparams.eta_min} if self.hparams.use_gumbel else None)
                            viz_rgbs_batch = viz_rgbs[i:i + self.hparams.image_pixel_batch_size]
                            
                            ## calculate metrics
                            typ = 'fine' if 'rgb_fine' in result_batch else 'coarse'
                            with torch.no_grad():
                                psnr_ = psnr(result_batch[f'rgb_{typ}'], viz_rgbs_batch)
                            metrics = {
                                'psnr': psnr_,
                            }
                            photo_loss = F.mse_loss(result_batch[f'rgb_{typ}'], viz_rgbs_batch, reduction='mean')
                            metrics['photo_loss'] = photo_loss
                            metrics['loss'] = photo_loss      
                            
                            
                            if self.hparams.reg_wt != 0:
                                reg_loss = torch.norm(self.nerf.module.latent_net.shape_codes.weight, dim=-1) + \
                                           torch.norm(self.nerf.module.latent_net.texture_codes.weight, dim=-1)
                                metrics['reg_loss'] =  torch.mean(reg_loss)
                                metrics['loss'] +=  metrics['reg_loss'] * self.hparams.reg_wt
                                
                            if self.writer is not None and i == (rays.shape[0] // 2):
                                image_path = metadata_item.image_path
                                image_idx = int(os.path.splitext(os.path.basename(image_path))[0])
                                instance_name = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
                                for key, value in optimizers.items():
                                    self.writer.add_scalar(f'tto_{instance_name}_{image_idx}/lr_{key}', value.param_groups[0]['lr'], train_iterations)
                                
                                for key, value in metrics.items():
                                    if not isinstance(value, torch.Tensor):
                                        self.writer.add_scalar(f'tto_{instance_name}_{image_idx}/{key}', value, train_iterations)
                                    else:
                                        self.writer.add_scalar(f'tto_{instance_name}_{image_idx}/{key}', value.item(), train_iterations)
                                        
                                        
                            ## backward
                            with nullcontext():
                                scaler.scale(metrics['loss']).backward()

                            for key, optimizer in optimizers.items():
                                scaler.step(optimizer)

                            scaler.update()
                            for optimizer in optimizers.values():
                                optimizer.zero_grad(set_to_none=True)

                        for scheduler in schedulers.values():
                            scheduler.step()
                        ## backward end
                            
                                
                                
            if 'RANK' in os.environ:
                dist.barrier()
                                        
            if self.is_master:
                main_log(f"save checkpoint after step {train_iterations}")
                tmp_optimizers = optimizers
                self._save_checkpoint(tmp_optimizers, scaler, train_iterations, 0, None)

        finally:
            if self.is_master and base_tmp_path is not None:
                shutil.rmtree(base_tmp_path)
        
    def _save_checkpoint(self, optimizers: Dict[str, any], scaler: GradScaler, train_index: int, dataset_index: int,
                         dataset_state: Optional[str]) -> None:
        dict = {
            'model_state_dict': self.nerf.state_dict(),
            'scaler': scaler.state_dict(),
            'optimizers': {k: v.state_dict() for k, v in optimizers.items()},
            'iteration': train_index,
            'torch_random_state': torch.get_rng_state(),
            'np_random_state': np.random.get_state(),
            'random_state': random.getstate(),
            'dataset_index': dataset_index
        }
        
        if dataset_state is not None:
            dict['dataset_state'] = dataset_state

        torch.save(dict, self.model_path / '{}.pt'.format(train_index))

    def render_image(self, metadata: ImageMetadata) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        directions = get_ray_directions(metadata.W,
                                        metadata.H,
                                        metadata.intrinsics[0],
                                        metadata.intrinsics[1],
                                        metadata.intrinsics[2],
                                        metadata.intrinsics[3],
                                        self.hparams.center_pixels,
                                        self.device)

        amp_dtype = torch.bfloat16 if self.hparams.amp_use_bfloat16 else torch.float32
        with torch.cuda.amp.autocast(enabled=self.hparams.amp, dtype=amp_dtype):
            rays = get_rays(directions, metadata.c2w.to(self.device), self.near, self.far)

            rays = rays.view(-1, 8).to(self.device, non_blocking=True)  # (H*W, 8)
            image_indices = metadata.image_index * torch.ones(rays.shape[0], device=rays.device) \
                if self.hparams.appearance_dim > 0 else None
            instance_ids = metadata.instanceid * torch.ones(rays.shape[0], device=rays.device) \
                if self.hparams.latent_dim > 0 else None
            results = {}

            if 'RANK' in os.environ:
                nerf = self.nerf.module
            else:
                nerf = self.nerf

            for i in range(0, rays.shape[0], self.hparams.image_pixel_batch_size):
                result_batch = render_rays(nerf=nerf, 
                                            rays=rays[i:i + self.hparams.image_pixel_batch_size],
                                            image_indices=image_indices[
                                                        i:i + self.hparams.image_pixel_batch_size] if self.hparams.appearance_dim > 0 else None,
                                            instance_ids=instance_ids[
                                                        i:i + self.hparams.image_pixel_batch_size] if self.hparams.latent_dim > 0 else None,
                                            hparams=self.hparams,
                                            get_depth=True,
                                            get_depth_variance=False,
                                            get_bg_fg_rgb=True,
                                            nerf_kwargs={'temperature': self.hparams.eta_min} if self.hparams.use_gumbel else None)

                for key, value in result_batch.items():
                    if key not in results:
                        results[key] = []

                    results[key].append(value.cpu())

            for key, value in results.items():
                results[key] = torch.cat(value)

            return results, rays
   
    @staticmethod
    def _create_result_image(rgbs: torch.Tensor, result_rgbs: torch.Tensor) -> Image:
        images = (rgbs * 255, result_rgbs * 255)
        return Image.fromarray(np.concatenate(images, 1).astype(np.uint8))
 
    def _get_srn_image_metadata(self) -> Tuple[List[ImageMetadata], List[ImageMetadata]]:
        dataset_path = Path(self.hparams.dataset_path)
        postfix = self.hparams.item_files_postfix

        if self.hparams.latent_dim != 0:
            
            ## shortcut
            if (dataset_path/f'train_items{postfix}.pt').exists():
                train_items = torch.load((dataset_path/f'train_items{postfix}.pt'))
                val_items = torch.load((dataset_path/f'test_items{postfix}.pt'))
                if self.hparams.task == "train":
                    pass
                elif self.hparams.task == "tto":
                    pass
                elif self.hparams.task == "test":
                    pass
                elif self.hparams.task == "test_on_train":
                    train_items, val_items = val_items, train_items
                elif self.hparams.task == "fastrun":
                    selected_instance = ["acf64e44b021fd4613b86a9df4269733",
                                         "d2efbf5a3b7ddbf94c0aa7c1668459cf",
                                         "510df40932e79779a324deea8acbeebe",
                                         "fd7741b7927726bda37f3fc191551700",
                                         "1d234607fafd576b6dede2eec861f76",
                                         "9f4bbcf9f51fe1e42957c02bdefc95c8"]
                    val_items = [v for v in val_items if any(s in str(v.image_path) for s in selected_instance)] 
                    train_items = train_items[0:1]
                else:
                    raise NotImplementedError
                
                val_items = [v for v in val_items if int(os.path.splitext(os.path.basename(v.image_path))[0]) not in self.hparams.exclude_img_index] 
                val_items = [v for v in val_items if len(self.hparams.img_index) == 0 or int(os.path.splitext(os.path.basename(v.image_path))[0]) in self.hparams.img_index] 
                return train_items, val_items
                
                
            if (dataset_path/'train_list_multi.txt').exists():
                with open((dataset_path/'train_list_multi.txt'), 'r') as file:
                    train_paths = file.readlines()
                    train_paths = [Path(i.rstrip('\n')) for i in train_paths]
            else:
                train_path_candidates = sorted(sum([list((carid_path / 'pose').iterdir()) for carid_path in (dataset_path / 'cars_train').iterdir() if not carid_path.is_file()], []))
                train_path_candidates = [i for i in train_path_candidates if 'intrinsics.txt' not in i.name]
                train_paths = train_path_candidates[::self.hparams.train_every]
                
                with open((dataset_path/'train_list_multi.txt'), 'w') as file:
                    for item in train_paths:
                        file.write(str(item) + '\n')
                
            if (dataset_path/'test_list_multi.txt').exists():
                with open((dataset_path/'test_list_multi.txt'), 'r') as file:
                    val_paths = file.readlines()
                    val_paths = [Path(i.rstrip('\n')) for i in val_paths]
            else:
                val_path_candidates = sorted(sum([list((carid_path / 'pose').iterdir()) for carid_path in (dataset_path / 'cars_test').iterdir() if not carid_path.is_file()], []))
                val_paths = [i for i in val_path_candidates if 'intrinsics.txt' not in i.name]
                
                with open((dataset_path/'test_list_multi.txt'), 'w') as file:
                    for item in val_paths:
                        file.write(str(item) + '\n')
            
        else:
            train_paths = sorted(list((dataset_path / 'cars_train' / '1a1dcd236a1e6133860800e6696b8284' / 'pose').iterdir()))
            train_paths = [i for i in train_paths if 'intrinsics.txt' not in i.name]
            val_paths = train_paths[-10:]
            train_paths = train_paths[:-10]
            
            train_paths += val_paths
            train_paths.sort(key=lambda x: x.name)
            
        val_paths_set = set(val_paths)
        image_indices = {train_path: i for i, train_path in enumerate(train_paths)}
        carid_map = {idname.stem: i for i, idname in enumerate((dataset_path/'cars_train').iterdir())}
        train_items = [
            self._get_srn_metadata_item(x, image_indices[x], x in val_paths_set, carid_map[x.parent.parent.stem] if self.hparams.latent_dim != 0 else -1) for x
            in tqdm(train_paths)]
        image_indices_test = {val_path: i for i, val_path in enumerate(val_paths)}
        carid_map_test = {idname.stem: i for i, idname in enumerate((dataset_path/'cars_test').iterdir())}
        val_items = [
            self._get_srn_metadata_item(x, image_indices_test[x], False, carid_map_test[x.parent.parent.stem] if self.hparams.latent_dim != 0 else -1) for x
            in tqdm(val_paths)]
        
        # save for shortcut
        if self.hparams.latent_dim != 0:
            torch.save(train_items, (dataset_path/f'train_items{postfix}.pt'))
            torch.save(val_items, (dataset_path/f'test_items{postfix}.pt'))
        return train_items, val_items
    
    def _get_srn_metadata_item(self, metadata_path: Path, image_index: int, is_val: bool, instanceid: int) -> ImageMetadata:
        image_path = None
        for extension in ['.jpg', '.JPG', '.png', '.PNG']:
            candidate = metadata_path.parent.parent / 'rgb' / '{}{}'.format(metadata_path.stem, extension)
            if candidate.exists():
                image_path = candidate
                break

        assert image_path.exists()
            
        intrinsics_path = metadata_path.parent.parent / 'intrinsics.txt'
        with open(intrinsics_path, 'r') as f:
            lines = f.readlines()
            focal = float(lines[0].split()[0])
            H, W = lines[-1].split()
            H, W = int(H), int(W)
        
        intrinsics = torch.Tensor([focal, focal, H // 2, W // 2])
        
        srn_coords_trans = torch.diag(torch.Tensor([1, -1, -1, 1])) # SRN dataset
        c2w = torch.Tensor(np.loadtxt(metadata_path).reshape(4,4))
        c2w = c2w @ srn_coords_trans
        c2w = c2w[:3, :]

        mask_path = None
        main_log('No mask path')
        return ImageMetadata(image_path, c2w, W, H,
                             intrinsics, image_index, None, is_val,
                             crop_img=self.hparams.crop_img, instanceid = instanceid)

