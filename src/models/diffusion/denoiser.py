from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data import Batch
from .inner_model import InnerModel, InnerModelConfig
from utils import LossAndLogs


def add_dims(input: Tensor, n: int) -> Tensor:
    return input.reshape(input.shape + (1,) * (n - input.ndim))


@dataclass
class Conditioners:
    c_in: Tensor
    c_out: Tensor
    c_skip: Tensor
    c_noise: Tensor


@dataclass
class SigmaDistributionConfig:
    loc: float
    scale: float
    sigma_min: float
    sigma_max: float


@dataclass
class DenoiserConfig:
    inner_model: InnerModelConfig
    sigma_data: float
    sigma_offset_noise: float


class Denoiser(nn.Module):
    def __init__(self, cfg: DenoiserConfig,
                 is_multiagent: bool = False,
                 num_agents: int = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.inner_model = InnerModel(cfg.inner_model, is_multiagent=is_multiagent, num_agents=num_agents)
        self.sample_sigma_training = None

        self.is_multiagent = is_multiagent
        self.num_agents = num_agents

    @property
    def device(self) -> torch.device:
        return self.inner_model.noise_emb.weight.device

    def setup_training(self, cfg: SigmaDistributionConfig) -> None:
        assert self.sample_sigma_training is None

        def sample_sigma(n: int, device: torch.device):
            s = torch.randn(n, device=device) * cfg.scale + cfg.loc
            return s.exp().clip(cfg.sigma_min, cfg.sigma_max)

        self.sample_sigma_training = sample_sigma

        if self.is_multiagent:
            self.cond_quantiles_probs = torch.arange(self.num_agents, dtype=torch.float32)[1:] / self.num_agents
            self.sigma_dist = torch.distributions.Normal(loc=cfg.loc, scale=cfg.scale)
            probs_st = self.sigma_dist.cdf(torch.log(torch.tensor(cfg.sigma_min)))
            probs_end = self.sigma_dist.cdf(torch.log(torch.tensor(cfg.sigma_max)))
            probs_diff = probs_end - probs_st
            self.cond_quantiles = self.sigma_dist.icdf(probs_st + self.cond_quantiles_probs * probs_diff).detach().exp()
    
    def apply_noise(self, x: Tensor, sigma: Tensor, sigma_offset_noise: float) -> Tensor:
        b, c, _, _ = x.shape 
        offset_noise = sigma_offset_noise * torch.randn(b, c, 1, 1, device=self.device)
        return x + offset_noise + torch.randn_like(x) * add_dims(sigma, x.ndim)

    def compute_conditioners(self, sigma: Tensor) -> Conditioners:
        sigma = (sigma**2 + self.cfg.sigma_offset_noise**2).sqrt()
        c_in = 1 / (sigma**2 + self.cfg.sigma_data**2).sqrt()
        c_skip = self.cfg.sigma_data**2 / (sigma**2 + self.cfg.sigma_data**2)
        c_out = sigma * c_skip.sqrt()
        c_noise = sigma.log() / 4
        return Conditioners(*(add_dims(c, n) for c, n in zip((c_in, c_out, c_skip, c_noise), (4, 4, 4, 1, 1))))

    def compute_model_output(self, noisy_next_obs: Tensor, obs: Tensor, act: Tensor, cs: Conditioners) -> Tensor:
        assert not self.is_multiagent or (act.size(2) == self.num_agents)
        rescaled_obs = obs / self.cfg.sigma_data
        rescaled_noise = noisy_next_obs * cs.c_in
        return self.inner_model(rescaled_noise, cs.c_noise, rescaled_obs, act)
    
    @torch.no_grad()
    def wrap_model_output(self, noisy_next_obs: Tensor, model_output: Tensor, cs: Conditioners) -> Tensor:
        d = cs.c_skip * noisy_next_obs + cs.c_out * model_output
        # Quantize to {0, ..., 255}, then back to [-1, 1]
        d = d.clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)
        return d
    
    @torch.no_grad()
    def denoise(self, noisy_next_obs: Tensor, sigma: Tensor, obs: Tensor, act: Tensor) -> Tensor:
        cs = self.compute_conditioners(sigma)
        model_output = self.compute_model_output(noisy_next_obs, obs, act, cs)
        denoised = self.wrap_model_output(noisy_next_obs, model_output, cs)
        return denoised

    def forward(self, batch: Batch) -> LossAndLogs:
        assert not self.is_multiagent or (batch.act.size(2) == self.num_agents)

        n = self.cfg.inner_model.num_steps_conditioning
        seq_length = batch.obs.size(1) - n

        all_obs = batch.obs.clone()

        # 这里对于multi-agent variant atari使用的是极端的处理方式，因为两个智能体的observation (RGB images)是一致的，所以我们只用考虑一张图
        if self.is_multiagent:   # (b, seq_length, num_agents, c, h, w)
            all_obs = all_obs.mean(dim=2)           # (b, seq_length, c, h, w)

        loss = 0

        for i in range(seq_length):
            obs = all_obs[:, i : n + i]
            next_obs = all_obs[:, n + i]
            act = batch.act[:, i : n + i]           # (b, seq_length, n) or (b, seq_length)
            mask = batch.mask_padding[:, n + i]     # (b, seq_length)

            b, t, c, h, w = obs.shape
            obs = obs.reshape(b, t * c, h, w)

            sigma = self.sample_sigma_training(b, self.device)
            noisy_next_obs = self.apply_noise(next_obs, sigma, self.cfg.sigma_offset_noise)

            cs = self.compute_conditioners(sigma)

            # implement sequential causal graph
            if act.ndim > 2:
                # last_timestep_agent_mask = torch.randint(0, self.num_agents, size=(b,)).to(self.device)
                # action_mask = torch.concat((torch.ones_like(act, device=self.device)[:, :-1], F.one_hot(last_timestep_agent_mask, num_classes=self.num_agents).unsqueeze(1).to(self.device)), dim=1)
                # act_cond = torch.masked_fill(act + 1, (1 - action_mask).to(torch.bool), 0)
                n_agents_mask = torch.sum(sigma.unsqueeze(1) > self.cond_quantiles.to(self.device), dim=1) + 1
                act_mask = torch.zeros(b, 1, *act.shape[2:], device=self.device)
                for k in range(b):
                    active_agents = n_agents_mask[k].item()
                    # 随机采样要激活的 agent 索引
                    active_indices = torch.randperm(self.num_agents)[:active_agents]
                    # 设置对应的 mask 为 1
                    act_mask[k, :, active_indices] = 1

                act_mask = torch.concat((torch.ones_like(act, device=self.device)[:, :-1], act_mask), dim=1)
                act_cond = torch.masked_fill(act + 1, (1 - act_mask).to(torch.bool), 0)
            else:
                act_cond = act.clone()

            model_output = self.compute_model_output(noisy_next_obs, obs, act_cond, cs)

            target = (next_obs - cs.c_skip * noisy_next_obs) / cs.c_out
            loss += F.mse_loss(model_output[mask], target[mask])

            denoised = self.wrap_model_output(noisy_next_obs, model_output, cs)
            all_obs[:, n + i] = denoised

        loss /= seq_length
        return loss, {"loss_denoising": loss.detach()}
