from dataclasses import dataclass
from typing import List, Tuple
from einops import repeat

import torch
import torch.nn.functional as F
from torch import Tensor

from .denoiser import Denoiser


@dataclass
class DiffusionSamplerConfig:
    num_steps_denoising: int
    sigma_min: float = 2e-3
    sigma_max: float = 5
    rho: int = 7
    order: int = 1
    s_churn: float = 0
    s_tmin: float = 0
    s_tmax: float = float("inf")
    s_noise: float = 1
    agent_order: str = ""


class DiffusionSampler:
    def __init__(self, denoiser: Denoiser, cfg: DiffusionSamplerConfig) -> None:
        self.denoiser = denoiser
        self.cfg = cfg
        
        self.sigmas = build_sigmas(cfg.num_steps_denoising, cfg.sigma_min, cfg.sigma_max, cfg.rho, denoiser.device)

    @torch.no_grad()
    def sample_agent_order(self, num_agents: int, order: str = "default"):
        assert self.cfg.num_steps_denoising % num_agents == 0
        denoising_steps_per_agent = self.cfg.num_steps_denoising // num_agents

        if order == 'default':
            agent_order = torch.arange(num_agents)

        elif order == 'reverse':
            agent_order = torch.flip(torch.arange(num_agents), [0])
        
        elif order == 'random':
            agent_order = torch.randperm(num_agents)
        
        else:
            raise NotImplementedError('Plz specify the agent order for denoising.')
        
        agent_order = repeat(agent_order, 'n -> n k', k=denoising_steps_per_agent).reshape(-1,)
        return agent_order

    @torch.no_grad()
    def sample(self, prev_obs: Tensor, prev_act: Tensor) -> Tuple[Tensor, List[Tensor]]:
        device = prev_obs.device
        # 这里对于multi-agent variant atari使用的是极端的处理方式，因为两个智能体的observation (RGB images)是一致的，所以我们只用考虑一张图
        if prev_obs.ndim == 6 and self.denoiser.is_multiagent:   # (b, seq_length, num_agents, c, h, w)
            prev_obs = prev_obs.mean(dim=2)

        b, t, c, h, w = prev_obs.size()
        prev_obs = prev_obs.reshape(b, t * c, h, w)
        s_in = torch.ones(b, device=device)
        gamma_ = min(self.cfg.s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) # TODO 意义还没搞清楚
        x = torch.randn(b, c, h, w, device=device)
        trajectory = [x]

        # implement sequential causal graph
        if self.denoiser.is_multiagent:
            num_agents = self.denoiser.num_agents
            agent_order = self.sample_agent_order(num_agents, self.cfg.agent_order)
            input_prev_act = prev_act.clone()
            
        # 每一个sigma就是一个denoising step
        for idx, (sigma, next_sigma) in enumerate(zip(self.sigmas[:-1], self.sigmas[1:])):
            gamma = gamma_ if self.cfg.s_tmin <= sigma <= self.cfg.s_tmax else 0
            sigma_hat = sigma * (gamma + 1)
            if gamma > 0:
                eps = torch.randn_like(x) * self.cfg.s_noise
                x = x + eps * (sigma_hat**2 - sigma**2) ** 0.5

            if self.denoiser.is_multiagent:
                current_enable_agent = agent_order[idx]
                action_mask = torch.concat((torch.ones_like(input_prev_act, device=device)[:, :-1], repeat(F.one_hot(current_enable_agent, num_classes=num_agents).to(device), 'n -> b 1 n', b=b)), dim=1)
                prev_act = torch.masked_fill(input_prev_act + 1, (1 - action_mask).to(torch.bool), 0)

            denoised = self.denoiser.denoise(x, sigma, prev_obs, prev_act)
            d = (x - denoised) / sigma_hat
            dt = next_sigma - sigma_hat
            if self.cfg.order == 1 or next_sigma == 0:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                denoised_2 = self.denoiser.denoise(x_2, next_sigma * s_in, prev_obs, prev_act)
                d_2 = (x_2 - denoised_2) / next_sigma
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
            trajectory.append(x)
        return x, trajectory


def build_sigmas(num_steps: int, sigma_min: float, sigma_max: float, rho: int, device: torch.device) -> Tensor:
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    l = torch.linspace(0, 1, num_steps, device=device)
    sigmas = (max_inv_rho + l * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat((sigmas, sigmas.new_zeros(1)))

