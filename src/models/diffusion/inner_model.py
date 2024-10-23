from dataclasses import dataclass
from typing import List, Optional
from einops import rearrange

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import Conv3x3, FourierFeatures, GroupNorm, UNet


@dataclass
class InnerModelConfig:
    img_channels: int
    num_steps_conditioning: int
    cond_channels: int
    depths: List[int]
    channels: List[int]
    attn_depths: List[bool]
    num_actions: Optional[int] = None


class InnerModel(nn.Module):
    def __init__(self, cfg: InnerModelConfig,
                 is_multiagent: bool = False,
                 num_agents: int = None) -> None:
        super().__init__()
        self.noise_emb = FourierFeatures(cfg.cond_channels)
        self.is_multiagent = is_multiagent
        if not is_multiagent:
            self.act_emb = nn.Sequential(
                nn.Embedding(cfg.num_actions, cfg.cond_channels // cfg.num_steps_conditioning),
                nn.Flatten(),  # b t e -> b (t e)
            )
        else:
            # implement our sequential causal graph
            ## Option 1
            self.act_emb = nn.Embedding(cfg.num_actions + 1, cfg.cond_channels // cfg.num_steps_conditioning, padding_idx=0)
            self.aggregate_conv1d = nn.Conv1d(num_agents, 1, kernel_size=1, stride=1, padding=0)

            ## Option 2
            # self.act_emb = nn.Sequential(
            #     nn.Linear(cfg.num_actions * num_agents, cfg.cond_channels // cfg.num_steps_conditioning),
            #     nn.SiLU(),
            #     nn.Linear(cfg.cond_channels // cfg.num_steps_conditioning, cfg.cond_channels // cfg.num_steps_conditioning),
            #     nn.Flatten(),  # b t e -> b (t e)
            # )

        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
            nn.SiLU(),
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
        )
        self.conv_in = Conv3x3((cfg.num_steps_conditioning + 1) * cfg.img_channels, cfg.channels[0])

        self.unet = UNet(cfg.cond_channels, cfg.depths, cfg.channels, cfg.attn_depths)

        self.norm_out = GroupNorm(cfg.channels[0])
        self.conv_out = Conv3x3(cfg.channels[0], cfg.img_channels)
        nn.init.zeros_(self.conv_out.weight)
    
    def compute_action_cond(self, act: Tensor) -> Tensor:
        # implement our sequential causal graph
        ## Option 1
        act_cond = self.act_emb(act)
        b, t, n, d = act_cond.shape
        act_cond = rearrange(act_cond, 'b t n d -> (b t) n d')
        act_cond = self.aggregate_conv1d(act_cond)
        act_cond = rearrange(act_cond.squeeze(1), '(b t) d -> b (t d)', b=b, t=t, d=d)

        ## Option 2
        ### 以Option 2处理时应默认act给的就是one-hot

        return act_cond

    def forward(self, noisy_next_obs: Tensor, c_noise: Tensor, obs: Tensor, act: Tensor) -> Tensor:
        if self.is_multiagent:
            act_cond = self.compute_action_cond(act)
            cond = self.cond_proj(self.noise_emb(c_noise) + act_cond)
        else:
            cond = self.cond_proj(self.noise_emb(c_noise) + self.act_emb(act))
        
        x = self.conv_in(torch.cat((obs, noisy_next_obs), dim=1))
        x, _, _ = self.unet(x, cond)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x
