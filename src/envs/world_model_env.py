from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Tuple
from einops import repeat

import torch
from torch import Tensor
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader

from coroutines import coroutine
from models.diffusion import Denoiser, DiffusionSampler, DiffusionSamplerConfig
from models.rew_end_model import RewEndModel

ResetOutput = Tuple[torch.FloatTensor, Dict[str, Any]]
StepOutput = Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]
InitialCondition = Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]


@dataclass
class WorldModelEnvConfig:
    horizon: int
    num_batches_to_preload: int
    diffusion_sampler: DiffusionSamplerConfig


class WorldModelEnv:
    def __init__(
        self,
        denoiser: Denoiser,
        rew_end_model: RewEndModel,
        data_loader: DataLoader,
        cfg: WorldModelEnvConfig,
        return_denoising_trajectory: bool = False,
        mode: str = "non-ensemble",
    ) -> None:
        self.sampler = DiffusionSampler(denoiser, cfg.diffusion_sampler)
        self.rew_end_model = rew_end_model
        self.horizon = cfg.horizon
        self.return_denoising_trajectory = return_denoising_trajectory
        self.num_envs = data_loader.batch_sampler.batch_size
        self.generator_init = self.make_generator_init(data_loader, cfg.num_batches_to_preload)
        self.mode = mode # 'ensemble'

    @property
    def device(self) -> torch.device:
        return self.sampler.denoiser.device

    @torch.no_grad()
    def reset(self, **kwargs) -> ResetOutput:
        obs, act, (hx, cx) = self.generator_init.send(self.num_envs)
        self.obs_buffer = obs
        self.act_buffer = act
        self.hx_rew_end = hx
        self.cx_rew_end = cx
        self.ep_len = torch.zeros(self.num_envs, dtype=torch.long, device=obs.device)
        return self.obs_buffer[:, -1], {}

    @torch.no_grad()
    def reset_dead(self, dead: torch.BoolTensor) -> None:
        obs, act, (hx, cx) = self.generator_init.send(dead.sum().item())
        self.obs_buffer[dead] = obs
        self.act_buffer[dead] = act
        self.hx_rew_end[:, dead] = hx
        self.cx_rew_end[:, dead] = cx
        self.ep_len[dead] = 0

    @torch.no_grad()
    def step(self, act: torch.LongTensor) -> StepOutput:
        self.act_buffer[:, -1] = act

        if self.mode == 'ensemble':
            next_obs, denoising_trajectory = self.predict_next_obs()
            next_obs = next_obs[0]
        else:
            next_obs, denoising_trajectory = self.predict_next_obs()
        

        # 因为前面的极端处理，这里要给每个agent拷贝一份next_obs
        if self.sampler.denoiser.is_multiagent:
            next_obs = repeat(next_obs, 'b c h w -> b n c h w', n=self.sampler.denoiser.num_agents)

        rew, end = self.predict_rew_end(next_obs.unsqueeze(1))  # 这里无论单智能体还是多智能体都是(b,) shape的end

        self.ep_len += 1
        trunc = (self.ep_len >= self.horizon).long()

        self.obs_buffer = self.obs_buffer.roll(-1, dims=1)
        self.act_buffer = self.act_buffer.roll(-1, dims=1)
        self.obs_buffer[:, -1] = next_obs

        dead = torch.logical_or(end, trunc)

        info = {}
        if self.return_denoising_trajectory:
            if self.mode == "ensemble":
                denoising_trajectory = [torch.stack(e, dim=1) for e in denoising_trajectory]
                info["denoising_trajectory"] = torch.concat(denoising_trajectory, dim=0)
            else:
                info["denoising_trajectory"] = torch.stack(denoising_trajectory, dim=1)

        if dead.any():
            self.reset_dead(dead)
            info["final_observation"] = next_obs[dead]
            info["burnin_obs"] = self.obs_buffer[dead, :-1]

        return self.obs_buffer[:, -1], rew, end, trunc, info

    @torch.no_grad()
    def predict_next_obs(self) -> Tuple[Tensor, List[Tensor]]:
        if self.mode == 'ensemble':
            return self.sampler.ensemble_sample(self.obs_buffer, self.act_buffer)
        else:
            return self.sampler.sample(self.obs_buffer, self.act_buffer)

    @torch.no_grad()
    def predict_rew_end(self, next_obs: Tensor) -> Tuple[Tensor, Tensor]:
        logits_rew, logits_end, (self.hx_rew_end, self.cx_rew_end) = self.rew_end_model.predict_rew_end(
            self.obs_buffer[:, -1:],
            self.act_buffer[:, -1:],
            next_obs,
            (self.hx_rew_end, self.cx_rew_end),
        )
        rew = Categorical(logits=logits_rew).sample().squeeze(1) - 1.0  # in {-1, 0, 1}
        end = Categorical(logits=logits_end).sample().squeeze(1)
        return rew, end

    @coroutine
    def make_generator_init(
        self,
        data_loader: DataLoader,
        num_batches_to_preload: int,
    ) -> Generator[InitialCondition, None, None]:
        num_dead = yield
        data_iterator = iter(data_loader)

        while True:
            # Preload on device and burnin rew/end model
            obs_, act_, hx_, cx_ = [], [], [], []
            for _ in range(num_batches_to_preload):
                batch = next(data_iterator)
                obs = batch.obs.to(self.device)
                act = batch.act.to(self.device)
                with torch.no_grad():
                    *_, (hx, cx) = self.rew_end_model.predict_rew_end(obs[:, :-1], act[:, :-1], obs[:, 1:])  # Burn-in of rew/end model
                assert hx.size(0) == cx.size(0) == 1
                obs_.extend(list(obs))
                act_.extend(list(act))
                hx_.extend(list(hx[0]))
                cx_.extend(list(cx[0]))

            # Yield new initial conditions for dead envs
            c = 0
            while c + num_dead <= len(obs_):
                obs = torch.stack(obs_[c : c + num_dead])
                act = torch.stack(act_[c : c + num_dead])
                hx = torch.stack(hx_[c : c + num_dead]).unsqueeze(0)
                cx = torch.stack(cx_[c : c + num_dead]).unsqueeze(0)
                c += num_dead
                num_dead = yield obs, act, (hx, cx)
