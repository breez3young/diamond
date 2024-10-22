from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import gymnasium
from gymnasium.vector import AsyncVectorEnv
import numpy as np
import supersuit.vector
import torch
from torch import Tensor

from .atari_preprocessing import AtariPreprocessing
from pettingzoo.atari import boxing_v2
from pettingzoo.atari import pong_v3
import supersuit
import ipdb
import matplotlib.pyplot as plt

def make_atari_env(
    id: str,
    num_envs: int,
    device: torch.device,
    done_on_life_loss: bool,
    size: int,
    max_episode_steps: Optional[int],
    is_multiagent: bool = False,
) -> TorchEnv:
    def env_fn():
        if not is_multiagent:
            env = gymnasium.make(
                id,
                full_action_space=False,
                frameskip=1,
                render_mode="rgb_array",
                max_episode_steps=max_episode_steps,
            )
            env = AtariPreprocessing(
                env=env,
                noop_max=30,
                frame_skip=4,
                screen_size=size,
            )

        else:
            assert id in ['pong_v3', 'boxing_v2']
            if id == 'pong_v3':
                env = pong_v3.parallel_env(
                    # frameskip=1,
                    render_mode="rgb_array",
                    # max_episode_steps=max_episode_steps,
                )
            
            elif id == 'boxing_v2':
                env = boxing_v2.parallel_env(
                    # frameskip=1,
                    render_mode="rgb_array",
                    # max_episode_steps=max_episode_steps,
                )
            
            else:
                raise NotImplementedError("!")
            
            # apply same wrapper
            # with multi-agent variant Atari, we cannnot add noop wrapper
            env = supersuit.max_observation_v0(env, 2)
            env = supersuit.frame_skip_v0(env, 4)
            env = supersuit.resize_v1(env, size, size)

        return env
    
    if not is_multiagent:
        env = AsyncVectorEnv([env_fn for _ in range(num_envs)])
    else:
        env = env_fn()
        env = supersuit.pettingzoo_env_to_vec_env_v1(env)   # 这里有多少个agent，就已经转换为num_envs = num_agents
        env = supersuit.concat_vec_envs_v1(env, num_vec_envs=num_envs)
        # obs, infos = env.reset(seed=42)
        # plt.imsave('test0.png', obs[0])

    # The AsyncVectorEnv resets the env on termination, which means that it will
    # reset the environment if we use the default AtariPreprocessing of gymnasium with
    # terminate_on_life_loss=True (which means that we will only see the first life).
    # Hence a separate wrapper for life_loss, coming after the AsyncVectorEnv.
    if done_on_life_loss and not is_multiagent:   # 这个life loss，只是atari做简化的一个方式，根据网上查阅的信息，为了缩短episode，第一条命失去即视为整个游戏结束（重新启动游戏）
        env = DoneOnLifeLoss(env)

    if is_multiagent:
        env = MATorchEnv(env, device)
    else:
        env = TorchEnv(env, device)

    return env


class DoneOnLifeLoss(gymnasium.Wrapper):
    def __init__(self, env: AsyncVectorEnv) -> None:
        super().__init__(env)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        obs, rew, end, trunc, info = self.env.step(actions)
        life_loss = info["life_loss"]
        if life_loss.any():
            end[life_loss] = True
            info["final_observation"] = obs
        return obs, rew, end, trunc, info


class TorchEnv(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env, device: torch.device) -> None:
        super().__init__(env)
        self.device = device
        self.num_envs = env.observation_space.shape[0]
        self.num_actions = env.unwrapped.single_action_space.n
        b, h, w, c = env.observation_space.shape
        self.observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(b, c, h, w))

    def reset(self, *args, **kwargs) -> Tuple[Tensor, Dict[str, Any]]:
        obs, info = self.env.reset(*args, **kwargs)
        return self._to_tensor(obs), info

    def step(self, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
        obs, rew, end, trunc, info = self.env.step(actions.cpu().numpy())
        dead = np.logical_or(end, trunc)
        if dead.any():
            info["final_observation"] = self._to_tensor(np.stack(info["final_observation"][dead]))
        obs, rew, end, trunc = (self._to_tensor(x) for x in (obs, rew, end, trunc))
        return obs, rew, end, trunc, info

    def _to_tensor(self, x: Tensor) -> Tensor:
        if x.ndim == 4:
            return torch.tensor(x, device=self.device).div(255).mul(2).sub(1).permute(0, 3, 1, 2).contiguous()
        elif x.dtype is np.dtype("bool"):
            return torch.tensor(x, dtype=torch.uint8, device=self.device)
        else:
            return torch.tensor(x, dtype=torch.float32, device=self.device)
        
class MATorchEnv(gymnasium.Wrapper):
    def __init__(self, env: supersuit.vector.concat_vec_env.ConcatVecEnv, device: torch.device) -> None:
        super().__init__(env)
        self.device = device
        # self.agents = env.par_env.possible_agents
        self.num_agents = env.vec_envs[0].num_envs
        self.num_envs = len(env.vec_envs)     # env.observation_space.shape[0]
        self.num_actions = env.action_space.n               # number of actions per agent
        h, w, c = env.observation_space.shape
        self.observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(self.num_envs, self.num_agents, c, h, w))

    def reset(self, *args, **kwargs) -> Tuple[Tensor, Dict[str, Any]]:
        obs, info = self.env.reset(*args, **kwargs)
        return self._to_tensor(obs), info

    def step(self, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
        if actions.ndim == 2:
            actions = actions.reshape(-1, )

        obs, rew, end, trunc, info = self.env.step(actions.cpu().numpy())
        dead = np.logical_or(end, trunc)
        new_info = {}
        if dead.any():
            # 这里目前应该只考虑了单个env里面的情况，如果是多个vec env还不知道如何解决
            terminal_obs = []
            for indi_info in info:
                if 'terminal_observation' in indi_info:
                    terminal_obs.append(indi_info['terminal_observation'])
            terminal_obs = np.stack(terminal_obs)
            new_info['final_observation'] = self._to_tensor(terminal_obs)
            # info["final_observation"] = self._to_tensor(np.stack(info["final_observation"][dead]))
        obs, rew, end, trunc = (self._to_tensor(x) for x in (obs, rew, end, trunc))
        return obs, rew, end, trunc, new_info

    def _to_tensor(self, x: Tensor) -> Tensor:
        if x.ndim == 4:
            return torch.tensor(x, device=self.device).div(255).mul(2).sub(1).permute(0, 3, 1, 2).reshape(x.shape[0] // self.num_agents, self.num_agents, *self.observation_space.shape[-3:]).contiguous()
        elif x.dtype is np.dtype("bool") or x.dtype is np.dtype("uint8"):
            return torch.tensor(x, dtype=torch.uint8, device=self.device).reshape(self.num_envs, self.num_agents,)
        else:
            return torch.tensor(x, dtype=torch.float32, device=self.device).reshape(self.num_envs, self.num_agents,)
