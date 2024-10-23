import random
from typing import Generator, Tuple, Union

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from . import coroutine
from envs import TorchEnv, WorldModelEnv


@coroutine
def make_env_loop(
    env: Union[TorchEnv, WorldModelEnv], model: nn.Module, epsilon: float = 0.0
) -> Generator[Tuple[torch.Tensor, ...], int, None]:
    num_steps = yield

    is_ma = model.is_ma
    num_agents = model.num_agents if is_ma else None

    if not is_ma:
        hx = torch.zeros(env.num_envs, model.lstm_dim, device=model.device)
        cx = torch.zeros(env.num_envs, model.lstm_dim, device=model.device)
    else:
        hx = torch.zeros(num_agents, env.num_envs, model.lstm_dim, device=model.device)
        cx = torch.zeros(num_agents, env.num_envs, model.lstm_dim, device=model.device)

    seed = random.randint(0, 2**31 - 1)
    obs, _ = env.reset(seed=[seed + i for i in range(env.num_envs)])

    while True:
        hx, cx = hx.detach(), cx.detach()
        all_ = []
        infos = []
        n = 0

        while n < num_steps:
            if is_ma:
                # act = []
                # val = []
                # logits_act = []
                # for m_id, m in enumerate(model.agents):
                #     m_logits_act, m_val, (hx[m_id], cx[m_id]) = m.predict_act_value(obs[:, m_id], (hx[m_id], cx[m_id]))
                #     action = Categorical(logits=m_logits_act).sample()

                #     if random.random() < epsilon:
                #         action = torch.randint(low=0, high=env.num_actions, size=(obs.size(0),), device=obs.device)

                #     act.append(action)
                #     val.append(m_val)
                #     logits_act.append(m_logits_act)

                # act = torch.stack(act, dim=1)
                # val = torch.stack(val, dim=1)
                # logits_act = torch.stack(logits_act, dim=1)
                logits_act, val, (hx, cx) = model.predict_act_value(obs, (hx, cx))
                act = Categorical(logits=logits_act).sample()

                if random.random() < epsilon:
                    act = torch.randint(low=0, high=env.num_actions, size=(obs.size(0), obs.size(1)), device=obs.device)

            else:
                logits_act, val, (hx, cx) = model.predict_act_value(obs, (hx, cx))
                act = Categorical(logits=logits_act).sample()

                if random.random() < epsilon:
                    act = torch.randint(low=0, high=env.num_actions, size=(obs.size(0),), device=obs.device)

            next_obs, rew, end, trunc, info = env.step(act)

            if n > 0:
                val_bootstrap = val.detach().clone()
                if dead.any():
                    val_bootstrap[dead] = val_final_obs
                all_[-1][-1] = val_bootstrap

            if is_ma and end.ndim == 2:
                dead = torch.logical_or(end, trunc).any(dim=-1) # 这里的end, trunc默认shape为(num_envs, num_agents)
            else:
                dead = torch.logical_or(end, trunc)             # 这里的end, trunc默认shape为(num_envs,)

            if dead.any():
                with torch.no_grad():
                    if is_ma:
                        _, val_final_obs, _ = model.predict_act_value(info["final_observation"], (hx[:, dead], cx[:, dead]))
                    else:
                        _, val_final_obs, _ = model.predict_act_value(info["final_observation"], (hx[dead], cx[dead]))
                reset_gate = 1 - dead.float().unsqueeze(1)
                if is_ma:
                    reset_gate = reset_gate.unsqueeze(0).repeat(num_agents, 1, 1)
                hx = hx * reset_gate
                cx = cx * reset_gate
                if "burnin_obs" in info:
                    burnin_obs = info["burnin_obs"]
                    for i in range(burnin_obs.size(1)):
                        if is_ma:
                            _, _, (hx[:, dead], cx[:, dead]) = model.predict_act_value(burnin_obs[:, i], (hx[:, dead], cx[:, dead]))
                        else:
                            _, _, (hx[dead], cx[dead]) = model.predict_act_value(burnin_obs[:, i], (hx[dead], cx[dead]))

            all_.append([obs, act, rew, end, trunc, logits_act, val, None])
            infos.append(info)

            obs = next_obs
            n += 1

        with torch.no_grad():
            _, val_bootstrap, _ = model.predict_act_value(next_obs, (hx, cx))  # do not update hx/cx

        if dead.any():
            val_bootstrap[dead] = val_final_obs

        all_[-1][-1] = val_bootstrap

        # 这里每一项前两维都是(num_envs, trajectory_length)
        all_obs, act, rew, end, trunc, logits_act, val, val_bootstrap = (torch.stack(x, dim=1) for x in zip(*all_))

        num_steps = yield all_obs, act, rew, end, trunc, logits_act, val, val_bootstrap, infos
