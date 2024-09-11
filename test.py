import torch
import numpy as np

from ocatari.core import OCAtari

from transformer import Encoder

from wrappers import OCWrapper


if __name__ == '__main__':
    env = OCAtari("ALE/Pong",
                  mode="ram", obs_mode="obj",
                  render_mode="human", hud=False)
    env = OCWrapper(env)
    obs, info = env.reset()
    done = False

    dims = env.observation_space.feature_space.shape

    print(dims, obs.shape)
    shape = obs.shape

    # transformer = torch.nn.Sequential(
    #     Encoder(dims[1],
    #                       1, 2, CUDA=True),
    #     torch.nn.Flatten(),
    #     torch.nn.Linear(in_features=np.prod(dims), out_features=6, device="cuda:0")
    # )

    while not done:
        env.render()
        action = env.action_space.sample()
        # cuda_obs = torch.from_numpy(obs).type(torch.float32).unsqueeze(0).cuda(0)
        # out = transformer(cuda_obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated