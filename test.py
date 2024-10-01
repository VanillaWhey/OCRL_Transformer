from ocatari.core import OCAtari
import torch
import numpy as np

from torch import nn

from ocrltransformer.wrappers import OCWrapper, OCMultiGame
from vit_pytorch import ViT
from gymnasium import logger

if __name__ == '__main__':
    logger.set_level(10)

    # env = OCAtari("ALE/Atlantis",
    #               mode="ram", obs_mode="obj",
    #               render_mode="human", hud=False)
    # env = OCWrapper(env)
    env = OCMultiGame("ALE/Phoenix", "ALE/Assault", "ALE/DemonAttack", "ALE/Galaxian",
                      exclude="ALE/Assault", wrappers=[OCWrapper],
                      mode="ram", obs_mode="obj", render_mode="rgb_array", hud=False)

    print(env.unwrapped.get_action_meanings())

    # dims = env.observation_space.feature_space.shape
    #
    # print(dims, obs.shape)
    # shape = obs.shape
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("Using device", device)

    # vit = ViT(
    #     image_size=84,
    #     patch_size=7,
    #     num_classes=1000,
    #     dim=1024,
    #     depth=6,
    #     heads=16,
    #     mlp_dim=2048,
    #     dropout=0.1,
    #     emb_dropout=0.1,
    #     channels=4,
    # ).to(device)
    options = {}
    last = 4
    for i in range(last):
        if i == last - 1:
            options["test"] = True
        obs, info = env.reset(options=options)
        done = False
        while not done:
            im = env.render()
            action = env.action_space.sample()
            # cuda_obs = torch.from_numpy(obs).type(torch.float32).unsqueeze(0).to(device)
            # out = vit(obs.unsqueeze(0).view(torch.float32).to(device))
            # out = transformer(cuda_obs)
            # o = lin(cuda_obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated