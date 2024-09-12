from typing import Tuple, Sequence, Dict, Union, Optional
import numpy as np
import math
import torch
import torch.nn as nn
import collections
import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from diffusers.optimization import get_scheduler
from attr import attrs, attrib, validators

# env import
import gym
from gym import spaces
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from skvideo.io import vwrite
from IPython.display import Video
import gdown
import os

from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy_state_pusht_demo import ConditionalUnet1D


def replace_bn_with_gn(
        root_module: nn.Module,
        features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


def model_from_rgb_to_gray_scale(
        root_module: nn.Module,
):
    """
    Convert all RGB input layers to grayscale.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.Conv2d),
        func=lambda x: nn.Conv2d(
            in_channels=1,
            out_channels=x.out_channels,
            kernel_size=x.kernel_size,
            stride=x.stride,
            padding=x.padding,
            dilation=x.dilation,
            groups=x.groups,
            bias=x.bias is not None)
    )
    return root_module


@attrs(auto_attribs=True, frozen=True, slots=True)
class ModelConfig:
    prediction_horizon: int = 16
    static_path_horizon: int = 20
    observation_horizon: int = 2
    action_dim: int = 3


def prepare_training(model_config: ModelConfig):

    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)
    vision_encoder = model_from_rgb_to_gray_scale(vision_encoder)

    vision_feature_dim = 512        # resnet18 feature dim
    # observation feature has 514 dims in total per step
    obs_dim = vision_feature_dim + model_config.action_dim + model_config.action_dim * model_config.static_path_horizon

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=model_config.action_dim,
        global_cond_dim=obs_dim * model_config.observation_horizon
    )

    # the final arch has 2 parts
    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net
    })

    # static_map_size = 540   # TODO: find out
    # # demo
    # with torch.no_grad():
    #     # example inputs
    #     image = torch.zeros((1, observation_horizon, 1, static_map_size, static_map_size))
    #     agent_pos = torch.zeros((1, observation_horizon, action_dim))
    #     # vision encoder
    #     image_features = nets['vision_encoder'](
    #         image.flatten(end_dim=1))
    #     # (2,512)
    #     image_features = image_features.reshape(*image.shape[:2], -1)
    #     # (1,2,512)
    #     obs = torch.cat([image_features, agent_pos],dim=-1)
    #     # (1,2,514)
    #
    #     noised_action = torch.randn((1, pred_horizon, action_dim))
    #     diffusion_iter = torch.zeros((1,))
    #
    #     # the noise prediction network
    #     # takes noisy action, diffusion iteration and observation as input
    #     # predicts the noise added to action
    #     noise = nets['noise_pred_net'](
    #         sample=noised_action,
    #         timestep=diffusion_iter,
    #         global_cond=obs.flatten(start_dim=1))
    #
    #     # illustration of removing noise
    #     # the actual noise removal is performed by NoiseScheduler
    #     # and is dependent on the diffusion noise schedule
    #     denoised_action = noised_action - noise

    # for this demo, we use DDPMScheduler with 100 diffusion iterations
    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )

    # device transfer
    device = torch.device('cuda')
    _ = nets.to(device)

    return noise_scheduler, nets, device


def prepare_data(data_file):
    # TODO: figure out
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )
    return



def train(
        model_config,
        dataloader,
        noise_scheduler,
        nets,
        device,
):
    num_epochs = 100

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        parameters=nets.parameters(),
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for _ in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    nimage = nbatch['image'][:, :model_config.observation_horizon].to(device)
                    nagent_pos = nbatch['agent_pos'][:, :model_config.observation_horizon].to(device)
                    static_path = nbatch['static_path'].to(device)
                    naction = nbatch['action'].to(device)
                    B = nagent_pos.shape[0]

                    # encoder vision features
                    image_features = nets['vision_encoder'](
                        nimage.flatten(end_dim=1))
                    image_features = image_features.reshape(*nimage.shape[:2],-1)

                    # concatenate vision feature and low-dim obs
                    obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                    obs_cond = obs_features.flatten(start_dim=1)
                    # (B, obs_horizon * obs_dim)

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)

                    # predict the noise residual
                    noise_pred = nets['noise_pred_net'](
                        noisy_actions, timesteps, global_cond=obs_cond)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(nets.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))

    # Weights of the EMA model
    # is used for inference
    ema_nets = nets
    ema.copy_to(ema_nets.parameters())


def inference():
    # limit enviornment interaction to 200 steps before termination
    max_steps = 200

    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(
        [obs] * obs_horizon, maxlen=obs_horizon)
    # save visualization and rewards
    imgs = [env.render(mode='rgb_array')]
    rewards = list()
    done = False
    step_idx = 0

    with tqdm(total=max_steps, desc="Eval PushTImageEnv") as pbar:
        while not done:
            B = 1
            # stack the last obs_horizon number of observations
            images = np.stack([x['image'] for x in obs_deque])
            agent_poses = np.stack([x['agent_pos'] for x in obs_deque])

            # normalize observation
            nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
            # images are already normalized to [0,1]
            nimages = images

            # device transfer
            nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
            # (2,3,96,96)
            nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
            # (2,2)

            # infer action
            with torch.no_grad():
                # get image features
                image_features = ema_nets['vision_encoder'](nimages)
                # (2,512)

                # concat with low-dim observations
                obs_features = torch.cat([image_features, nagent_poses], dim=-1)

                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                # initialize action from Guassian noise
                noisy_action = torch.randn(
                    (B, pred_horizon, action_dim), device=device)
                naction = noisy_action

                # init scheduler
                noise_scheduler.set_timesteps(num_diffusion_iters)

                for k in noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = ema_nets['noise_pred_net'](
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample

            # unnormalize action
            naction = naction.detach().to('cpu').numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=stats['action'])

            # only take action_horizon number of actions
            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end,:]
            # (action_horizon, action_dim)

            # execute action_horizon number of steps
            # without replanning
            for i in range(len(action)):
                # stepping env
                obs, reward, done, _, info = env.step(action[i])
                # save observations
                obs_deque.append(obs)
                # and reward/vis
                rewards.append(reward)
                imgs.append(env.render(mode='rgb_array'))

                # update progress bar
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if step_idx > max_steps:
                    done = True
                if done:
                    break

    # print out the maximum target coverage
    print('Score: ', max(rewards))

    # visualize
    from IPython.display import Video
    vwrite('vis.mp4', imgs)
    Video('vis.mp4', embed=True, width=256, height=256)

