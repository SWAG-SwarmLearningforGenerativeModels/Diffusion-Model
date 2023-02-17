import os
import math
import copy
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

import logging
from tqdm import tqdm

from utils import (gather, get_model, get_optimizer, inverse_transform)
from dataloaders import *
from unet.modules import EMA


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s",
                    level=logging.INFO, datefmt="%I:%M:%S")


class ConditionDiffusion:
    def __init__(self, args, config, model_config, noise_steps=1000, schedule="linear", beta_start=1e-4, beta_end=0.02, img_size=256):

        self.config = config
        self.model_config = model_config
        self.args = args
        self.device = config.device

        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule = schedule
        self.beta = self.prepare_noise_schedule().to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.mse_loss = nn.MSELoss()

    def prepare_noise_schedule(self, cosine_s=8e-3):
        if self.schedule == "quad":
            beta_t = (torch.linspace(self.beta_start ** 0.5,
                      self.beta_end ** 0.5, self.noise_steps) ** 2)

        elif self.schedule == "linear":
            beta_t = torch.linspace(
                self.beta_start, self.beta_end, self.noise_steps)

        elif self.schedule == "cosine":
            timesteps = (
                torch.arange(self.noise_steps + 1)/self.noise_steps + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            beta_t = betas.clamp(max=0.999)

        return beta_t

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def q_xt_x0(self, x0, t):
        mean = gather(self.alpha_hat, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_hat, t)

        # return mean and variance of image at time t
        return mean, var

    def q_sample(self, x0, t, eps):

        if eps is None:
            eps = torch.randn_like(x0)

        mean, var = self.q_xt_x0(x0, t)

        # noise image and return noised image
        return mean + (var ** 0.5) * eps

    def p_sample(self, eps_model, xt, t, y, eps, cfg_scale=3):
        eps_theta = eps_model(xt, t, y)
        if cfg_scale > 0:
            eps_theta2 = eps_model(xt, t, None)
            eps_theta = torch.lerp(eps_theta2, eps_theta, cfg_scale)
        alpha_hat = gather(self.alpha_hat, t)
        alpha = gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_hat) ** .5
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = gather(self.beta, t)

        return mean + (var ** .5) * eps

    def loss(self, eps_model, x0, y, noise=None):

        batch_size = x0.shape[0]
        t = self.sample_timesteps(batch_size).to(self.device)

        if noise is None:
            noise = torch.randn_like(x0)

        xt = self.q_sample(x0, t, eps=noise)

        if np.random.random() < 0.1:
            y = None

        eps_theta = eps_model(xt, t, y)

        # MSE loss
        return self.mse_loss(noise, eps_theta)

    def sample(self, model, n, labels):
        logging.info(f"Sampling {n} new images....")
        with torch.no_grad():
            x = torch.randn((n, self.config.data.channels,
                            self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                time = (torch.ones(n) * i).long().to(self.device)
                if i > 1:
                    eps = torch.randn_like(x, device=x.device)
                else:
                    eps = torch.zeros_like(x, device=x.device)

                x = self.p_sample(eps_model=model, xt=x,
                                  t=time, y=labels, eps=eps)

        x = inverse_transform(x)
        return x

    def train(self):
        dataloader = get_data(self.config)
        model = get_model(args=self.args, config=self.config,
                          model_config=self.model_config)

        model.to(self.device)
        print(model.parameters)

        ema = EMA(0.995)
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)

        optimizer = get_optimizer(self.config, model.parameters())

        logger = SummaryWriter(self.config.logger)
        l = len(dataloader)

        start_epoch = 0

        if self.args.resume_training:
            # load it
            states = torch.load(os.path.join(
                self.args.log_path, "models", 'ckpt_states.pt'))
            model.load_state_dict(states[0], strict=True)
            ema_model.load_state_dict(states[1], strict=True)

            optimizer.load_state_dict(states[2])
            start_epoch = states[3]
            print('Pretrained model loaded successfully')

        for epoch in range(start_epoch, self.config.training.n_epochs):
            model.train()
            logging.info(f"Starting epoch {epoch}:")
            pbar = tqdm(dataloader)
            for i, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                noise = torch.randn_like(images)

                loss = self.loss(eps_model=model, x0=images,
                                 y=labels, noise=noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ema.step_ema(ema_model, model)

                pbar.set_postfix(MSE=loss.item())
                logger.add_scalar("MSE", loss.item(),
                                  global_step=epoch * l + i)

            if epoch % self.config.training.snapshot_freq == 0:
                model.eval()
                with torch.no_grad():
                    states = [
                        model.state_dict(),
                        ema_model.state_dict(),
                        optimizer.state_dict(),
                        epoch
                    ]

                    if self.config.training.snapshot_sampling:
                        labels = torch.arange(self.config.data.num_classes).repeat(
                            1, self.config.sampling.batch_size).squeeze().long().to(self.device)
                        sampled_images = self.sample(
                            model, n=len(labels), labels=labels)
                        ema_sampled_images = self.sample(
                            ema_model, n=len(labels), labels=labels)

                        save_image(sampled_images, os.path.join(
                            self.args.log_path, 'results', f"{epoch}.jpg"), nrow=self.config.data.num.classes)
                        save_image(ema_sampled_images, os.path.join(
                            self.args.log_path, 'results', f"{epoch}_ema.jpg"), nrow=self.config.data.num.classes)

                    torch.save(states, os.path.join(self.args.log_path,
                                                    'models', 'ckpt_states.pt'.format(epoch)))

                    torch.save(model, os.path.join(self.args.log_path,
                                                   'models', 'ckpt_model_{}.pt'.format(epoch)))

                    torch.save(ema_model, os.path.join(self.args.log_path,
                                                       'models', 'ckpt_ema_model_{}.pt'.format(epoch)))
