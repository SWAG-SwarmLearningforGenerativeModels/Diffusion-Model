import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from dataloaders import *
from modules import UNet, EMA
import math
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s",
                    level=logging.INFO, datefmt="%I:%M:%S")


class UnconditionDiffusion:
    def __init__(self, args, config, noise_steps=1000, schedule="cosine", beta_start=1e-4, beta_end=0.02, img_size=256):

        self.config = config
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

    def prepare_noise_schedule(self, cosine_s=8e-3):
        if self.schedule == "quad":
            betas = (
                torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.noise_steps, dtype=torch.float64
                               )
                ** 2
            )

        elif self.schedule == "linear":
            betas = torch.linspace(
                self.beta_start, self.beta_end, self.noise_steps, dtype=torch.float64
            )

        elif self.schedule == "cosine":
            timesteps = (
                torch.arange(self.noise_steps + 1, dtype=torch.float64) /
                self.noise_steps + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        return betas

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(
            1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn(
                (n, self.config.data.channels, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat)))
                                             * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = inverse_transform(x)
        return x

    def train(self):
        device = self.device
        dataloader = get_data(self.config)
        model = UNet(c_in=self.config.data.channels, c_out=self.config.data.channels,
                     image_size=self.config.data.image_size).to(self.device)
        print(model.parameters)
        ema = EMA(0.995)
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)

        optimizer = get_optimizer(self.config, model.parameters())
        mse = nn.MSELoss()
        logger = SummaryWriter(self.config.logger)
        l = len(dataloader)

        start_epoch = 0

        if self.config.training.resume_training:
            # load it
            states = torch.load(os.path.join(
                self.args.log_path, "models", 'ckpt_states.pt'))
            model.load_state_dict(states[0], strict=True)
            ema_model.load_state_dict(states[1], strict=True)

            optimizer.load_state_dict(states[2])
            start_epoch = states[3]
            print('Pretrained model loaded successfully')

        for epoch in range(start_epoch, self.config.training.n_epochs):
            logging.info(f"Starting epoch {epoch}:")
            pbar = tqdm(dataloader)
            for i, (images, _) in enumerate(pbar):
                images = images.to(self.device)

                t = self.sample_timesteps(images.shape[0]).to(self.device)
                x_t, noise = self.noise_images(images, t)

                predicted_noise = model(x_t, t)
                loss = mse(noise, predicted_noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ema.step_ema(ema_model, model)

                pbar.set_postfix(MSE=loss.item())
                logger.add_scalar("MSE", loss.item(),
                                  global_step=epoch * l + i)

            if epoch % self.config.training.snapshot_freq == 0:
                states = [
                    model.state_dict(),
                    ema_model.state_dict(),
                    optimizer.state_dict(),
                    epoch
                ]

                if self.config.training.snapshot_sampling:
                    sampled_images = self.sample(
                        model, n=self.config.sampling.batch_size)
                    ema_sampled_images = self.sample(
                        ema_model, n=self.config.sampling.batch_size)

                    save_images(sampled_images, os.path.join(
                        self.args.log_path, 'results', f"{epoch}.jpg"), nrow=math.ceil(math.sqrt(self.config.sampling.batch_size)))
                    save_images(ema_sampled_images, os.path.join(
                        self.args.log_path, 'results', f"{epoch}_ema.jpg"), nrow=math.ceil(math.sqrt(self.config.sampling.batch_size)))

                torch.save(states, os.path.join(self.args.log_path,
                           'models', 'ckpt_states.pt'.format(epoch)))

                torch.save(model, os.path.join(self.args.log_path,
                           'models', 'ckpt_model_{}.pt'.format(epoch)))

                torch.save(ema_model, os.path.join(self.args.log_path,
                           'models', 'ckpt_ema_model_{}.pt'.format(epoch)))
