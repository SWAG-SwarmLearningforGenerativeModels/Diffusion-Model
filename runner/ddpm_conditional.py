import os
import copy
import numpy as np
import torch
import torch.nn as nn
from utils import *
from dataloaders import *
from modules import UNet_conditional, EMA
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s",
                    level=logging.INFO, datefmt="%I:%M:%S")


class ConditionDiffusion:
    def __init__(self, args, config, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256):

        self.config = config
        self.args = args
        self.device = config.device

        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(
            1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn(
                (n, self.config.data.channels, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(
                        uncond_predicted_noise, predicted_noise, cfg_scale)
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
        model = UNet_conditional(num_classes=self.config.data.num_classes,
                                 image_size=self.config.data.image_size,
                                 c_in=self.config.data.channels,
                                 c_out=self.config.data.channels).to(self.device)
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
            for i, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)
                t = self.sample_timesteps(images.shape[0]).to(self.device)
                x_t, noise = self.noise_images(images, t)
                if np.random.random() < 0.1:
                    labels = None
                predicted_noise = model(x_t, t, labels)
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
                    labels = torch.arange(self.config.data.num_classes).repeat(
                        1, self.config.sampling.batch_size).squeeze().long().to(device)
                    sampled_images = self.sample(
                        model, n=len(labels), labels=labels)
                    ema_sampled_images = self.sample(
                        ema_model, n=len(labels), labels=labels)

                    save_images(sampled_images, os.path.join(
                        self.args.log_path, 'results', f"{epoch}.jpg"), nrow=self.config.data.num_classes)
                    save_images(ema_sampled_images, os.path.join(
                        self.args.log_path, 'results', f"{epoch}_ema.jpg"), nrow=self.config.data.num_classes)

                torch.save(states, os.path.join(self.args.log_path,
                           'models', 'ckpt_states.pt'.format(epoch)))

                torch.save(model, os.path.join(self.args.log_path,
                           'models', 'ckpt_model_{}.pt'.format(epoch)))

                torch.save(ema_model, os.path.join(self.args.log_path,
                           'models', 'ckpt_ema_model_{}.pt'.format(epoch)))
