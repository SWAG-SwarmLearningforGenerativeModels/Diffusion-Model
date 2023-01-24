
import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, num_groups=32, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(1 if in_channels == 1 else num_groups, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(1 if in_channels == 1 else num_groups, out_channels),
        )

    def forward(self, x, t=None):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, is_maxpool=True, has_attn=False, emb_dim=256):
        super().__init__()
        if is_maxpool:
            DownSample = nn.MaxPool2d(2)
        else:
            DownSample = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=1)

        self.down_conv = nn.Sequential(DownSample,
                                       DoubleConv(
                                           in_channels, in_channels, residual=True),
                                       DoubleConv(
                                           in_channels, out_channels),
                                       )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

        if has_attn:
            self.attn_layer = SelfAttention(out_channels, image_size)
        else:
            self.attn_layer = nn.Identity()

    def forward(self, x, t):
        x = self.down_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(
            1, 1, x.shape[-2], x.shape[-1])
        x = x + emb

        return self.attn_layer(x)


class MiddleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, has_attn=False):
        super().__init__()
        self.res1 = DoubleConv(in_channels, out_channels)
        self.res2 = DoubleConv(out_channels, out_channels)
        self.res3 = DoubleConv(out_channels, out_channels)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, out_image_dim, num_groups=32, emb_dim=256, has_attn=False, is_output=False, is_upsample=False):
        super().__init__()

        if is_upsample:
            self.up = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels//2, in_channels//2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        if not is_output:
            self.down_conv = nn.Sequential(nn.Conv2d(out_channels, out_channels//2,
                                                     kernel_size=3, stride=1, padding=1, bias=False),
                                           nn.GroupNorm(1 if in_channels == 1 else num_groups, out_channels//2))
        else:
            self.down_conv = nn.Identity()

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

        if has_attn:
            self.attn_layer = SelfAttention(out_channels, out_image_dim)
        else:
            self.attn_layer = nn.Identity()

    def forward(self, x, skip_x, t):
        if x.shape[-1] != skip_x.shape[-1]:
            x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(
            1, 1, x.shape[-2], x.shape[-1])
        x = x + emb
        return self.down_conv(self.attn_layer(x))


class UNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, n_channels=[64, 128, 256, 512, 1024], time_dim=256, image_size=224, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.channels = n_channels

        in_channel = c_in
        n_resolution = len(self.channels)

        attn = [False]*(n_resolution-1)  # True for attention layers
        up_isfinal = [False]*(n_resolution-2)
        up_isfinal.append(True)

        # Down
        down = []
        down_mul = 2
        down_channels = self.channels
        for i in range(n_resolution-1):
            if in_channel == c_in:
                out_channel = down_channels[i]
                down.append(DoubleConv(in_channel, out_channel))
                in_channel = out_channel
            out_channel = down_channels[i+1]
            down.append(Down(in_channel, out_channel,
                        image_size//down_mul, has_attn=attn[i]))
            down_mul *= 2
            in_channel = out_channel

        self.down = nn.ModuleList(down)

        # Middle
        middle_channel = self.channels
        in_channel = middle_channel[-1]
        out_channel = middle_channel[-2]
        self.middle = MiddleBlock(in_channel, out_channel)

        # Up
        up = []
        up_mul = down_mul//4
        up_channels = self.channels
        up_channels.reverse()
        for i in range(n_resolution-1):
            out_channel = up_channels[i+1]
            up.append(Up(in_channel, out_channel,
                         image_size//up_mul, has_attn=attn[i], is_output=up_isfinal[i]))
            up_mul //= 2
            in_channel = out_channel

        up.append(nn.Conv2d(in_channel, c_out, kernel_size=1))

        self.up = nn.ModuleList(up)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        # Encoder
        h = []  # store conv output for skip connection
        for m in self.down:
            x = m(x, t)
            h.append(x)

        # Bottle Neck
        x = self.middle(x)

        # Decoder
        h.reverse()
        for idx in range(len(self.up)-1):
            skip_x = h[idx+1]
            x = self.up[idx](x, skip_x, t)

        # Ouput Layer
        out = self.up[-1](x)

        return out


class UNet_conditional(nn.Module):
    def __init__(self, c_in=1, c_out=1, n_channels=[64, 128, 256, 512, 1024], time_dim=256, num_classes=None, image_size=224, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.channels = n_channels

        in_channel = c_in
        n_resolution = len(self.channels)
        attn = [False]*(n_resolution-1)  # True for attention layers
        up_isfinal = [False]*(n_resolution-2)
        up_isfinal.append(True)

        # Down
        down = []
        down_mul = 2
        down_channels = self.channels
        for i in range(n_resolution-1):
            if in_channel == c_in:
                out_channel = down_channels[i]
                down.append(DoubleConv(in_channel, out_channel))
                in_channel = out_channel
            out_channel = down_channels[i+1]
            down.append(Down(in_channel, out_channel,
                        image_size//down_mul, has_attn=attn[i]))
            down_mul *= 2
            in_channel = out_channel

        self.down = nn.ModuleList(down)

        # Middle
        middle_channel = self.channels
        in_channel = middle_channel[-1]
        out_channel = middle_channel[-2]
        self.middle = MiddleBlock(in_channel, out_channel)

        # Up
        up = []
        up_mul = down_mul//4
        up_channels = self.channels
        up_channels.reverse()
        for i in range(n_resolution-1):
            out_channel = up_channels[i+1]
            up.append(Up(in_channel, out_channel,
                         image_size//up_mul, has_attn=attn[i], is_output=up_isfinal[i]))
            up_mul //= 2
            in_channel = out_channel

        up.append(nn.Conv2d(in_channel, c_out, kernel_size=1))

        self.up = nn.ModuleList(up)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        # Add Label to Time Embedding
        if y is not None:
            t += self.label_emb(y)

        # Encoder
        h = []  # store conv output for skip connection
        for m in self.down:
            x = m(x, t)
            h.append(x)

        # Bottle Neck
        x = self.middle(x)

        # Decoder
        h.reverse()
        for idx in range(len(self.up)-1):
            skip_x = h[idx+1]
            x = self.up[idx](x, skip_x, t)

        # Output Layer
        out = self.up[-1](x)

        return out
