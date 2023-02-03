from torch.nn.functional import one_hot
import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
from modules import *


class ClassifierFreeGuidance(pl.LightningModule):
    def __init__(self, num_classes, class_embed_size, in_size, t_range, img_depth):
        super().__init__()
        self.beta_small = 1e-4
        self.beta_large = 0.02
        self.t_range = t_range
        self.in_size = in_size
        # one_hot класс отображается в свой эмбеддинг
        self.num_classes = num_classes
        self.class_embed_size = class_embed_size
        self.linear_map_class = nn.Sequential(
            nn.Linear(num_classes, class_embed_size),
            nn.ReLU(),
            nn.Linear(class_embed_size, class_embed_size),
        )

        bilinear = True
        self.inc = DoubleConv(img_depth + class_embed_size, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, img_depth)
        self.sa1 = SAWrapper(256, 8)
        self.sa2 = SAWrapper(256, 4)
        self.sa3 = SAWrapper(128, 8)

    def pos_encoding(self, t, channels, embed_size):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.view(-1, channels, 1, 1).repeat(1, 1, embed_size, embed_size)

    def forward(self, x, t):
        """
        Model is U-Net with added positional encodings and self-attention layers.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1) + self.pos_encoding(t, 128, 16)
        x3 = self.down2(x2) + self.pos_encoding(t, 256, 8)
        x3 = self.sa1(x3)
        x4 = self.down3(x3) + self.pos_encoding(t, 256, 4)
        x4 = self.sa2(x4)
        x = self.up1(x4, x3) + self.pos_encoding(t, 128, 8)
        x = self.sa3(x)
        x = self.up2(x, x2) + self.pos_encoding(t, 64, 16)
        x = self.up3(x, x1) + self.pos_encoding(t, 64, 32)
        output = self.outc(x)
        return output

    def beta(self, t):
        return self.beta_small + (t / self.t_range) * (
            self.beta_large - self.beta_small
        )

    def alpha(self, t):
        return 1 - self.beta(t)

    def alpha_bar(self, t):
        return math.prod([self.alpha(j) for j in range(t)])

    def get_loss(self, batch, batch_idx):
        # достаем класс из батча
        batch, c = batch
        # строим one_hot репрезентацию класса
        c = one_hot(c, self.num_classes).float()
        # выбираем случайно элементы которые будут учиться с классом
        is_class_cond = (
            torch.rand(size=(batch.shape[0], 1), device=batch.device) >= 0.5
        )
        c = c * is_class_cond.float()
        # строим эмбеддинг класса
        emb_c = self.linear_map_class(c)
        # подгоняем шейп под попадание в канал
        emb_c = emb_c.view(*emb_c.shape, 1, 1)
        emb_c = emb_c.expand(-1, -1, batch.shape[-2], batch.shape[-1])

        ts = torch.randint(0, self.t_range, [batch.shape[0]], device=self.device)
        noise_imgs = []
        epsilons = torch.randn(batch.shape, device=self.device)

        for i in range(len(ts)):
            a_hat = self.alpha_bar(ts[i])
            noise_imgs.append(
                (math.sqrt(a_hat) * batch[i]) + (math.sqrt(1 - a_hat) * epsilons[i])
            )
        noise_imgs = torch.stack(noise_imgs, dim=0)
        noise_imgs = torch.cat([noise_imgs, emb_c], dim=1).to(self.device)
        e_hat = self.forward(noise_imgs, ts.unsqueeze(-1).type(torch.float))
        loss = nn.functional.mse_loss(
            e_hat.reshape(-1, self.in_size), epsilons.reshape(-1, self.in_size)
        )
        return loss

    def denoise_sample(self, x, t, emb_c, w=3.0):
        """
        Corresponds to the inner loop of Algorithm 2 from (Ho et al., 2020).
        """
        self.w = w 
        with torch.no_grad():
            if t > 1:
                z = torch.randn(x.shape, device=self.device)
            else:
                z = torch.tensor(0)

            eps1 = (1 + self.w) * self.forward(
                torch.cat([x, emb_c], dim=1), t.view(1, 1).repeat(x.shape[0], 1)
            )
            eps2 = self.w * self.forward(
                torch.cat([x, emb_c * 0], dim=1), t.view(1, 1).repeat(x.shape[0], 1)
            )
            e_hat = eps1 - eps2
            pre_scale = 1 / math.sqrt(self.alpha(t))
            e_scale = (1 - self.alpha(t)) / math.sqrt(1 - self.alpha_bar(t))
            post_sigma = math.sqrt(self.beta(t)) * z
            x = pre_scale * (x - e_scale * e_hat) + post_sigma
            return x

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        self.log("val/loss", loss)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer
