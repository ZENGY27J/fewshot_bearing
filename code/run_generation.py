#!/usr/bin/env python3
"""Unified generation script: WGAN-GP and Diffusion data generation.

Produces fake samples for all shot settings and augmentation ratios.
Saves models, generated data, and training configs.

Usage:
    python run_generation.py --dataset cwru              # CWRU only
    python run_generation.py --dataset pu                 # PU only
    python run_generation.py --dataset jnu                # JNU only
    python run_generation.py --dataset cwru --generator wgan_gp
    python run_generation.py --dataset pu --generator diffusion --shot 5
    python run_generation.py --dataset cwru --skip_existing
    python run_generation.py --dataset cwru --generator diffusion --diameter --shot 5
        # Diameter conditioning: saves to diffusion_diameter/ (CWRU only)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FAKE_RATIOS = [1.0, 2.0, 5.0, 10.0, 15.0]
SHOT_SETTINGS = [1, 3, 5]

DATASET_CLASSES = {
    "cwru": ["Normal", "B007", "B014", "B021", "IR007", "IR014", "IR021", "OR007", "OR014", "OR021"],
    "pu": ["ROLL", "IR", "OR"],
    "jnu": ["Normal", "Inner", "Outer", "Ball"],
}

DATASET_SPLIT_DIRS = {
    "cwru": Path("data/splits/cwru"),
    "pu": Path("data/splits/pu"),
    "jnu": Path("data/splits/jnu"),
}

DATASET_OUTPUT_DIRS = {
    "cwru": Path("results/cwru_generated"),
    "pu": Path("results/pu_generated"),
    "jnu": Path("results/jnu_generated"),
}

LEGACY_DIRS = {
    "cwru": {
        "wgan_gp": Path("results/cwru_wgan_gp"),
        "diffusion": Path("results/cwru_diffusion"),
    },
}

WGAN_Z_DIM = 100
WGAN_LAMBDA_GP = 10
WGAN_N_CRITIC = 5
WGAN_G_LR = 1e-4
WGAN_D_LR = 1e-4
WGAN_BETA1 = 0.0
WGAN_BETA2 = 0.9
WGAN_EPOCHS = {1: 2000, 3: 1200, 5: 1000}

DIFF_T = 1000
DIFF_BETA_START = 1e-4
DIFF_BETA_END = 0.02
DIFF_LR = 1e-4
DIFF_WD = 1e-4
DIFF_BATCH = 16
DIFF_EMA = 0.999
DIFF_EPOCHS = {1: 2000, 3: 1200, 5: 1000}
DIFF_UNET_CH = [32, 64, 128]
DIFF_TIME_DIM = 256

# ---- Diameter conditioning (CWRU fault severity) ----
# Damage diameter (inches) indexed by class label.
# Normal has no fault -> mask=0; fault classes share severity across
# Ball/Inner-Race/Outer-Race locations at the same diameter.
CWRU_DIAMETER_MAP = {
    0: 0.0,    # Normal   -> mask=0 (no fault)
    1: 0.007,  # B007     -> mask=1
    2: 0.014,  # B014     -> mask=1
    3: 0.021,  # B021     -> mask=1
    4: 0.007,  # IR007    -> mask=1
    5: 0.014,  # IR014    -> mask=1
    6: 0.021,  # IR021    -> mask=1
    7: 0.007,  # OR007    -> mask=1
    8: 0.014,  # OR014    -> mask=1
    9: 0.021,  # OR021    -> mask=1
}
CWRU_DIAM_MIN = 0.007   # smallest fault diameter (for log normalization)
CWRU_DIAM_MAX = 0.021   # largest  fault diameter

# ---- Physics-consistency guidance (Option B: sampling-time FCF guidance) ----
# CWRU drive-end bearing: SKF 6205-2RS JEM.  Geometry-derived fault
# characteristic frequency (FCF) factors, expressed as multiples of shaft
# rotating frequency f_r.  NOTE: these factors depend on ball/pitch diameter
# (bearing geometry), NOT on the machined DEFECT diameter -- so B007/B014/B021
# share the same FCF *locations*; what differs with severity is the ENERGY /
# harmonic distribution at those bands.  The guidance therefore pulls each
# generated sample's spectrum toward its class's real few-shot profile, with
# extra weight on the physically-meaningful FCF harmonic bands.
CWRU_FS = 12000.0            # sampling rate (Hz)
CWRU_FR_NOMINAL = 29.95      # shaft rotating freq (Hz) @ 1797 rpm (load 0)
CWRU_FCF_FACTORS = {         # multiples of f_r
    "BPFO": 3.585,
    "BPFI": 5.415,
    "BSF": 2.357,            # ball spin freq; ball defect often shows 2xBSF
    "FTF": 0.3983,
}
PHYS_LOW_HZ = 3000.0         # FCF harmonics live in the low band (diagnosis focus)
PHYS_N_HARM = 8              # number of FCF harmonics to emphasise
PHYS_BUMP_HZ = 40.0         # Gaussian bump half-width around each FCF harmonic (Hz)
PHYS_BUMP_GAIN = 3.0         # extra weight at FCF harmonics (on top of base band)


class WGANGenerator(nn.Module):
    def __init__(self, z_dim=WGAN_Z_DIM):
        super().__init__()
        self.fc = nn.Linear(z_dim, 256 * 16)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 4, 2, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 4, 2, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 4, 2, 1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.ConvTranspose1d(32, 16, 4, 2, 1), nn.BatchNorm1d(16), nn.ReLU(),
            nn.ConvTranspose1d(16, 8, 4, 2, 1), nn.BatchNorm1d(8), nn.ReLU(),
            nn.ConvTranspose1d(8, 1, 4, 2, 1), nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z).view(z.size(0), 256, 16)
        return self.net(x)


class WGANCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, 4, 2, 1), nn.InstanceNorm1d(32), nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 4, 2, 1), nn.InstanceNorm1d(64), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 4, 2, 1), nn.InstanceNorm1d(128), nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 4, 2, 1), nn.InstanceNorm1d(256), nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, 4, 2, 1), nn.InstanceNorm1d(512), nn.LeakyReLU(0.2),
        )
        self.head = nn.Conv1d(512, 1, 16, 1, 0)

    def forward(self, x):
        return self.head(self.net(x)).view(x.size(0), -1)


def wgan_gradient_penalty(critic, real, fake):
    B = real.size(0)
    eps = torch.rand(B, 1, 1, device=real.device)
    interp = eps * real + (1 - eps) * fake
    interp.requires_grad_(True)
    d_interp = critic(interp)
    grad = torch.autograd.grad(d_interp, interp, torch.ones_like(d_interp),
                               create_graph=True, retain_graph=True)[0]
    return ((grad.view(B, -1).norm(2, dim=1) - 1) ** 2).mean()


def train_wgan_gp_per_class(real_data, n_epochs, save_dir, cls_name):
    real_t = torch.from_numpy(real_data).float().unsqueeze(1).to(DEVICE)
    loader = DataLoader(TensorDataset(real_t), batch_size=min(32, len(real_t)),
                        shuffle=True, drop_last=True)
    G = WGANGenerator().to(DEVICE)
    D = WGANCritic().to(DEVICE)
    opt_G = torch.optim.Adam(G.parameters(), lr=WGAN_G_LR, betas=(WGAN_BETA1, WGAN_BETA2))
    opt_D = torch.optim.Adam(D.parameters(), lr=WGAN_D_LR, betas=(WGAN_BETA1, WGAN_BETA2))
    history = {"g_loss": [], "d_loss": [], "gp": []}

    for epoch in range(n_epochs):
        g_ep, d_ep, gp_ep, nb = 0.0, 0.0, 0.0, 0
        for (x_real,) in loader:
            B = x_real.size(0)
            for _ in range(WGAN_N_CRITIC):
                z = torch.randn(B, WGAN_Z_DIM, device=DEVICE)
                with torch.no_grad():
                    x_fake = G(z)
                d_real = D(x_real)
                d_fake = D(x_fake.detach())
                gp = wgan_gradient_penalty(D, x_real, x_fake.detach())
                d_loss = d_fake.mean() - d_real.mean() + WGAN_LAMBDA_GP * gp
                opt_D.zero_grad(); d_loss.backward(); opt_D.step()

            z = torch.randn(B, WGAN_Z_DIM, device=DEVICE)
            g_loss = -D(G(z)).mean()
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()

            g_ep += g_loss.item(); d_ep += d_loss.item(); gp_ep += gp.item(); nb += 1

        history["g_loss"].append(g_ep / nb)
        history["d_loss"].append(d_ep / nb)
        history["gp"].append(gp_ep / nb)
        if (epoch + 1) % max(1, n_epochs // 5) == 0:
            print(f"      [{cls_name}] ep {epoch+1}/{n_epochs}  D={history['d_loss'][-1]:.4f}  G={history['g_loss'][-1]:.4f}")

    torch.save(G.state_dict(), save_dir / f"G_{cls_name}.pt")
    with open(save_dir / f"history_{cls_name}.json", "w") as f:
        json.dump(history, f, indent=2)
    return G


@torch.no_grad()
def wgan_generate(G, n_samples, cls_label):
    G.eval()
    all_fake = []
    bs = 64
    for start in range(0, n_samples, bs):
        B = min(bs, n_samples - start)
        z = torch.randn(B, WGAN_Z_DIM, device=DEVICE)
        fake = G(z).squeeze(1).cpu().numpy()
        all_fake.append(fake)
    return np.concatenate(all_fake, axis=0)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class DiffResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, stride=1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch * 2))
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1, stride=stride) if in_ch != out_ch or stride > 1 else nn.Identity()

    def forward(self, x, t_emb):
        sc = self.time_proj(t_emb).unsqueeze(-1)
        scale, shift = sc.chunk(2, dim=1)
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = self.norm2(h) * (1 + scale) + shift
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.shortcut(x)


def build_diameter_lookup(dataset, num_classes, device):
    """Build per-class log-normalized diameter and mask lookup tensors.

    Condition per sample:  emb += diameter_proj([log_d * mask, mask])
      Normal / unknown-severity -> mask=0, log_d=0  (diameter silenced)
      CWRU fault classes        -> mask=1, log_d in [0,1] (log-normalized)

    Log normalization: 0.007" -> 0.0,  0.014" -> 0.631,  0.021" -> 1.0

    Returns:
        (log_diam_lookup, mask_lookup)  tensors of shape (num_classes,) on `device`
        or (None, None) if the dataset has no diameter metadata.
    """
    if dataset != "cwru":
        return None, None
    log_min, log_max = math.log(CWRU_DIAM_MIN), math.log(CWRU_DIAM_MAX)
    log_range = log_max - log_min
    log_diam = torch.zeros(num_classes, device=device)
    mask = torch.zeros(num_classes, device=device)
    for c in range(num_classes):
        d = CWRU_DIAMETER_MAP.get(c, 0.0)
        if d > 0:
            log_diam[c] = (math.log(d) - log_min) / log_range
            mask[c] = 1.0
    return log_diam, mask


class CondUNet1D(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, num_classes=10, time_dim=DIFF_TIME_DIM,
                 use_diameter=False):
        super().__init__()
        self.use_diameter = use_diameter
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim), nn.Linear(time_dim, time_dim),
            nn.SiLU(), nn.Linear(time_dim, time_dim),
        )
        self.class_emb = nn.Embedding(num_classes, time_dim)
        if use_diameter:
            # condition input: [log_d * mask, mask]  (2-dim)
            # 2-layer MLP with nonlinearity for sufficient capacity
            self.diameter_proj = nn.Sequential(
                nn.Linear(2, time_dim), nn.SiLU(),
                nn.Linear(time_dim, time_dim))
        self.init_conv = nn.Conv1d(in_ch, base_ch, 3, padding=1)
        self.enc1 = DiffResBlock(base_ch, base_ch * 2, time_dim, stride=2)
        self.enc2 = DiffResBlock(base_ch * 2, base_ch * 4, time_dim, stride=2)
        self.mid1 = DiffResBlock(base_ch * 4, base_ch * 4, time_dim)
        self.mid2 = DiffResBlock(base_ch * 4, base_ch * 4, time_dim)
        self.dec2_up = nn.ConvTranspose1d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2_res = DiffResBlock(base_ch * 4, base_ch * 2, time_dim)
        self.dec1_up = nn.ConvTranspose1d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1_res = DiffResBlock(base_ch * 2, base_ch, time_dim)
        self.out_norm = nn.GroupNorm(min(8, base_ch), base_ch)
        self.out_conv = nn.Conv1d(base_ch, in_ch, 3, padding=1)

    def forward(self, x, t, class_labels, log_diam=None, mask=None):
        emb = self.time_mlp(t) + self.class_emb(class_labels)
        if self.use_diameter and log_diam is not None and mask is not None:
            # d_input: (B, 2) = [log_d * mask, mask]
            d_input = torch.stack([log_diam * mask, mask], dim=1)
            emb = emb + self.diameter_proj(d_input)
        e0 = self.init_conv(x)
        e1 = self.enc1(e0, emb)
        e2 = self.enc2(e1, emb)
        h = self.mid1(e2, emb)
        h = self.mid2(h, emb)
        h = self.dec2_up(h)
        h = self.dec2_res(torch.cat([h, e1], dim=1), emb)
        h = self.dec1_up(h)
        h = self.dec1_res(torch.cat([h, e0], dim=1), emb)
        return self.out_conv(F.silu(self.out_norm(h)))


class EMA:
    def __init__(self, model, decay=DIFF_EMA):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v

    def apply(self, model):
        model.load_state_dict(self.shadow)


def make_schedule(T, start, end):
    betas = torch.linspace(start, end, T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


def extract(a, t, shape):
    out = a.to(t.device).gather(0, t)
    return out.reshape([t.size(0)] + [1] * (len(shape) - 1))


def q_sample(x, t, alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x)
    return extract(torch.sqrt(alphas_cumprod), t, x.shape) * x + \
           extract(torch.sqrt(1 - alphas_cumprod), t, x.shape) * noise, noise


# ======================================================================
# Physics-consistency guidance (Option B): sampling-time FCF guidance.
# Reuses the frozen plain-diffusion generator; injects a differentiable
# spectral-consistency gradient at each reverse step (DPS-style x0 guidance).
# ======================================================================

def _norm_mag_spectrum(x, win):
    """Differentiable L2-normalized Hann magnitude spectrum.

    Mirrors fft_diag_diffusion.mag_spectrum (the domain where the class
    separability degradation 1.02->0.54 was diagnosed).

    x:   (B, L) real signal
    win: (L,) Hann window
    ->   (B, L//2+1) unit-L2-norm magnitude spectrum
    """
    xw = x * win
    spec = torch.fft.rfft(xw, dim=-1).abs()
    spec = spec / (spec.norm(dim=-1, keepdim=True) + 1e-8)
    return spec


def build_fcf_weight(n=1024, fs=CWRU_FS, device=None):
    """Physics-informed frequency weight over the rfft bins.

    Base weight 1.0 in the FCF band (0..PHYS_LOW_HZ), 0.15 above it, plus
    Gaussian bumps at the first PHYS_N_HARM harmonics of BPFO/BPFI/2xBSF/FTF.
    Uses the nominal f_r; load-induced FCF drift (<4%) is absorbed by the
    finite bump width.
    """
    freqs = torch.fft.rfftfreq(n, d=1.0 / fs)  # (L//2+1,)
    if device is not None:
        freqs = freqs.to(device)
    w = torch.where(freqs <= PHYS_LOW_HZ,
                    torch.ones_like(freqs), torch.full_like(freqs, 0.15))
    fr = CWRU_FR_NOMINAL
    bump_centers = []
    for name, fac in CWRU_FCF_FACTORS.items():
        base = fac * fr * (2.0 if name == "BSF" else 1.0)  # 2xBSF for ball defect
        for h in range(1, PHYS_N_HARM + 1):
            fc = base * h
            if fc <= PHYS_LOW_HZ:
                bump_centers.append(fc)
    for fc in bump_centers:
        w = w + PHYS_BUMP_GAIN * torch.exp(-0.5 * ((freqs - fc) / PHYS_BUMP_HZ) ** 2)
    return w  # (L//2+1,)


def build_spectral_targets(train_X, train_y, num_classes, win):
    """Per-class target spectrum = mean normalized magnitude spectrum over the
    real few-shot training samples of that class.  (num_classes, L//2+1)."""
    Xt = torch.from_numpy(train_X).float().to(win.device)   # (M, L)
    yt = torch.from_numpy(train_y).long()
    specs = _norm_mag_spectrum(Xt, win)                     # (M, F)
    F_bins = specs.shape[1]
    targets = torch.zeros(num_classes, F_bins, device=win.device)
    for c in range(num_classes):
        sel = (yt == c)
        if sel.any():
            targets[c] = specs[sel.to(win.device)].mean(dim=0)
    return targets


def phys_guidance_grad(x0_hat, class_labels, targets, weight, win):
    """Gradient of the weighted spectral-consistency loss w.r.t. x0_hat.

    L_phys = sum_f  weight_f * (spec(x0_hat)_f - target[class]_f)^2
    Returns grad with the same shape as x0_hat (B, 1, L)."""
    x = x0_hat.detach().requires_grad_(True)
    xs = x.squeeze(1)                                   # (B, L)
    spec = _norm_mag_spectrum(xs, win)                 # (B, F)
    tgt = targets[class_labels]                        # (B, F)
    loss = (weight.unsqueeze(0) * (spec - tgt) ** 2).sum(dim=1).mean()
    (grad,) = torch.autograd.grad(loss, x)
    return grad, float(loss.detach())


def p_sample_guided(model, x, t, t_idx, class_labels, betas, alphas, alphas_cumprod,
                    targets, weight, win, lambda_phys):
    """One reverse step with physics guidance in x0 space (no backprop through UNet)."""
    beta_t = extract(betas, t, x.shape)
    alpha_t = extract(alphas, t, x.shape)
    alpha_cumprod_t = extract(alphas_cumprod, t, x.shape)
    with torch.no_grad():
        eps = model(x, t, class_labels)
        # predicted clean signal
        x0_hat = (x - torch.sqrt(1 - alpha_cumprod_t) * eps) / torch.sqrt(alpha_cumprod_t)
    # physics guidance nudge (grad only through the differentiable spectrum)
    grad, loss_val = phys_guidance_grad(x0_hat, class_labels, targets, weight, win)
    g = grad / (grad.flatten(1).norm(dim=1).clamp_min(1e-8).view(-1, 1, 1))
    x0_g = (x0_hat - lambda_phys * g).clamp(-6.0, 6.0)
    # DDPM posterior mean from the guided x0
    alpha_cumprod_prev = extract(
        torch.cat([torch.ones(1, device=alphas_cumprod.device), alphas_cumprod[:-1]]), t, x.shape)
    if t_idx == 0:
        return x0_g, loss_val
    coef1 = torch.sqrt(alpha_cumprod_prev) * beta_t / (1 - alpha_cumprod_t)
    coef2 = torch.sqrt(alpha_t) * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t)
    mean = coef1 * x0_g + coef2 * x
    var = beta_t * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t)
    return mean + torch.sqrt(var) * torch.randn_like(x), loss_val


def diffusion_generate_guided(model, n_samples, class_label, betas, alphas, alphas_cumprod,
                              targets, weight, win, lambda_phys):
    model.eval()
    all_samples = []
    bs = 16
    for start in range(0, n_samples, bs):
        B = min(bs, n_samples - start)
        y = torch.full((B,), class_label, device=DEVICE, dtype=torch.long)
        x = torch.randn(B, 1, 1024, device=DEVICE)
        for i in reversed(range(DIFF_T)):
            t = torch.full((B,), i, device=DEVICE, dtype=torch.long)
            x, _ = p_sample_guided(model, x, t, i, y, betas, alphas, alphas_cumprod,
                                   targets, weight, win, lambda_phys)
        all_samples.append(x.squeeze(1).detach().cpu().numpy())
    return np.concatenate(all_samples, axis=0)


@torch.no_grad()
def p_sample(model, x, t, t_idx, class_labels, betas, alphas, alphas_cumprod,
             log_diam=None, mask=None):
    beta_t = extract(betas, t, x.shape)
    alpha_t = extract(alphas, t, x.shape)
    alpha_cumprod_t = extract(alphas_cumprod, t, x.shape)
    eps = model(x, t, class_labels, log_diam, mask)
    coef = beta_t / torch.sqrt(1 - alpha_cumprod_t)
    mean = (1 / torch.sqrt(alpha_t)) * (x - coef * eps)
    if t_idx == 0:
        return mean
    alpha_cumprod_prev = extract(
        torch.cat([torch.ones(1, device=alphas_cumprod.device), alphas_cumprod[:-1]]), t, x.shape)
    var = beta_t * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t)
    return mean + torch.sqrt(var) * torch.randn_like(x)


@torch.no_grad()
def diffusion_generate(model, n_samples, class_label, betas, alphas, alphas_cumprod,
                       log_diam_lookup=None, mask_lookup=None):
    model.eval()
    all_samples = []
    bs = 16
    for start in range(0, n_samples, bs):
        B = min(bs, n_samples - start)
        y = torch.full((B,), class_label, device=DEVICE, dtype=torch.long)
        if log_diam_lookup is not None:
            log_diam = log_diam_lookup[y]
            mask = mask_lookup[y]
        else:
            log_diam, mask = None, None
        x = torch.randn(B, 1, 1024, device=DEVICE)
        for i in reversed(range(DIFF_T)):
            t = torch.full((B,), i, device=DEVICE, dtype=torch.long)
            x = p_sample(model, x, t, i, y, betas, alphas, alphas_cumprod,
                         log_diam, mask)
        all_samples.append(x.squeeze(1).cpu().numpy())
    return np.concatenate(all_samples, axis=0)


def train_diffusion(train_X, train_y, n_epochs, save_dir, num_classes,
                    use_diameter=False, log_diam_lookup=None, mask_lookup=None):
    train_t = torch.from_numpy(train_X).float().unsqueeze(1).to(DEVICE)
    train_yt = torch.from_numpy(train_y).long().to(DEVICE)
    loader = DataLoader(TensorDataset(train_t, train_yt),
                        batch_size=min(DIFF_BATCH, len(train_t)), shuffle=True, drop_last=False)
    betas, alphas, alphas_cumprod = make_schedule(DIFF_T, DIFF_BETA_START, DIFF_BETA_END)
    model = CondUNet1D(num_classes=num_classes, use_diameter=use_diameter).to(DEVICE)
    ema = EMA(model)
    opt = torch.optim.AdamW(model.parameters(), lr=DIFF_LR, weight_decay=DIFF_WD)
    history = {"loss": []}

    for epoch in range(n_epochs):
        model.train()
        ep_loss, n = 0.0, 0
        for x, y in loader:
            B = x.size(0)
            t = torch.randint(0, DIFF_T, (B,), device=DEVICE, dtype=torch.long)
            noise = torch.randn_like(x)
            x_t, _ = q_sample(x, t, alphas_cumprod, noise)
            if use_diameter and log_diam_lookup is not None:
                log_diam = log_diam_lookup[y]
                mask = mask_lookup[y]
            else:
                log_diam, mask = None, None
            pred = model(x_t, t, y, log_diam, mask)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad(); loss.backward(); opt.step()
            ema.update(model)
            ep_loss += loss.item() * B; n += B
        history["loss"].append(ep_loss / n)
        if (epoch + 1) % max(1, n_epochs // 5) == 0:
            print(f"      [Diffusion] ep {epoch+1}/{n_epochs}  loss={history['loss'][-1]:.6f}")

    ema.apply(model)
    torch.save(model.state_dict(), save_dir / "diffusion_model.pt")
    with open(save_dir / "diffusion_history.json", "w") as f:
        json.dump(history, f, indent=2)
    return model, betas, alphas, alphas_cumprod


def run_wgan_gp(dataset, n_shot, skip_existing):
    classes = DATASET_CLASSES[dataset]
    split_dir = DATASET_SPLIT_DIRS[dataset]
    output_base = DATASET_OUTPUT_DIRS[dataset]
    legacy_base = LEGACY_DIRS.get(dataset, {}).get("wgan_gp")

    print(f"\n{'='*60}\n  WGAN-GP  {dataset.upper()}  {n_shot}-shot\n{'='*60}")
    shot_dir = output_base / "wgan_gp" / f"{n_shot}shot"
    model_dir = shot_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    train_X = np.load(split_dir / f"{n_shot}shot" / "train_X.npy")
    train_y = np.load(split_dir / f"{n_shot}shot" / "train_y.npy")

    n_real = len(train_X)
    n_per_class = {c: int((train_y == i).sum()) for i, c in enumerate(classes)}

    config = {
        "dataset": dataset, "generator": "wgan_gp", "n_shot": n_shot, "n_real": n_real,
        "n_per_class": n_per_class, "ratios": FAKE_RATIOS,
        "z_dim": WGAN_Z_DIM, "lambda_gp": WGAN_LAMBDA_GP, "n_critic": WGAN_N_CRITIC,
        "g_lr": WGAN_G_LR, "d_lr": WGAN_D_LR, "epochs": WGAN_EPOCHS[n_shot],
    }
    with open(shot_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    generators = {}
    for cls_idx, cls_name in enumerate(classes):
        cls_data = train_X[train_y == cls_idx]
        g_path = model_dir / f"G_{cls_name}.pt"
        legacy_g_path = legacy_base / f"{n_shot}shot" / "gan_models" / f"G_{cls_name}.pt" if legacy_base else None

        if skip_existing and g_path.exists():
            print(f"  Loading existing G for {cls_name}")
            G = WGANGenerator().to(DEVICE)
            G.load_state_dict(torch.load(g_path, map_location=DEVICE, weights_only=True))
        elif skip_existing and legacy_g_path and legacy_g_path.exists():
            print(f"  Loading legacy G for {cls_name}")
            G = WGANGenerator().to(DEVICE)
            G.load_state_dict(torch.load(legacy_g_path, map_location=DEVICE, weights_only=True))
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(G.state_dict(), g_path)
        else:
            print(f"  Training WGAN-GP for {cls_name} ({len(cls_data)} samples, {WGAN_EPOCHS[n_shot]} epochs)")
            G = train_wgan_gp_per_class(cls_data, WGAN_EPOCHS[n_shot], model_dir, cls_name)
        generators[cls_idx] = G

    for ratio in FAKE_RATIOS:
        gen_dir = shot_dir / "generated" / f"ratio{ratio}"
        if skip_existing and gen_dir.exists() and (gen_dir / "fake_X.npy").exists():
            print(f"  Skipping ratio {ratio} (already exists)")
            continue
        gen_dir.mkdir(parents=True, exist_ok=True)
        all_fake_X, all_fake_y = [], []
        per_class_info = {}
        for cls_idx, cls_name in enumerate(classes):
            n_fake = max(1, int(n_per_class[cls_name] * ratio))
            fake = wgan_generate(generators[cls_idx], n_fake, cls_idx)
            np.save(gen_dir / f"fake_{cls_name}.npy", fake)
            all_fake_X.append(fake)
            all_fake_y.append(np.full(n_fake, cls_idx, dtype=np.int64))
            per_class_info[cls_name] = n_fake

        fake_X = np.concatenate(all_fake_X, axis=0)
        fake_y = np.concatenate(all_fake_y, axis=0)
        np.save(gen_dir / "fake_X.npy", fake_X)
        np.save(gen_dir / "fake_y.npy", fake_y)
        print(f"  Generated ratio={ratio}: {fake_X.shape[0]} fake samples ({per_class_info})")

    print(f"  WGAN-GP {dataset.upper()} {n_shot}-shot done. Saved to {shot_dir}")


def run_diffusion(dataset, n_shot, skip_existing, use_diameter=False):
    classes = DATASET_CLASSES[dataset]
    num_classes = len(classes)
    split_dir = DATASET_SPLIT_DIRS[dataset]
    output_base = DATASET_OUTPUT_DIRS[dataset]
    legacy_base = LEGACY_DIRS.get(dataset, {}).get("diffusion")

    gen_name = "diffusion_diameter" if use_diameter else "diffusion"
    diam_tag = " + diameter condition" if use_diameter else ""
    print(f"\n{'='*60}\n  {gen_name.capitalize()}  {dataset.upper()}  {n_shot}-shot{diam_tag}\n{'='*60}")
    shot_dir = output_base / gen_name / f"{n_shot}shot"
    model_dir = shot_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Diameter condition lookups (CWRU only)
    if use_diameter:
        log_diam_lookup, mask_lookup = build_diameter_lookup(dataset, num_classes, DEVICE)
        if log_diam_lookup is None:
            print(f"  [WARN] No diameter metadata for {dataset.upper()}, falling back to plain diffusion")
            use_diameter = False
        else:
            print(f"  [Diameter] log-normalized: " +
                  ", ".join(f"{classes[c]}={log_diam_lookup[c].item():.3f}/m={int(mask_lookup[c].item())}"
                            for c in range(num_classes)))
    else:
        log_diam_lookup, mask_lookup = None, None

    train_X = np.load(split_dir / f"{n_shot}shot" / "train_X.npy")
    train_y = np.load(split_dir / f"{n_shot}shot" / "train_y.npy")

    n_real = len(train_X)
    n_per_class = {c: int((train_y == i).sum()) for i, c in enumerate(classes)}

    config = {
        "dataset": dataset, "generator": gen_name, "n_shot": n_shot, "n_real": n_real,
        "n_per_class": n_per_class, "ratios": FAKE_RATIOS,
        "T": DIFF_T, "beta_start": DIFF_BETA_START, "beta_end": DIFF_BETA_END,
        "lr": DIFF_LR, "weight_decay": DIFF_WD, "ema_decay": DIFF_EMA,
        "epochs": DIFF_EPOCHS[n_shot], "unet_channels": DIFF_UNET_CH,
        "num_classes": num_classes,
        "use_diameter": use_diameter,
    }
    if use_diameter:
        config["diameter_map"] = CWRU_DIAMETER_MAP
        config["diam_log_norm"] = {"min": CWRU_DIAM_MIN, "max": CWRU_DIAM_MAX}
    if not skip_existing:
        with open(shot_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    model_path = model_dir / "diffusion_model.pt"
    legacy_model_path = legacy_base / f"{n_shot}shot" / "diffusion_model" / "diffusion_model.pt" if legacy_base else None

    if skip_existing and model_path.exists():
        print(f"  Loading existing {gen_name} model")
        model = CondUNet1D(num_classes=num_classes, use_diameter=use_diameter).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        betas, alphas, alphas_cumprod = make_schedule(DIFF_T, DIFF_BETA_START, DIFF_BETA_END)
    elif skip_existing and legacy_model_path and legacy_model_path.exists() and not use_diameter:
        print(f"  Loading legacy Diffusion model")
        model = CondUNet1D(num_classes=num_classes).to(DEVICE)
        model.load_state_dict(torch.load(legacy_model_path, map_location=DEVICE, weights_only=True))
        betas, alphas, alphas_cumprod = make_schedule(DIFF_T, DIFF_BETA_START, DIFF_BETA_END)
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
    else:
        print(f"  Training {gen_name} ({n_real} samples, {DIFF_EPOCHS[n_shot]} epochs)")
        model, betas, alphas, alphas_cumprod = train_diffusion(
            train_X, train_y, DIFF_EPOCHS[n_shot], model_dir, num_classes,
            use_diameter=use_diameter, log_diam_lookup=log_diam_lookup,
            mask_lookup=mask_lookup)

    for ratio in FAKE_RATIOS:
        gen_dir = shot_dir / "generated" / f"ratio{ratio}"
        if skip_existing and gen_dir.exists() and (gen_dir / "fake_X.npy").exists():
            print(f"  Skipping ratio {ratio} (already exists)")
            continue
        gen_dir.mkdir(parents=True, exist_ok=True)
        all_fake_X, all_fake_y = [], []
        per_class_info = {}
        for cls_idx, cls_name in enumerate(classes):
            n_fake = max(1, int(n_per_class[cls_name] * ratio))
            fake = diffusion_generate(model, n_fake, cls_idx, betas, alphas, alphas_cumprod,
                                      log_diam_lookup=log_diam_lookup,
                                      mask_lookup=mask_lookup)
            np.save(gen_dir / f"fake_{cls_name}.npy", fake)
            all_fake_X.append(fake)
            all_fake_y.append(np.full(n_fake, cls_idx, dtype=np.int64))
            per_class_info[cls_name] = n_fake

        fake_X = np.concatenate(all_fake_X, axis=0)
        fake_y = np.concatenate(all_fake_y, axis=0)
        np.save(gen_dir / "fake_X.npy", fake_X)
        np.save(gen_dir / "fake_y.npy", fake_y)
        print(f"  Generated ratio={ratio}: {fake_X.shape[0]} fake samples ({per_class_info})")

    print(f"  Diffusion {dataset.upper()} {n_shot}-shot done. Saved to {shot_dir}")


def run_diffusion_physguide(dataset, n_shot, lambda_phys, only_ratio=None):
    """Option B: reuse the frozen plain-diffusion generator and add sampling-time
    physics (FCF spectral-consistency) guidance.  Saves to diffusion_physguide/.
    Never retrains -- the 89.22% baseline generator is used as-is."""
    if dataset != "cwru":
        raise SystemExit("[physguide] only supported for CWRU (FCF geometry known)")
    classes = DATASET_CLASSES[dataset]
    num_classes = len(classes)
    split_dir = DATASET_SPLIT_DIRS[dataset]
    output_base = DATASET_OUTPUT_DIRS[dataset]

    gen_name = "diffusion_physguide"
    print(f"\n{'='*60}\n  PhysGuide  {dataset.upper()}  {n_shot}-shot  "
          f"lambda_phys={lambda_phys}\n{'='*60}")
    shot_dir = output_base / gen_name / f"{n_shot}shot"
    shot_dir.mkdir(parents=True, exist_ok=True)

    # ---- load frozen plain-diffusion baseline (no retraining) ----
    base_model_path = output_base / "diffusion" / f"{n_shot}shot" / "models" / "diffusion_model.pt"
    if not base_model_path.exists():
        raise SystemExit(f"[physguide] baseline model missing: {base_model_path}")
    print(f"  Loading frozen baseline generator: {base_model_path}")
    model = CondUNet1D(num_classes=num_classes, use_diameter=False).to(DEVICE)
    model.load_state_dict(torch.load(base_model_path, map_location=DEVICE, weights_only=True))
    betas, alphas, alphas_cumprod = make_schedule(DIFF_T, DIFF_BETA_START, DIFF_BETA_END)

    # ---- build physics targets & FCF weight from real few-shot data ----
    train_X = np.load(split_dir / f"{n_shot}shot" / "train_X.npy")
    train_y = np.load(split_dir / f"{n_shot}shot" / "train_y.npy")
    n_per_class = {c: int((train_y == i).sum()) for i, c in enumerate(classes)}
    L = train_X.shape[1]
    win = torch.hann_window(L, periodic=False, device=DEVICE)
    targets = build_spectral_targets(train_X, train_y, num_classes, win)
    weight = build_fcf_weight(n=L, fs=CWRU_FS, device=DEVICE)
    print(f"  FCF weight: bins={weight.numel()}  max={weight.max():.2f}  "
          f"band<= {PHYS_LOW_HZ:.0f}Hz emphasised, {PHYS_N_HARM} harmonics bumped")

    config = {
        "dataset": dataset, "generator": gen_name, "n_shot": n_shot,
        "n_per_class": n_per_class, "lambda_phys": lambda_phys,
        "fs": CWRU_FS, "fr_nominal": CWRU_FR_NOMINAL, "fcf_factors": CWRU_FCF_FACTORS,
        "phys_low_hz": PHYS_LOW_HZ, "n_harm": PHYS_N_HARM,
        "bump_hz": PHYS_BUMP_HZ, "bump_gain": PHYS_BUMP_GAIN,
        "base_model": str(base_model_path),
    }
    with open(shot_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    ratios = [only_ratio] if only_ratio is not None else FAKE_RATIOS
    for ratio in ratios:
        gen_dir = shot_dir / "generated" / f"ratio{ratio}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        all_fake_X, all_fake_y = [], []
        per_class_info = {}
        for cls_idx, cls_name in enumerate(classes):
            n_fake = max(1, int(n_per_class[cls_name] * ratio))
            fake = diffusion_generate_guided(
                model, n_fake, cls_idx, betas, alphas, alphas_cumprod,
                targets, weight, win, lambda_phys)
            np.save(gen_dir / f"fake_{cls_name}.npy", fake)
            all_fake_X.append(fake)
            all_fake_y.append(np.full(n_fake, cls_idx, dtype=np.int64))
            per_class_info[cls_name] = n_fake
            print(f"    [{cls_name}] generated {n_fake} guided samples")
        fake_X = np.concatenate(all_fake_X, axis=0)
        fake_y = np.concatenate(all_fake_y, axis=0)
        np.save(gen_dir / "fake_X.npy", fake_X)
        np.save(gen_dir / "fake_y.npy", fake_y)
        print(f"  Generated ratio={ratio}: {fake_X.shape[0]} guided fake samples")

    print(f"  PhysGuide {dataset.upper()} {n_shot}-shot done. Saved to {shot_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cwru", "pu", "jnu"], required=True)
    parser.add_argument("--generator", choices=["wgan_gp", "diffusion", "all"], default="all")
    parser.add_argument("--shot", type=int, nargs="*", default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--diameter", action="store_true",
                        help="Enable damage-diameter conditioning (CWRU only). "
                             "Saves to diffusion_diameter/ to keep plain diffusion baseline intact.")
    parser.add_argument("--phys_guide", action="store_true",
                        help="Option B: reuse frozen baseline generator + sampling-time "
                             "FCF spectral-consistency guidance (CWRU only). "
                             "Saves to diffusion_physguide/, never retrains.")
    parser.add_argument("--lambda_phys", type=float, default=0.05,
                        help="Physics guidance strength (default 0.05).")
    parser.add_argument("--only_ratio", type=float, default=None,
                        help="Restrict generation to a single ratio (e.g. 15.0).")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Dataset: {args.dataset.upper()}  Classes: {DATASET_CLASSES[args.dataset]}")

    # Deterministic seed for reproducible baseline
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[seed] All seeds set to {SEED}")

    shots = args.shot or SHOT_SETTINGS

    if args.phys_guide:
        for s in shots:
            run_diffusion_physguide(args.dataset, s, args.lambda_phys,
                                    only_ratio=args.only_ratio)
        print("\nAll generation tasks complete.")
        return

    if args.generator in ("wgan_gp", "all"):
        for s in shots:
            run_wgan_gp(args.dataset, s, args.skip_existing)

    if args.generator in ("diffusion", "all"):
        for s in shots:
            run_diffusion(args.dataset, s, args.skip_existing, use_diameter=args.diameter)

    print("\nAll generation tasks complete.")


if __name__ == "__main__":
    main()
