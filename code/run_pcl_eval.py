#!/usr/bin/env python3
"""P0优化方案一：原型对比损失（PCL）评估脚本。

在 run_augmented_eval.py 基础上新增 PrototypeContrastiveLoss，
训练目标改为 L = L_CE + lambda * L_PCL，其余流程保持一致。

Classifiers: CWT+ResidualAttention (main), CNN+Transformer (comparison)
Generators:  wgan_gp, diffusion
Shots:       1, 3, 5
Ratios:      0.0, 1.0, 2.0, 5.0, 10.0, 15.0
Metrics:     Accuracy, F1-macro, AUC-ROC, per-class F1/Accuracy
Repeats:     10 (seeds 0-9)

Usage:
    python run_pcl_eval.py --dataset cwru                         # CWRU all
    python run_pcl_eval.py --dataset cwru --lambda_pcl 0.5        # 指定PCL权重
    python run_pcl_eval.py --dataset cwru --classifier cwt_ran
    python run_pcl_eval.py --dataset cwru --ratio 0.0 --shot 1    # 仅基线
    python run_pcl_eval.py --dataset cwru --skip_existing
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import cycle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 数据集配置 ====================
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

DATASET_GEN_DIRS = {
    "cwru": Path("results/cwru_generated"),
    "pu": Path("results/pu_generated"),
    "jnu": Path("results/jnu_generated"),
}

# PCL 评估结果输出到独立目录，避免覆盖原评估结果
DATASET_PCL_EVAL_DIRS = {
    "cwru": Path("results/cwru_pcl_eval"),
    "pu": Path("results/pu_pcl_eval"),
    "jnu": Path("results/jnu_pcl_eval"),
}

DATASET_CACHE_DIRS = {
    "cwru": Path("results/cwru_eval_cache"),
    "pu": Path("results/pu_eval_cache"),
    "jnu": Path("results/jnu_eval_cache"),
}

GEN_DIR_LEGACY = {
    "cwru": {
        "wgan_gp": Path("results/cwru_wgan_gp"),
        "diffusion": Path("results/cwru_diffusion"),
    },
}

# ==================== 实验参数 ====================
FAKE_RATIOS = [0.0, 1.0, 2.0, 5.0, 10.0, 15.0]
SHOT_SETTINGS = [1, 3, 5]
REPEATS = 10
SEEDS = list(range(REPEATS))

BATCH_SIZE = 32
CLS_LR = 5e-4
CLS_WD = 0.01
CLS_EPOCHS = 100
CLS_T_MAX = 80
CLS_PATIENCE = 20
LABEL_SMOOTHING = 0.0

CWT_SCALES = 128
CWT_W0 = 6
CWT_IMG_SIZE = 128

# PCL 超参数
PCL_LAMBDA = 0.5
PCL_TAU = 0.07


# ==================== CWT 预处理 ====================
@torch.no_grad()
def cwt_batch(signals, n_scales=CWT_SCALES, w0=CWT_W0):
    B, N = signals.shape
    device = signals.device
    scales = torch.logspace(np.log10(2), np.log10(N / 4), n_scales, device=device)
    fft = torch.fft.rfft(signals, dim=1)
    freqs = torch.fft.rfftfreq(N, device=device)
    coeffs = []
    sqrt2pi = math.sqrt(2 * math.pi)
    pi14 = math.pow(math.pi, -0.25)
    for scale in scales:
        omega = 2.0 * math.pi * freqs * scale
        wavelet_fft = pi14 * torch.sqrt(scale) * torch.exp(-0.5 * (omega - w0) ** 2)
        wavelet_fft = wavelet_fft * (freqs > 0).float()
        coeff = torch.fft.irfft(fft * wavelet_fft.unsqueeze(0), n=N, dim=1)
        coeffs.append(coeff.abs())
    coeffs = torch.stack(coeffs, dim=1).unsqueeze(1)
    coeffs = F.interpolate(coeffs, size=(CWT_IMG_SIZE, CWT_IMG_SIZE), mode="bilinear", align_corners=False)
    coeffs = torch.log(coeffs + 1e-6)
    mean = coeffs.mean(dim=(2, 3), keepdim=True)
    std = coeffs.std(dim=(2, 3), keepdim=True) + 1e-6
    coeffs = (coeffs - mean) / std
    return coeffs


def precompute_cwt(X, cache_path):
    if cache_path.exists():
        return np.load(cache_path)["cwt"]
    x_t = torch.from_numpy(X).float().to(DEVICE)
    all_cwt = []
    for start in range(0, len(x_t), 64):
        cwt = cwt_batch(x_t[start:start + 64]).cpu().numpy()
        all_cwt.append(cwt)
    result = np.concatenate(all_cwt, axis=0).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, cwt=result)
    return result


def precompute_stft(X, cache_path):
    if cache_path.exists():
        return np.load(cache_path)["stft"]
    x_t = torch.from_numpy(X).float().to(DEVICE)
    all_stft = []
    for start in range(0, len(x_t), 64):
        s = compute_stft(x_t[start:start + 64]).cpu().numpy()
        all_stft.append(s)
    result = np.concatenate(all_stft, axis=0).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, stft=result)
    return result


def compute_stft(windows):
    spec = torch.stft(windows, n_fft=1024, hop_length=256,
                      window=torch.hann_window(1024, device=windows.device),
                      center=True, return_complex=True)
    return spec.abs().float().unsqueeze(1)


# ==================== 分类器模型 ====================
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class AttentionModule(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.stride = stride
        self.out_ch = out_ch
        mid_ch = max(out_ch // 4, 16)

        self.trunk = nn.Sequential(
            ResidualBlock(in_ch, out_ch, stride),
            ResidualBlock(out_ch, out_ch),
        )

        self.mask_pool1 = nn.MaxPool2d(2, 2)
        self.mask_enc1 = ResidualBlock(in_ch, mid_ch)
        self.mask_skip_enc1 = nn.Sequential(nn.Conv2d(in_ch, mid_ch, 1), nn.BatchNorm2d(mid_ch))
        self.mask_pool2 = nn.MaxPool2d(2, 2)
        self.mask_enc2 = ResidualBlock(mid_ch, mid_ch)
        self.mask_mid = nn.Sequential(
            ResidualBlock(mid_ch, mid_ch), ResidualBlock(mid_ch, mid_ch))

        self.mask_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(mid_ch * 2, mid_ch, 1), nn.BatchNorm2d(mid_ch))
        self.mask_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(mid_ch * 2, out_ch, 1), nn.BatchNorm2d(out_ch))

        self.mask_out = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1), nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, 1), nn.BatchNorm2d(out_ch),
            nn.Sigmoid(),
        )
        self.last_conv = nn.Sequential(nn.Conv2d(out_ch, out_ch, 1), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        trunk_out = self.trunk(x)
        skip_h = self.mask_skip_enc1(x)
        e1 = self.mask_enc1(x)
        e1_p = self.mask_pool1(e1)
        skip_h2 = self.mask_enc2(e1_p)
        e2_p = self.mask_pool2(skip_h2)
        mid = self.mask_mid(e2_p)
        d1 = self.mask_up1(torch.cat([mid, e2_p], dim=1))
        d2 = self.mask_up2(torch.cat([d1, skip_h2], dim=1))
        if d2.shape[2:] != trunk_out.shape[2:]:
            d2 = F.interpolate(d2, size=trunk_out.shape[2:], mode="bilinear", align_corners=False)
        mask = self.mask_out(d2)
        out = trunk_out * (1.0 + mask)
        return self.last_conv(out)


class CWTResidualAttentionNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stage1 = AttentionModule(64, 128, stride=1)
        self.stage2 = AttentionModule(128, 256, stride=2)
        self.stage3 = AttentionModule(256, 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)

    def extract_features(self, x):
        x = self.init_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.avgpool(x).flatten(1)


class CNNTransformer(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 128))
        enc = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256,
                                         dropout=0.1, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=3)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        return self.classifier(x[:, 0, :])

    def extract_features(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        return self.transformer(x)[:, 0, :]


# ==================== PCL 模块（新增） ====================
class PrototypeContrastiveLoss(nn.Module):
    """Batch-level prototype contrastive loss.

    For each sample, computes cosine similarity to:
    - Its own class prototype (positive)
    - All other class prototypes (negative)

    Uses temperature-scaled cross-entropy over similarities.
    Only classes present in the current batch contribute.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, features, labels):
        # features: (B, D)  — L2归一化后的嵌入
        # labels:   (B,)   — 类别标签
        features = F.normalize(features, dim=1, p=2)
        unique_labels = labels.unique()

        # 计算每个类的原型（均值嵌入）
        prototypes = []
        for lbl in unique_labels:
            mask = (labels == lbl)
            prototypes.append(features[mask].mean(dim=0))
        prototypes = torch.stack(prototypes, dim=0)  # (C_batch, D)
        prototypes = F.normalize(prototypes, dim=1, p=2)

        # 余弦相似度矩阵: (B, C_batch)
        sim = features @ prototypes.T / self.tau

        # 目标：每个样本匹配其类别对应原型的索引
        label_to_idx = {lbl.item(): i for i, lbl in enumerate(unique_labels)}
        target = torch.tensor([label_to_idx[l.item()] for l in labels],
                            device=features.device, dtype=torch.long)

        return F.cross_entropy(sim, target)


class EMAPrototypeBank(nn.Module):
    """Real-anchor EMA prototype bank: updated only with real sample features.

    Maintains persistent class prototypes across batches via exponential moving
    average. Only real (non-augmented) samples contribute to prototype updates,
    keeping anchors stable against generated noise.

    Used as an alternative to batch-level PrototypeContrastiveLoss.
    """

    def __init__(self, num_classes, feat_dim, momentum=0.99, tau=0.07):
        super().__init__()
        self.momentum = momentum
        self.tau = tau
        self.prototypes = nn.Parameter(
            torch.zeros(num_classes, feat_dim), requires_grad=False)
        self.initialized = False

    @torch.no_grad()
    def update(self, features, labels, is_real):
        """EMA update with real samples only."""
        if not is_real.any():
            return
        real_feat = features[is_real]
        real_labels = labels[is_real]
        for c in real_labels.unique():
            mask = real_labels == c
            class_mean = F.normalize(real_feat[mask].mean(dim=0), dim=0)
            if not self.initialized:
                self.prototypes[c] = class_mean
            else:
                self.prototypes[c] = (
                    self.momentum * self.prototypes[c] +
                    (1 - self.momentum) * class_mean
                )
        self.initialized = True

    def forward(self, features, labels):
        """PCL loss: cosine similarity to EMA prototypes, temperature-scaled CE."""
        features = F.normalize(features, dim=1)
        proto = F.normalize(self.prototypes, dim=1)
        sim = features @ proto.T / self.tau
        return F.cross_entropy(sim, labels)


# ==================== 训练与评估 ====================
def train_one_epoch_pcl(model, loader, optimizer, ce_criterion, pcl_criterion,
                        lambda_pcl, ema_bank=None):
    """单epoch训练：L = L_CE + lambda_pcl * L_PCL

    When ema_bank is provided, uses EMA prototypes (updated with real samples only)
    instead of batch-level prototypes.
    """
    model.train()
    total_ce, total_pcl, total_loss, n = 0.0, 0.0, 0.0, 0
    for x_img, y, is_real in loader:
        x_img, y = x_img.to(DEVICE), y.to(DEVICE)
        is_real = is_real.to(DEVICE)

        logits = model(x_img)
        features = model.extract_features(x_img)

        ce_loss = ce_criterion(logits, y)
        if ema_bank is not None:
            ema_bank.update(features.detach(), y, is_real.bool())
            pcl_loss = ema_bank(features, y)
        else:
            pcl_loss = pcl_criterion(features, y)
        loss = ce_loss + lambda_pcl * pcl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        B = x_img.size(0)
        total_ce += ce_loss.item() * B
        total_pcl += pcl_loss.item() * B
        total_loss += loss.item() * B
        n += B

    return total_loss / n, total_ce / n, total_pcl / n


@torch.no_grad()
def evaluate_full(model, loader, criterion, classes, return_features=False):
    model.eval()
    total, n = 0.0, 0
    all_preds, all_labels, all_probs, all_logits = [], [], [], []
    all_features = [] if return_features else None
    num_classes = len(classes)

    for x_img, y in loader:
        x_img, y = x_img.to(DEVICE), y.to(DEVICE)

        if return_features:
            all_features.append(model.extract_features(x_img).cpu().numpy())

        logits = model(x_img)
        loss = criterion(logits, y)
        probs = F.softmax(logits, dim=1)

        total += loss.item() * x_img.size(0); n += x_img.size(0)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(y.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_logits.append(logits.detach().cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)
    logits_all = np.concatenate(all_logits)

    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
    per_class_acc = {}
    cm = confusion_matrix(labels, preds)
    for i, cls in enumerate(classes):
        if i < cm.shape[0] and cm[i].sum() > 0:
            per_class_acc[cls] = float(cm[i, i] / cm[i].sum())
        else:
            per_class_acc[cls] = 0.0

    labels_bin = label_binarize(labels, classes=list(range(num_classes)))
    try:
        auc_roc = float(roc_auc_score(labels_bin, probs, average="macro", multi_class="ovr"))
    except ValueError:
        auc_roc = 0.0

    result = {
        "loss": total / n,
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "auc_roc_macro": auc_roc,
        "per_class_f1": {cls: float(v) for cls, v in zip(classes, per_class_f1)},
        "per_class_accuracy": per_class_acc,
        "preds": preds,
        "labels": labels,
        "probs": probs,
        "logits": logits_all,
    }
    if return_features:
        result["features"] = np.concatenate(all_features, axis=0)
    return result


def train_classifier_pcl(model, train_X_img, train_y, val_loader, n_shot,
                         lambda_pcl, pcl_tau, proto_mode="batch", n_real=None):
    """PCL 训练流程：L = L_CE + lambda_pcl * L_PCL

    proto_mode:
      "batch" — batch-level prototypes (PrototypeContrastiveLoss)
      "ema"   — EMA real-anchor prototypes (EMAPrototypeBank)
    """
    n_train = len(train_X_img)
    eff_batch = min(BATCH_SIZE, n_train)

    opt = torch.optim.AdamW(model.parameters(), lr=CLS_LR, weight_decay=CLS_WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CLS_T_MAX)
    ce_criterion = nn.CrossEntropyLoss()
    pcl_criterion = PrototypeContrastiveLoss(temperature=pcl_tau)

    # Real/fake mask for EMA prototype updates
    if n_real is None:
        n_real = n_train
    is_real = torch.zeros(n_train)
    is_real[:n_real] = 1.0

    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_X_img), torch.from_numpy(train_y).long(), is_real),
        batch_size=eff_batch, shuffle=True, drop_last=False)

    # EMA prototype bank
    ema_bank = None
    if proto_mode == "ema":
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 128, 128).to(DEVICE)
            feat_dim = model.extract_features(dummy).shape[1]
        num_classes = model.classifier.out_features
        ema_bank = EMAPrototypeBank(num_classes, feat_dim, tau=pcl_tau).to(DEVICE)
        # Initialize prototypes with all real samples
        model.eval()
        with torch.no_grad():
            real_X = torch.from_numpy(train_X_img[:n_real]).to(DEVICE)
            real_y = torch.from_numpy(train_y[:n_real]).long().to(DEVICE)
            for i in range(0, n_real, eff_batch):
                batch_X = real_X[i:i+eff_batch]
                batch_y = real_y[i:i+eff_batch]
                feats = model.extract_features(batch_X)
                ema_bank.update(feats, batch_y, torch.ones(len(batch_y)).bool().to(DEVICE))
        model.train()

    best_loss, best_state, pat = float("inf"), None, 0
    history = {"train_loss": [], "train_ce": [], "train_pcl": [], "val_loss": [], "val_acc": []}

    for epoch in range(CLS_EPOCHS):
        tloss, tce, tpcl = train_one_epoch_pcl(
            model, loader, opt, ce_criterion, pcl_criterion, lambda_pcl, ema_bank=ema_bank)
        vm = evaluate_full(model, val_loader, ce_criterion, [])
        sched.step()
        history["train_loss"].append(tloss)
        history["train_ce"].append(tce)
        history["train_pcl"].append(tpcl)
        history["val_loss"].append(vm["loss"])
        history["val_acc"].append(vm["accuracy"])
        if vm["loss"] < best_loss:
            best_loss = vm["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat >= CLS_PATIENCE:
            break

    model.load_state_dict(best_state)
    return model, history


# ==================== 可视化 ====================
def plot_confusion_matrix(labels, preds, classes, title, save_path):
    num_classes = len(classes)
    # 手动构建混淆矩阵: 兼容单类测试集(留一轴承时测试集仅含一类),
    # 避免 sklearn confusion_matrix(labels=classes) 在 y_true 缺类时抛
    # "At least one label specified must be in y_true".
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    labels = np.asarray(labels).ravel().astype(int)
    preds = np.asarray(preds).ravel().astype(int)
    for t, p in zip(labels, preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    cm_n = cm.astype(float) / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(max(6, num_classes * 0.9), max(5, num_classes * 0.8)))
    im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cm_n[i, j] > 0.5 else "black")
    ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tsne(features, labels, classes, title, save_path):
    num_classes = len(classes)
    n_pts = min(2000, features.shape[0])
    idx = np.random.RandomState(42).choice(features.shape[0], n_pts, replace=False)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_pts // 4)).fit_transform(features[idx])
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.tab10
    for ci in range(num_classes):
        m = labels[idx] == ci
        if m.any():
            ax.scatter(tsne[m, 0], tsne[m, 1], c=[cmap(ci)], s=8, alpha=0.5, label=classes[ci])
    ax.legend(fontsize=7, markerscale=2, loc="lower left")
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(history, title, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train (CE+PCL)", alpha=0.8)
    ax1.plot(history["train_ce"], label="Train CE", alpha=0.6, ls=":")
    ax1.plot(history["train_pcl"], label="Train PCL", alpha=0.6, ls="--")
    ax1.plot(history["val_loss"], label="Val", alpha=0.8, ls="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title(f"Loss - {title}")
    ax1.legend(fontsize=8)
    ax2.plot(history["val_acc"], alpha=0.8)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val Accuracy"); ax2.set_title(f"Val Acc - {title}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==================== 单次实验配置 ====================
# ---------------------------------------------------------------------------
# 两级分层指标 (§5.1) + grouped-softmax 两级重打分
#   位置分组 (按故障部位): Normal / Ball(B) / InnerRace(IR) / OuterRace(OR)
#   组内按损伤直径 (007/014/021) 细分.
# grouped-softmax 规则: 先对"位置分组 logits 求和"取 argmax 定位置,
#   再在预测位置组内取 logits argmax 定直径 -> 两级(层次)解码.
# ---------------------------------------------------------------------------
LOC_GROUPS = {
    # cwru 类序: 0 Normal,1 B007,2 B014,3 B021,4 IR007,5 IR014,6 IR021,7 OR007,8 OR014,9 OR021
    "cwru":  [[0], [1, 2, 3], [4, 5, 6], [7, 8, 9]],
}
LOCATION_NAMES = {"cwru": ["Normal", "Ball", "IR", "OR"]}


def grouped_softmax_twolevel(logits, labels, num_classes, groups=None):
    """对一组 logits 做 grouped-softmax 两级解码, 返回重打分指标与混淆矩阵.

    logits : (N, num_classes)   labels : (N,) 真类索引
    groups : list[list[int]]   每个元素是一个"位置组"的类索引列表
    """
    if groups is None:
        groups = [[i] for i in range(num_classes)]  # 退化为平铺 10 类
    n_loc = len(groups)

    # ---- 第一级: 位置 (按组 logits 求和取 argmax) ----
    loc_logits = np.stack([logits[:, g].sum(axis=1) for g in groups], axis=1)  # (N, n_loc)
    pred_loc = loc_logits.argmax(axis=1)

    # ---- 第二级: 直径 (预测位置组内取 argits argmax) ----
    pred_class = np.zeros(len(labels), dtype=np.int64)
    for li, g in enumerate(groups):
        mask = pred_loc == li
        if mask.any():
            sub = logits[mask][:, g]
            sev = sub.argmax(axis=1)
            pred_class[mask] = np.array(g, dtype=np.int64)[sev]

    # ---- 真值位置 / 组内直径下标 ----
    true_loc = np.zeros(len(labels), dtype=np.int64)
    true_sev_in_group = np.zeros(len(labels), dtype=np.int64)
    for li, g in enumerate(groups):
        for idx, c in enumerate(g):
            m = labels == c
            true_loc[m] = li
            true_sev_in_group[m] = idx

    flat_pred = logits.argmax(axis=1)
    location_acc = float((pred_loc == true_loc).mean())
    twolevel_acc = float((pred_class == labels).mean())
    flat_acc = float((flat_pred == labels).mean())

    # 组内直径是否正确 (仅在位置判对时有意义)
    loc_correct = pred_loc == true_loc
    sev_correct = np.zeros(len(labels), dtype=bool)
    for li, g in enumerate(groups):
        mask = pred_loc == li
        if mask.any():
            sub = logits[mask][:, g]
            sev = sub.argmax(axis=1)
            sev_correct[mask] = (sev == true_sev_in_group[mask])
    cond_sev_all = float(sev_correct[loc_correct].mean()) if loc_correct.any() else 0.0
    # 故障类条件严重度 (排除 Normal 组, 其组内仅 1 类, 直径恒"正确")
    fault_mask = true_loc != 0
    cond_sev_fault = float(sev_correct[loc_correct & fault_mask].mean()) \
        if (loc_correct & fault_mask).any() else 0.0

    loc_cm = confusion_matrix(true_loc, pred_loc, labels=list(range(n_loc)))
    twolevel_cm = confusion_matrix(labels, pred_class, labels=list(range(num_classes)))
    metrics = {
        "flat_acc": flat_acc,
        "location_acc": location_acc,
        "twolevel_acc": twolevel_acc,
        "cond_sev_acc_all": cond_sev_all,
        "cond_sev_acc_fault": cond_sev_fault,
    }
    return {"metrics": metrics, "loc_cm": loc_cm, "twolevel_cm": twolevel_cm}


def run_single_config(dataset, classifier_name, generator_name, n_shot, ratio,
                      lambda_pcl, pcl_tau, skip_existing, proto_mode="batch",
                      use_filtered=False):
    classes = DATASET_CLASSES[dataset]
    num_classes = len(classes)
    split_dir = DATASET_SPLIT_DIRS[dataset]
    gen_dir = DATASET_GEN_DIRS[dataset]
    output_base = DATASET_PCL_EVAL_DIRS[dataset]
    cache_dir = DATASET_CACHE_DIRS[dataset]
    gen_legacy = GEN_DIR_LEGACY.get(dataset, {})

    exp_name = f"{classifier_name}_{generator_name}"
    # PCL 结果保存到独立子目录，标注 lambda 值和原型模式
    lambda_tag = f"lambda{lambda_pcl}" if proto_mode == "batch" else f"lambda{lambda_pcl}_{proto_mode}"
    if use_filtered:
        lambda_tag += "_filtered"
    out_dir = output_base / exp_name / f"{n_shot}shot" / f"ratio{ratio}" / lambda_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    result_path = out_dir / "metrics.json"
    if skip_existing and result_path.exists():
        print(f"  Skipping {exp_name} {n_shot}-shot ratio={ratio} lambda={lambda_pcl} (exists)")
        return

    train_X_real = np.load(split_dir / f"{n_shot}shot" / "train_X.npy")
    train_y_real = np.load(split_dir / f"{n_shot}shot" / "train_y.npy").astype(np.int64)

    if ratio > 0:
        gen_subdir = "generated_filtered" if use_filtered else "generated"
        fake_dir = gen_dir / generator_name / f"{n_shot}shot" / gen_subdir / f"ratio{ratio}"
        fake_X_path = fake_dir / "fake_X.npy"
        fake_y_path = fake_dir / "fake_y.npy"

        if not fake_X_path.exists():
            legacy_base = gen_legacy.get(generator_name)
            if legacy_base is not None:
                legacy_dir = legacy_base / f"{n_shot}shot" / "generated" / f"fakes_ratio{ratio}"
                legacy_X = legacy_dir / "fake_X.npy"
                legacy_y = legacy_dir / "fake_y.npy"
                if legacy_X.exists():
                    fake_dir = legacy_dir
                    fake_X_path = legacy_X
                    fake_y_path = legacy_y

        if not fake_X_path.exists():
            print(f"  WARNING: {fake_X_path} not found, skipping")
            return
        fake_X = np.load(fake_X_path)
        fake_y = np.load(fake_y_path).astype(np.int64)
        train_X = np.concatenate([train_X_real, fake_X], axis=0)
        train_y = np.concatenate([train_y_real, fake_y], axis=0)
    else:
        train_X = train_X_real
        train_y = train_y_real

    val_X = np.load(split_dir / f"{n_shot}shot" / "val_X.npy")
    val_y = np.load(split_dir / f"{n_shot}shot" / "val_y.npy").astype(np.int64)
    test_X = np.load(split_dir / f"{n_shot}shot" / "test_X.npy")
    test_y = np.load(split_dir / f"{n_shot}shot" / "test_y.npy").astype(np.int64)

    preprocessor = "cwt" if classifier_name == "cwt_ran" else "stft"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Precomputing {preprocessor} for train/val/test...")
    filtered_tag = "_filtered" if use_filtered else ""
    if preprocessor == "cwt":
        train_X_img = precompute_cwt(train_X, cache_dir / f"{dataset}_cwt_train_{n_shot}shot_ratio{ratio}_{generator_name}{filtered_tag}.npz")
        val_X_img = precompute_cwt(val_X, cache_dir / f"{dataset}_cwt_val_{n_shot}shot.npz")
        test_X_img = precompute_cwt(test_X, cache_dir / f"{dataset}_cwt_test_{n_shot}shot.npz")
    else:
        train_X_img = precompute_stft(train_X, cache_dir / f"{dataset}_stft_train_{n_shot}shot_ratio{ratio}_{generator_name}.npz")
        val_X_img = precompute_stft(val_X, cache_dir / f"{dataset}_stft_val_{n_shot}shot.npz")
        test_X_img = precompute_stft(test_X, cache_dir / f"{dataset}_stft_test_{n_shot}shot.npz")

    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_X_img), torch.from_numpy(val_y)),
                            batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(test_X_img), torch.from_numpy(test_y)),
                              batch_size=BATCH_SIZE, shuffle=False)

    repeat_metrics = []
    acc_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    all_logits_list, all_labels_list = [], []
    best_acc, best_result, best_history = 0.0, None, None

    for rep in range(REPEATS):
        seed = SEEDS[rep]
        torch.manual_seed(seed); np.random.seed(seed)

        if classifier_name == "cwt_ran":
            model = CWTResidualAttentionNet(num_classes=num_classes).to(DEVICE)
        else:
            model = CNNTransformer(num_classes=num_classes).to(DEVICE)

        model, history = train_classifier_pcl(
            model, train_X_img, train_y, val_loader, n_shot, lambda_pcl, pcl_tau,
            proto_mode=proto_mode, n_real=len(train_X_real))

        crit = nn.CrossEntropyLoss()
        tm = evaluate_full(model, test_loader, crit, classes,
                           return_features=(rep == REPEATS - 1))

        rep_metric = {
            "repeat": rep, "seed": seed,
            "train_epochs": len(history["train_loss"]),
            "test_accuracy": tm["accuracy"],
            "test_f1_macro": tm["f1_macro"],
            "test_precision_macro": tm["precision_macro"],
            "test_recall_macro": tm["recall_macro"],
            "test_auc_roc_macro": tm["auc_roc_macro"],
            "per_class_f1": tm["per_class_f1"],
            "per_class_accuracy": tm["per_class_accuracy"],
            "val_best_acc": float(max(history["val_acc"])),
        }
        repeat_metrics.append(rep_metric)

        # 累加跨重复混淆矩阵 (供 §5.1 两级分解) 与 logits (供 grouped-softmax 重打分)
        acc_cm += confusion_matrix(tm["labels"], tm["preds"], labels=list(range(num_classes)))
        all_logits_list.append(tm["logits"])
        all_labels_list.append(tm["labels"])

        if tm["accuracy"] > best_acc:
            best_acc = tm["accuracy"]
            best_result = tm
            best_history = history

        print(f"    Rep {rep+1:2d}/{REPEATS}  acc={tm['accuracy']:.4f}  f1={tm['f1_macro']:.4f}  "
              f"auc={tm['auc_roc_macro']:.4f}  ep={len(history['train_loss'])}")

    accs = [r["test_accuracy"] for r in repeat_metrics]
    f1s = [r["test_f1_macro"] for r in repeat_metrics]
    aucs = [r["test_auc_roc_macro"] for r in repeat_metrics]
    summary = {
        "dataset": dataset, "classifier": classifier_name, "generator": generator_name,
        "n_shot": n_shot, "ratio": ratio,
        "lambda_pcl": lambda_pcl, "pcl_tau": pcl_tau,
        "proto_mode": proto_mode,
        "n_real": len(train_X_real), "n_fake": len(train_X) - len(train_X_real),
        "n_train_total": len(train_X),
        "mean_accuracy": float(np.mean(accs)), "std_accuracy": float(np.std(accs, ddof=1)),
        "mean_f1_macro": float(np.mean(f1s)), "std_f1_macro": float(np.std(f1s, ddof=1)),
        "mean_auc_roc": float(np.mean(aucs)), "std_auc_roc": float(np.std(aucs, ddof=1)),
        "repeats": repeat_metrics,
        "config": {
            "dataset": dataset,
            "batch_size": BATCH_SIZE, "lr": CLS_LR, "weight_decay": CLS_WD,
            "epochs": CLS_EPOCHS, "t_max": CLS_T_MAX, "patience": CLS_PATIENCE,
            "label_smoothing": LABEL_SMOOTHING,
            "preprocessor": preprocessor,
            "num_classes": num_classes,
            "model_params": sum(p.numel() for p in model.parameters()),
            "pcl_lambda": lambda_pcl,
            "pcl_tau": pcl_tau,
            "proto_mode": proto_mode,
        },
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(out_dir / "history.json", "w") as f:
        json.dump(best_history, f, indent=2)

    # ---- 两级分层指标产物 (§5.1 分解 + grouped-softmax 两级重打分) ----
    full_logits = np.concatenate(all_logits_list, axis=0)
    full_labels = np.concatenate(all_labels_list, axis=0)
    np.save(out_dir / "confusion_matrix.npy", acc_cm)
    np.save(out_dir / "logits.npy", full_logits)
    np.save(out_dir / "labels.npy", full_labels)
    groups = LOC_GROUPS.get(dataset)
    tl = grouped_softmax_twolevel(full_logits, full_labels, num_classes, groups)
    np.save(out_dir / "twolevel_cm.npy", tl["twolevel_cm"])
    np.save(out_dir / "loc_cm.npy", tl["loc_cm"])
    with open(out_dir / "twolevel_metrics.json", "w") as f:
        json.dump(tl["metrics"], f, indent=2)
    print(f"    [two-level] flat={tl['metrics']['flat_acc']:.4f} "
          f"loc={tl['metrics']['location_acc']:.4f} "
          f"twolevel={tl['metrics']['twolevel_acc']:.4f} "
          f"cond_sev_fault={tl['metrics']['cond_sev_acc_fault']:.4f}")

    if best_result is not None:
        plot_confusion_matrix(
            best_result["labels"], best_result["preds"], classes,
            f"PCL(λ={lambda_pcl}) {exp_name} {n_shot}-shot r={ratio}\nAcc={np.mean(accs):.4f}±{np.std(accs, ddof=1):.4f}",
            out_dir / "confusion_matrix.png")
        if "features" in best_result:
            plot_tsne(best_result["features"], best_result["labels"], classes,
                      f"PCL(λ={lambda_pcl}) {exp_name} {n_shot}-shot r={ratio}",
                      out_dir / "tsne.png")
    if best_history is not None:
        plot_loss_curves(best_history, f"PCL(λ={lambda_pcl}) {exp_name} {n_shot}-shot r={ratio}",
                         out_dir / "loss_curves.png")

    print(f"  → PCL(λ={lambda_pcl}) {exp_name} {dataset.upper()} {n_shot}-shot ratio={ratio}: "
          f"Acc={np.mean(accs):.4f}±{np.std(accs, ddof=1):.4f}  "
          f"F1={np.mean(f1s):.4f}  AUC={np.mean(aucs):.4f}")


def main():
    parser = argparse.ArgumentParser(description="P0优化方案一：原型对比损失(PCL)评估")
    parser.add_argument("--dataset", choices=["cwru", "pu", "jnu"], required=True)
    parser.add_argument("--classifier", choices=["cwt_ran", "cnn_transformer", "all"], default="all")
    parser.add_argument("--generator", type=str, default="all",
                        help="Generator name (e.g., wgan_gp, diffusion, proto_guided_v2, proto_v2B_fft0.01, or 'all')")
    parser.add_argument("--shot", type=int, nargs="*", default=None)
    parser.add_argument("--ratio", type=float, nargs="*", default=None)
    parser.add_argument("--lambda_pcl", type=float, nargs="*", default=None,
                        help="PCL损失权重，可指定多个值进行扫描，如 --lambda_pcl 0.1 0.3 0.5 1.0 2.0")
    parser.add_argument("--pcl_tau", type=float, default=PCL_TAU,
                        help=f"PCL温度系数，默认 {PCL_TAU}")
    parser.add_argument("--proto_mode", choices=["batch", "ema"], default="batch",
                        help="原型模式: batch=批量级原型, ema=EMA真实锚点原型")
    parser.add_argument("--use_filtered", action="store_true",
                        help="使用频谱过滤后的生成数据（generated_filtered目录）")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset
    classes = DATASET_CLASSES[dataset]
    num_classes = len(classes)

    # 默认 lambda_pcl = 0.5
    lambda_pcl_list = args.lambda_pcl if args.lambda_pcl else [PCL_LAMBDA]

    print(f"Device: {DEVICE}")
    print(f"Dataset: {dataset.upper()}  Classes: {classes}  num_classes: {num_classes}")
    print(f"PCL Lambda values: {lambda_pcl_list}")
    print(f"PCL Tau: {args.pcl_tau}")
    print(f"Proto mode: {args.proto_mode}")
    if args.use_filtered:
        print(f"Using filtered data (generated_filtered/)")
    n_params_ran = sum(p.numel() for p in CWTResidualAttentionNet(num_classes=num_classes).parameters())
    n_params_cnn = sum(p.numel() for p in CNNTransformer(num_classes=num_classes).parameters())
    print(f"CWT+RAN params: {n_params_ran:,}")
    print(f"CNN+Trans params: {n_params_cnn:,}")

    classifiers = ["cwt_ran", "cnn_transformer"] if args.classifier == "all" else [args.classifier]
    generators = ["wgan_gp", "diffusion", "proto_guided_v2"] if args.generator == "all" else [args.generator]
    shots = args.shot or SHOT_SETTINGS
    ratios = args.ratio or FAKE_RATIOS

    total = len(classifiers) * len(generators) * len(shots) * len(ratios) * len(lambda_pcl_list)
    count = 0
    for cls_name in classifiers:
        for gen_name in generators:
            for n_shot in shots:
                for ratio in ratios:
                    for lambda_pcl in lambda_pcl_list:
                        count += 1
                        print(f"\n[{count}/{total}] PCL(λ={lambda_pcl}, {args.proto_mode}) {cls_name} + {gen_name}  "
                              f"{dataset.upper()} {n_shot}-shot  ratio={ratio}")
                        run_single_config(dataset, cls_name, gen_name, n_shot, ratio,
                                          lambda_pcl, args.pcl_tau, args.skip_existing,
                                          proto_mode=args.proto_mode,
                                          use_filtered=args.use_filtered)

    print("\nAll PCL evaluation tasks complete.")


if __name__ == "__main__":
    main()
