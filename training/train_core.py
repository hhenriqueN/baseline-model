"""Core training loop shared by CV-fold and final training (spec section 17-18)."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

TRAINING_CONFIG = {
    "seed": 42,
    "optimizer": "AdamW",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 256,
    "max_epochs": 300,
    "early_stopping_patience": 25,
    "grad_clip_norm": 5.0,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TrainResult:
    best_state_dict: dict
    best_epoch: int
    best_val_loss: float
    history: list = field(default_factory=list)  # [{"epoch":..,"train_loss":..,"val_loss":..}]


def train_model(model: nn.Module, X_train, y_train, X_val, y_val, cfg: dict = TRAINING_CONFIG,
                  device: str = "cpu") -> TrainResult:
    set_seed(cfg["seed"])
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(X_train, y_train)
    g = torch.Generator()
    g.manual_seed(cfg["seed"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, generator=g)

    X_val = X_val.to(device)
    y_val = y_val.to(device)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(cfg["max_epochs"]):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
            opt.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses))

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = float(loss_fn(val_pred, y_val).item())

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss - 1e-9:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg["early_stopping_patience"]:
                break

    return TrainResult(best_state_dict=best_state, best_epoch=best_epoch,
                         best_val_loss=best_val_loss, history=history)


def train_model_no_val(model: nn.Module, X_train, y_train, n_epochs: int, cfg: dict = TRAINING_CONFIG,
                         device: str = "cpu") -> TrainResult:
    """Final training: fixed epoch count (median best-epoch from CV), no early
    stopping (no held-out split for the final model -- spec section 18)."""
    set_seed(cfg["seed"])
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(X_train, y_train)
    g = torch.Generator()
    g.manual_seed(cfg["seed"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, generator=g)

    history = []
    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
            opt.step()
            train_losses.append(loss.item())
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses))})

    return TrainResult(best_state_dict=copy.deepcopy(model.state_dict()), best_epoch=n_epochs - 1,
                         best_val_loss=float("nan"), history=history)
