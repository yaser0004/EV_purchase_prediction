"""Ingredient-only smooth MLP prior (track I).

Prior MLPs plateaued (0.93862 small, 0.93863 big) on the full 86-col
feature set — but that is a different, noisier signal. This model sees
only the 13 raw ingredients (6 cats + Env-as-cat one-hots, 6 numerics +
log_income), small net (64-32), heavy dropout/weight-decay: a smooth
universal approximator of the generator surface, embedding-free.

Fold loop mirrors mlp.py (fold-AUC early stopping, pos-weighted BCE,
sequential folds, test = fold-model average). Numerics standardized
with train-fold stats only.

Usage: python -m src.mlp_prior --seed 42 [--max-epochs 60 --patience 8]
"""
import argparse
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from .config import ID_COL, N_SPLITS, OUTPUTS, TARGET
from .utils import make_folds, seed_all

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PRIOR_CATS = [
    "Subsidy_Available",
    "Range_Anxiety_Level",
    "Home_Charging_Possible",
    "City_Type",
    "Current_Car_Type",
    "Gender",
]
PRIOR_NUMS = [
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Age",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
]


def prior_matrix(tr: pd.DataFrame, te: pd.DataFrame):
    """Joint target-free ingredient matrix (one-hots + raw numerics)."""
    joint = pd.concat([tr, te], ignore_index=True)
    parts = []
    for c in PRIOR_CATS:
        parts.append(pd.get_dummies(joint[c].astype("string"), prefix=c,
                                    dtype=np.float32))
    parts.append(pd.get_dummies(joint["Environmental_Concern_Level"].astype("string"),
                                prefix="Env", dtype=np.float32))
    for c in PRIOR_NUMS:
        v = pd.to_numeric(joint[c], errors="coerce")
        parts.append(pd.DataFrame(
            {c: np.asarray(v.fillna(v.median()), dtype=np.float32)}))
    inc = np.asarray(pd.to_numeric(joint["Annual_Income_USD"], errors="coerce")
                     .fillna(pd.to_numeric(joint["Annual_Income_USD"],
                                           errors="coerce").median()),
                     dtype=np.float64)
    parts.append(pd.DataFrame({"log_income": np.log(inc).astype(np.float32)}))
    full = pd.concat(parts, axis=1).fillna(0.0)
    return (full.iloc[:len(tr)].reset_index(drop=True).to_numpy(dtype=np.float32),
            full.iloc[len(tr):].reset_index(drop=True).to_numpy(dtype=np.float32))


class PriorMLP(nn.Module):
    def __init__(self, n_in, hidden=(64, 32), dropout=(0.3, 0.2)):
        super().__init__()
        layers, prev = [], n_in
        for h, p in zip(hidden, dropout):
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def _fit_fold(X_tr, y_tr, X_va, y_va, seed, max_epochs, patience, batch, lr):
    seed_all(seed)
    torch.manual_seed(seed)
    ds = TensorDataset(torch.from_numpy(X_tr),
                       torch.from_numpy(y_tr.astype(np.float32)))
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)
    model = PriorMLP(X_tr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    pos_w = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                         device=DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=3)
    v_x = torch.from_numpy(np.ascontiguousarray(X_va)).to(DEVICE)
    best_auc, best_state, bad = -1.0, None, 0
    for _ in range(max_epochs):
        model.train()
        for bx, by in dl:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(v_x)).cpu().numpy()
        auc = roc_auc_score(y_va, pv)
        sched.step(auc)
        if auc > best_auc + 1e-5:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model, best_auc


def main(seed: int = 42, max_epochs: int = 60, patience: int = 8,
         batch: int = 8192, lr: float = 3e-3):
    from .config import DATA_RAW
    seed_all(seed)
    tr = pd.read_csv(DATA_RAW / "train.csv")
    te = pd.read_csv(DATA_RAW / "test.csv")
    y = (tr[TARGET] == "Yes").astype(int).to_numpy()
    test_ids = te[ID_COL]
    X, Xt = prior_matrix(tr, te)
    print(f"[mlp_prior] device={DEVICE} dim={X.shape[1]}", flush=True)
    del tr, te
    gc.collect()

    oof = np.zeros(len(y))
    tsum = np.zeros(len(Xt))
    for fold, (itr, iva) in enumerate(make_folds(pd.Series(y), N_SPLITS, seed)):
        mu, sd = X[itr].mean(0), X[itr].std(0) + 1e-6
        model, auc = _fit_fold(((X[itr] - mu) / sd).astype(np.float32), y[itr],
                               ((X[iva] - mu) / sd).astype(np.float32), y[iva],
                               seed * 100 + fold, max_epochs, patience, batch, lr)
        print(f"[mlp_prior seed={seed}] fold={fold} AUC={auc:.5f}", flush=True)
        model.eval()
        with torch.no_grad():
            oof[iva] = torch.sigmoid(model(
                torch.from_numpy(np.ascontiguousarray(
                    ((X[iva] - mu) / sd).astype(np.float32))).to(DEVICE),
            )).cpu().numpy()
            tsum += torch.sigmoid(model(
                torch.from_numpy(np.ascontiguousarray(
                    ((Xt - mu) / sd).astype(np.float32))).to(DEVICE),
            )).cpu().numpy() / N_SPLITS
        del model
        gc.collect()
        torch.cuda.empty_cache()
    print(f"[mlp_prior seed={seed}] OOF AUC={roc_auc_score(y, oof):.5f}")
    np.save(OUTPUTS / f"oof_mlpprior_seed{seed}.npy", oof)
    np.save(OUTPUTS / f"test_mlpprior_seed{seed}.npy", tsum)
    print(f"saved outputs/oof_mlpprior_seed{seed}.npy + test_mlpprior_seed{seed}.npy")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-3)
    a = ap.parse_args()
    main(a.seed, a.max_epochs, a.patience, a.batch, a.lr)
