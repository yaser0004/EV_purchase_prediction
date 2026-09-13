"""GPU MLP with entity embeddings: 5-fold OOF + fold-averaged test preds.

Diversity model for the GBDT-only stack. Entity embeddings handle the 21
categorical/binned columns without blowing RAM with one-hots; numerics are
standardized with train-fold stats only (leak-free). Sequential folds.
"""
import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from .config import ID_COL, N_SPLITS, OUTPUTS, SUBMISSIONS, TARGET
from .features import build_features
from .utils import check_submission, make_folds, seed_all

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _encode_frame(X_all: pd.DataFrame):
    """Split joint features into numeric matrix + per-col category codes.

    Codes reserve 0 for unknown; unified categories from the joint frame
    (unsupervised, no target involved).
    """
    cat_cols = [c for c in X_all.columns
                if str(X_all[c].dtype) == "category" or c.endswith("_bin")]
    num_cols = [c for c in X_all.columns
                if c not in cat_cols and c != ID_COL
                and pd.api.types.is_numeric_dtype(X_all[c])]
    codes, cards = {}, {}
    for c in cat_cols:
        cats = pd.Categorical(X_all[c]).categories
        codes[c] = pd.Categorical(X_all[c], categories=cats).codes.astype(
            np.int32) + 1  # 0 = unknown
        cards[c] = len(cats) + 1
    X_num = X_all[num_cols].to_numpy(dtype=np.float32)
    X_cat = np.column_stack([codes[c] for c in cat_cols]).astype(np.int32)
    return X_num, X_cat, num_cols, cat_cols, [cards[c] for c in cat_cols]


class TabMLP(nn.Module):
    def __init__(self, cards, n_num, emb_rule="sqrt", hidden=(256, 128, 64),
                 dropout=(0.25, 0.15, 0.05)):
        super().__init__()
        self.embs = nn.ModuleList()
        tot = 0
        for n in cards:
            d = max(2, min(16, int(np.sqrt(n)) + 1)) if emb_rule == "sqrt" else 8
            self.embs.append(nn.Embedding(n + 1, d, padding_idx=0))
            tot += d
        layers, prev = [], tot + n_num
        for h, p in zip(hidden, dropout):
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xc, xn):
        e = torch.cat([emb(xc[:, i]) for i, emb in enumerate(self.embs)], dim=1)
        return self.net(torch.cat([e, xn], dim=1)).squeeze(1)


def _fit_fold(Xn_tr, Xc_tr, y_tr, Xn_va, Xc_va, y_va, cards, seed,
              max_epochs=40, patience=6, batch=8192, lr=3e-3):
    seed_all(seed)
    torch.manual_seed(seed)
    ds = TensorDataset(torch.from_numpy(Xc_tr), torch.from_numpy(Xn_tr),
                       torch.from_numpy(y_tr.astype(np.float32)))
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)
    model = TabMLP(cards, Xn_tr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    pos_w = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                         device=DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=3)
    v_c = torch.from_numpy(np.ascontiguousarray(Xc_va)).long().to(DEVICE)
    v_n = torch.from_numpy(np.ascontiguousarray(Xn_va)).to(DEVICE)
    best_auc, best_state, bad = -1.0, None, 0
    for _ in range(max_epochs):
        model.train()
        for bc, bn, by in dl:
            bc, bn, by = bc.long().to(DEVICE), bn.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(bc, bn), by)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(v_c, v_n)).cpu().numpy()
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


def run_cv(train_path: Path, seed: int = 42, test_path: Path | None = None,
           out_suffix: str = "", max_epochs: int = 40, patience: int = 6,
           batch: int = 8192, lr: float = 3e-3):
    seed_all(seed)
    df = pd.read_csv(train_path)
    y_raw = df[TARGET]
    y = (y_raw.astype(int).to_numpy() if pd.api.types.is_numeric_dtype(y_raw)
         else (y_raw == "Yes").astype(int).to_numpy())
    n_train = len(df)
    frames = [df.drop(columns=[TARGET])]
    test_ids = None
    if test_path is not None:
        tdf = pd.read_csv(test_path)
        test_ids = tdf[ID_COL]
        frames.append(tdf)
    X_all = build_features(pd.concat(frames, ignore_index=True))
    X_num, X_cat, num_cols, cat_cols, cards = _encode_frame(X_all)
    print(f"[mlp] device={DEVICE} numerics={len(num_cols)} "
          f"catcols={len(cat_cols)} cards={cards}", flush=True)
    del X_all, frames, df
    gc.collect()
    X_te_num = X_num[n_train:] if test_ids is not None else None
    X_te_cat = X_cat[n_train:] if test_ids is not None else None
    X_num, X_cat = X_num[:n_train], X_cat[:n_train]

    oof = np.zeros(n_train, dtype=np.float64)
    test_pred = (np.zeros(len(X_te_num)) if X_te_num is not None else None)
    for fold, (tr, va) in enumerate(make_folds(pd.Series(y), N_SPLITS, seed)):
        mu, sd = X_num[tr].mean(0), X_num[tr].std(0) + 1e-6
        Xn_tr, Xn_va = ((X_num[tr] - mu) / sd).astype(np.float32), \
            ((X_num[va] - mu) / sd).astype(np.float32)
        model, auc = _fit_fold(Xn_tr, np.ascontiguousarray(X_cat[tr]), y[tr],
                               Xn_va, np.ascontiguousarray(X_cat[va]), y[va],
                               cards, seed * 100 + fold,
                               max_epochs=max_epochs, patience=patience,
                               batch=batch, lr=lr)
        print(f"[mlp seed={seed}] fold={fold} AUC={auc:.5f}", flush=True)
        model.eval()
        with torch.no_grad():
            oof[va] = torch.sigmoid(model(
                torch.from_numpy(
                    np.ascontiguousarray(X_cat[va])).long().to(DEVICE),
                torch.from_numpy(np.ascontiguousarray(Xn_va)).to(DEVICE),
            )).cpu().numpy()
            if test_pred is not None:
                Xn_te = ((X_te_num - mu) / sd).astype(np.float32)
                test_pred += torch.sigmoid(model(
                    torch.from_numpy(
                        np.ascontiguousarray(X_te_cat)).long().to(DEVICE),
                    torch.from_numpy(np.ascontiguousarray(Xn_te)).to(DEVICE),
                )).cpu().numpy() / N_SPLITS
        del model
        gc.collect()
        torch.cuda.empty_cache()
    print(f"[mlp{out_suffix} seed={seed}] OOF AUC={roc_auc_score(y, oof):.5f}")
    out = OUTPUTS / f"oof_mlp{out_suffix}_seed{seed}.npy"
    np.save(out, oof)
    print(f"saved {out}")
    if test_pred is not None:
        tp = OUTPUTS / f"test_mlp{out_suffix}_seed{seed}.npy"
        np.save(tp, test_pred)
        sub = pd.DataFrame({ID_COL: test_ids.values, TARGET: test_pred})
        check_submission(sub, test_ids)
        sp = SUBMISSIONS / f"sub_mlp{out_suffix}_seed{seed}.csv"
        sub.to_csv(sp, index=False)
        print(f"saved {tp} and {sp}")
    return oof


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--test", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-3)
    a = ap.parse_args()
    run_cv(a.train, a.seed, a.test, a.out_suffix,
           a.max_epochs, a.patience, a.batch, a.lr)
