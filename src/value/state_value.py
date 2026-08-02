"""State-value network V(s): expected points from a state (roadmap 3.2d).

A small permutation-invariant deep set over the 11 entities (10 players + ball):
each entity -> shared MLP -> masked mean pool -> concat global features -> head
-> a scalar expected-points value. ~50k parameters; this does not need to be big.

Trained with the possession's realized points as the target (MSE), frames in the
first ~2 seconds down-weighted (value there is near-constant and uninformative,
3.2d). Noise augmentation (:mod:`src.value.augment`) is applied every epoch so
the permutation-invariant architecture is exercised on variable entity counts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.value import features as F
from src.value.augment import AugmentConfig, augment_sample
from src.value.features import PlayerVocab, entity_tensor


class StateValueNet(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 16, hidden: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.phi = nn.Sequential(
            nn.Linear(F.ENTITY_DIM + emb_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + F.GLOBAL_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, ents, pidx, mask, globals_):
        # ents [B,N,8], pidx [B,N], mask [B,N], globals_ [B,G]
        e = self.emb(pidx)                                   # [B,N,emb]
        x = torch.cat([ents, e], dim=-1)                     # [B,N,8+emb]
        h = self.phi(x)                                      # [B,N,hidden]
        m = mask.unsqueeze(-1)                               # [B,N,1]
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)  # masked mean
        out = self.head(torch.cat([pooled, globals_], dim=-1))    # [B,1]
        return out.squeeze(-1)


# --- Array construction ------------------------------------------------------
def build_vs_arrays(possessions, vocab: PlayerVocab):
    """Flatten possession trajectories into padded per-state arrays for V(s)."""
    E, P, M, G, Y, W = [], [], [], [], [], []
    for poss in possessions:
        for state, step in zip(poss.states, poss.step_index):
            ents, pidx, mask, glob = entity_tensor(state)
            E.append(ents)
            P.append(vocab.encode(pidx))
            M.append(mask)
            G.append(glob)
            Y.append(poss.realized_points)
            W.append(0.4 if step == 0 else 1.0)  # down-weight the first frames
    return {
        "ents": np.stack(E).astype(np.float32),
        "pidx": np.stack(P).astype(np.int64),
        "mask": np.stack(M).astype(np.float32),
        "globals": np.stack(G).astype(np.float32),
        "y": np.array(Y, dtype=np.float32),
        "w": np.array(W, dtype=np.float32),
    }


def _augment_batch(arr, rng, cfg):
    ents = arr["ents"].copy()
    pidx = arr["pidx"].copy()
    mask = arr["mask"].copy()
    for i in range(len(ents)):
        ents[i], pidx[i], mask[i] = augment_sample(ents[i], pidx[i], mask[i], rng, cfg)
    return ents, pidx, mask


class ValueModel:
    """Trained V(s): holds the net + vocab and evaluates states."""

    def __init__(self, net: StateValueNet, vocab: PlayerVocab):
        self.net = net
        self.vocab = vocab
        self.net.eval()

    @torch.no_grad()
    def value(self, state) -> float:
        return float(self.value_batch([state])[0])

    @torch.no_grad()
    def value_batch(self, states) -> np.ndarray:
        E, P, M, G = [], [], [], []
        for s in states:
            ents, pidx, mask, glob = entity_tensor(s)
            E.append(ents); P.append(self.vocab.encode(pidx)); M.append(mask); G.append(glob)
        self.net.eval()
        out = self.net(
            torch.tensor(np.stack(E), dtype=torch.float32),
            torch.tensor(np.stack(P), dtype=torch.long),
            torch.tensor(np.stack(M), dtype=torch.float32),
            torch.tensor(np.stack(G), dtype=torch.float32),
        )
        return out.numpy()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.net.state_dict(),
                    "vocab": self.vocab._map,
                    "vocab_size": self.net.emb.num_embeddings}, path)

    @staticmethod
    def load(path):
        blob = torch.load(path, weights_only=False)
        vocab = PlayerVocab()
        vocab._map = blob["vocab"]
        net = StateValueNet(blob["vocab_size"])
        net.load_state_dict(blob["state_dict"])
        return ValueModel(net, vocab)


def resolve_device(device: str | None) -> torch.device:
    """'auto' / None -> cuda if available else cpu."""
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train_value_model(
    possessions,
    vocab: PlayerVocab,
    augment: bool = True,
    epochs: int = 60,
    lr: float = 1.5e-3,
    batch_size: int = 256,
    seed: int = 0,
    val_possessions=None,
    verbose: bool = False,
    device: str | None = "auto",
    weight_decay: float = 0.0,
    early_stop: bool = False,
) -> tuple[ValueModel, dict]:
    """Train V(s) with MSE to realized points on the GPU when available.

    Training runs on ``device`` (the RTX GPU by default); the returned
    ValueModel is moved back to CPU, because scoring evaluates V one state at a
    time and those single-sample forwards are faster on CPU than paying GPU
    launch/transfer overhead per pause (the roadmap's <2 s latency path).
    """
    dev = resolve_device(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cfg = AugmentConfig()

    arr = build_vs_arrays(possessions, vocab)
    n = len(arr["y"])
    y = torch.tensor(arr["y"]).to(dev)
    w = torch.tensor(arr["w"]).to(dev)
    glob = torch.tensor(arr["globals"]).to(dev)

    net = StateValueNet(vocab_size=len(vocab)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss(reduction="none")
    best_val, best_state = float("inf"), None

    history = {"train_mse": [], "val_mse": [], "device": str(dev)}
    val_arr = build_vs_arrays(val_possessions, vocab) if val_possessions else None
    if val_arr is not None:
        v_ents = torch.tensor(val_arr["ents"]).to(dev)
        v_pidx = torch.tensor(val_arr["pidx"]).to(dev)
        v_mask = torch.tensor(val_arr["mask"]).to(dev)
        v_glob = torch.tensor(val_arr["globals"]).to(dev)
        v_y = torch.tensor(val_arr["y"]).to(dev)

    if verbose:
        print(f"Training V(s) on device: {dev}")

    for epoch in range(epochs):
        net.train()
        if augment:
            ents_np, pidx_np, mask_np = _augment_batch(arr, rng, cfg)
        else:
            ents_np, pidx_np, mask_np = arr["ents"], arr["pidx"], arr["mask"]
        ents = torch.tensor(ents_np).to(dev)
        pidx = torch.tensor(pidx_np).to(dev)
        mask = torch.tensor(mask_np).to(dev)

        perm = torch.randperm(n, device=dev)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = net(ents[idx], pidx[idx], mask[idx], glob[idx])
            losses = loss_fn(pred, y[idx])
            loss = (losses * w[idx]).mean()
            loss.backward()
            opt.step()
            epoch_loss += loss.detach().item() * len(idx)
        history["train_mse"].append(epoch_loss / n)

        if val_arr is not None:
            net.eval()
            with torch.no_grad():
                vpred = net(v_ents, v_pidx, v_mask, v_glob)
                vmse = float(((vpred - v_y) ** 2).mean())
            history["val_mse"].append(vmse)
            if early_stop and vmse < best_val:
                best_val = vmse
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            msg = f"epoch {epoch:3d}  train_mse={history['train_mse'][-1]:.4f}"
            if val_arr is not None:
                msg += f"  val_mse={history['val_mse'][-1]:.4f}"
            print(msg)

    net.to("cpu")  # portable, low-latency single-state inference for scoring
    if early_stop and best_state is not None:
        net.load_state_dict(best_state)  # restore the best-val checkpoint
        history["best_val_mse"] = best_val
    return ValueModel(net, vocab), history
