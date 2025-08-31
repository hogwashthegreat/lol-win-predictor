# train_dl_comp.py
# Deep learning baseline for composition-only win prediction from riot_comp.db
#
# Arch:
#   - Champion embedding (and optional role embedding)
#   - Permutation-invariant mean pooling per team (5 champs)
#   - Side feature (blue flag)
#   - MLP on [blue_vec, red_vec, blue-red, blue*red, side]
#   - Temperature scaling on a calibration split
#
# Usage:
#   pip install torch tqdm scikit-learn joblib
#   python train_dl_comp.py --db riot_comp.db --use-roles --since-days 60 --symmetrize
#
# Artifacts:
#   artifacts/dl_comp_model.pt       (state_dict)
#   artifacts/dl_comp_config.json    (vocab + hyperparams + metrics)

import os, time, json, sqlite3, argparse, math, random
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from joblib import dump

# ------------------ Data loading ------------------

Sample = Tuple[List[int], List[int], List[str], List[str], int, int, int]
# (blue_champs, red_champs, blue_roles, red_roles, blue_win(0/1), game_start_ms, ally_is_blue=1)

def load_samples(db_path: str, queue: int = 420) -> List[Sample]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT m.match_id, m.game_start
        FROM matches m
        JOIN teams t1 ON t1.match_id=m.match_id AND t1.team_id=100
        JOIN teams t2 ON t2.match_id=m.match_id AND t2.team_id=200
        WHERE m.queue_id=?
    """, (queue,))
    rows = cur.fetchall()
    mids = [r[0] for r in rows]
    game_start = {r[0]: int(r[1] or 0) for r in rows}

    cur.execute("SELECT match_id, team_id, win FROM teams WHERE team_id IN (100,200)")
    result = defaultdict(dict)
    for mid, tid, win in cur.fetchall():
        result[mid][tid] = int(win or 0)

    cur.execute("""
        SELECT match_id, team_id, champion_id, COALESCE(team_pos,''), COALESCE(lane,''), COALESCE(role,'')
        FROM participants
    """)
    parts = defaultdict(list)
    for mid, tid, cid, tp, ln, rl in cur.fetchall():
        parts[mid].append((int(tid), int(cid), (tp or "").upper(), (ln or "").upper(), (rl or "").upper()))
    conn.close()

    out: List[Sample] = []
    for mid in mids:
        p = parts.get(mid)
        if not p: continue
        blue = [(cid, tp) for (tid, cid, tp, _, _) in p if tid == 100]
        red  = [(cid, tp) for (tid, cid, tp, _, _) in p if tid == 200]
        if len(blue) != 5 or len(red) != 5: continue
        if mid not in result or 100 not in result[mid]: continue
        y = int(result[mid][100])

        b_ids = [c for (c, _) in blue]
        b_roles = [r for (_, r) in blue]
        r_ids = [c for (c, _) in red]
        r_roles = [r for (_, r) in red]

        out.append((b_ids, r_ids, b_roles, r_roles, y, game_start.get(mid, 0), 1))
    return out

# ------------------ Vocab ------------------

ROLE_SET = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY", ""]  # last = unknown

def build_vocabs(samples: List[Sample]) -> Tuple[Dict[int,int], Dict[str,int]]:
    champs = sorted({c for s in samples for c in (s[0] + s[1])})
    champ2idx = {c: i+1 for i, c in enumerate(champs)}  # 0 reserved
    role2idx = {r: i for i, r in enumerate(ROLE_SET)}   # includes unknown ""
    return champ2idx, role2idx

# ------------------ Dataset ------------------

class CompDataset(Dataset):
    def __init__(self, samples: List[Sample],
                 champ2idx: Dict[int,int], role2idx: Dict[str,int],
                 use_roles: bool, symmetrize: bool, since_ms: Optional[int]):
        data = []
        for s in samples:
            blue, red, br, rr, y, ts, _ = s
            if since_ms and ts and ts < since_ms: continue
            data.append((blue, red, br, rr, y, ts, 1))
            if symmetrize:
                data.append((red, blue, rr, br, 1-y, ts, 0))
        self.samples = data
        self.champ2idx = champ2idx
        self.role2idx = role2idx
        self.use_roles = use_roles

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        blue, red, br, rr, y, ts, blue_side = self.samples[i]
        b = [self.champ2idx.get(c, 0) for c in blue]
        r = [self.champ2idx.get(c, 0) for c in red]
        if self.use_roles:
            br_i = [self.role2idx.get(x, self.role2idx[""]) for x in br]
            rr_i = [self.role2idx.get(x, self.role2idx[""]) for x in rr]
        else:
            br_i = [self.role2idx[""]]*5
            rr_i = [self.role2idx[""]]*5
        return (
            torch.tensor(b, dtype=torch.long),
            torch.tensor(r, dtype=torch.long),
            torch.tensor(br_i, dtype=torch.long),
            torch.tensor(rr_i, dtype=torch.long),
            torch.tensor([blue_side], dtype=torch.float32),
            torch.tensor([y], dtype=torch.float32)
        )

# ------------------ Model ------------------

class CompNet(nn.Module):
    def __init__(self, n_champs: int, n_roles: int,
                 d_champ: int = 32, d_role: int = 8,
                 hidden: int = 128, use_roles: bool = True, dropout: float = 0.1):
        super().__init__()
        self.use_roles = use_roles
        self.champ_emb = nn.Embedding(n_champs + 1, d_champ, padding_idx=0)
        self.role_emb  = nn.Embedding(n_roles, d_role) if use_roles else None
        d_in = d_champ + (d_role if use_roles else 0)
        # MLP operates on pooled team vectors + interactions
        self.mlp = nn.Sequential(
            nn.Linear(d_in*4 + 1, hidden),  # [blue, red, blue-red, blue*red, side]
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # Init
        nn.init.xavier_uniform_(self.champ_emb.weight)
        if self.role_emb is not None:
            nn.init.xavier_uniform_(self.role_emb.weight)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def team_encode(self, champs: torch.Tensor, roles: torch.Tensor) -> torch.Tensor:
        # champs: [B,5], roles: [B,5]
        ce = self.champ_emb(champs)  # [B,5,d_c]
        if self.use_roles:
            re = self.role_emb(roles) # [B,5,d_r]
            x = torch.cat([ce, re], dim=-1)  # [B,5,d_c+d_r]
        else:
            x = ce
        # Permutation-invariant pooling
        return x.mean(dim=1)  # [B,d]

    def forward(self, blue_c, red_c, blue_r, red_r, side_flag):
        b_vec = self.team_encode(blue_c, blue_r)  # [B,d]
        r_vec = self.team_encode(red_c, red_r)    # [B,d]
        diff  = b_vec - r_vec
        prod  = b_vec * r_vec
        z = torch.cat([b_vec, r_vec, diff, prod, side_flag], dim=1)  # [B, 4d+1]
        logit = self.mlp(z).squeeze(1)  # [B]
        return logit

# ------------------ Utils ------------------

def ece_score(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0,1,n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (p >= lo) & (p < hi) if i < n_bins-1 else (p >= lo) & (p <= hi)
        if not np.any(mask): continue
        ece += (mask.mean()) * abs(y[mask].mean() - p[mask].mean())
    return float(ece)

def evaluate(model, loader, device, T: float = 1.0):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for bc, rc, br, rr, side, y in loader:
            bc, rc, br, rr, side = bc.to(device), rc.to(device), br.to(device), rr.to(device), side.to(device)
            y = y.to(device)
            logits = model(bc, rc, br, rr, side) / T
            p = torch.sigmoid(logits)
            ys.append(y.cpu().numpy().ravel())
            ps.append(p.cpu().numpy().ravel())
    y = np.concatenate(ys); p = np.concatenate(ps)
    return {
        "AUC": float(roc_auc_score(y, p)),
        "LogLoss": float(log_loss(y, p, labels=[0,1])),
        "Brier": float(brier_score_loss(y, p)),
        "ECE@10": ece_score(p, y, n_bins=10),
        "mean_p": float(p.mean()),
        "N": int(y.size)
    }, p, y

def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    # Simple scalar temperature via Newton steps on NLL
    T = 1.0
    for _ in range(50):
        z = logits / T
        p = 1/(1+np.exp(-z))
        # gradient dNLL/dT
        grad = np.mean((p - y) * z / T)
        # hessian approx
        hess = np.mean(p*(1-p)*(z**2) / (T**2)) + 1e-8
        step = grad / hess
        T = max(0.05, min(10.0, T + step))  # clamp
        if abs(step) < 1e-5:
            break
    return float(T)

# ------------------ Training ------------------

def main():
    ap = argparse.ArgumentParser(description="DL comp-only win predictor (PyTorch)")
    ap.add_argument("--db", default="riot_comp.db")
    ap.add_argument("--since-days", type=int, default=60)
    ap.add_argument("--use-roles", action="store_true")
    ap.add_argument("--symmetrize", action="store_true")

    ap.add_argument("--d-champ", type=int, default=32)
    ap.add_argument("--d-role", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.10)

    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)

    ap.add_argument("--calib-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    print("Loading DB…")
    samples = load_samples(args.db)
    print(f"Total matches: {len(samples)}")

    # Temporal filter
    if args.since_days > 0:
        now_ms = int(time.time()*1000)
        cutoff = now_ms - args.since_days*24*3600*1000
        pre = len(samples)
        samples = [s for s in samples if s[5] and s[5] >= cutoff]
        print(f"Filtered to last {args.since_days} days: {len(samples)} (from {pre})")
    if not samples:
        raise SystemExit("No samples after filtering.")

    # Temporal split
    samples.sort(key=lambda x: x[5])
    n = len(samples)
    n_te = max(1, int(n*args.test_ratio))
    n_ca = max(1, int(n*args.calib_ratio))
    n_tr = n - n_ca - n_te
    train_set = samples[:n_tr]
    calib_set = samples[n_tr:n_tr+n_ca]
    test_set  = samples[n_tr+n_ca:]
    print(f"Splits → Train {len(train_set)} | Calib {len(calib_set)} | Test {len(test_set)}")

    champ2idx, role2idx = build_vocabs(train_set)  # fit vocab on train only

    since_ms = None  # already filtered temporally
    ds_tr = CompDataset(train_set, champ2idx, role2idx, args.use_roles, args.symmetrize, since_ms)
    ds_ca = CompDataset(calib_set, champ2idx, role2idx, args.use_roles, args.symmetrize, since_ms)
    ds_te = CompDataset(test_set,  champ2idx, role2idx, args.use_roles, args.symmetrize, since_ms)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)
    dl_ca = DataLoader(ds_ca, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = CompNet(n_champs=len(champ2idx), n_roles=len(role2idx),
                    d_champ=args.d_champ, d_role=args.d_role,
                    hidden=args.hidden, use_roles=args.use_roles, dropout=args.dropout).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    best_auc = -1.0
    best_state = None
    patience = 4
    bad = 0

    print("Training…")
    for epoch in range(1, args.epochs+1):
        model.train()
        tot_loss = 0.0; n_batches = 0
        for bc, rc, br, rr, side, y in dl_tr:
            bc, rc, br, rr, side, y = bc.to(device), rc.to(device), br.to(device), rr.to(device), side.to(device), y.to(device)
            logits = model(bc, rc, br, rr, side)
            loss = loss_fn(logits, y.view_as(logits))
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += float(loss.item()); n_batches += 1
        m_calib, _, _ = evaluate(model, dl_ca, device)
        m_test, _, _  = evaluate(model, dl_te, device)
        print(f"Epoch {epoch:02d} | train loss {tot_loss/max(1,n_batches):.4f} | CALIB AUC {m_calib['AUC']:.3f} | TEST AUC {m_test['AUC']:.3f}")

        if m_calib['AUC'] > best_auc:
            best_auc = m_calib['AUC']; best_state = {k: v.cpu() for k,v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    # Load best on calib
    if best_state is not None:
        model.load_state_dict(best_state)

    # Temperature scaling on calib
    model.eval()
    logits_list = []
    y_list = []
    with torch.no_grad():
        for bc, rc, br, rr, side, y in dl_ca:
            bc, rc, br, rr, side = bc.to(device), rc.to(device), br.to(device), rr.to(device), side.to(device)
            logits = model(bc, rc, br, rr, side)
            logits_list.append(logits.cpu().numpy())
            y_list.append(y.numpy())
    calib_logits = np.concatenate(logits_list).ravel()
    calib_y = np.concatenate(y_list).ravel()
    T = fit_temperature(calib_logits, calib_y)
    print(f"Fitted temperature: {T:.3f}")

    # Final eval
    m_calib, _, _ = evaluate(model, dl_ca, device, T=T)
    m_test, p_test, y_test = evaluate(model, dl_te, device, T=T)
    print("CALIB metrics:", json.dumps(m_calib, indent=2))
    print("TEST  metrics:", json.dumps(m_test, indent=2))

    # Save
    os.makedirs(args.artifacts, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.artifacts, "dl_comp_model.pt"))
    cfg = {
        "champ2idx": champ2idx,
        "role2idx": role2idx,
        "use_roles": args.use_roles,
        "symmetrize": args.symmetrize,
        "d_champ": args.d_champ,
        "d_role": args.d_role,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "since_days": args.since_days,
        "calib_ratio": args.calib_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "metrics_calib": m_calib,
        "metrics_test": m_test,
        "temperature": T
    }
    with open(os.path.join(args.artifacts, "dl_comp_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print("Saved → artifacts/dl_comp_model.pt and artifacts/dl_comp_config.json")

if __name__ == "__main__":
    main()
