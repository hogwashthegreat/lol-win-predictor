# train_comp_model.py
# Composition-only win predictor from riot_comp.db (SQLite).
#
# Improvements:
#   - Symmetrization (optional): add a swapped-side copy of every sample.
#   - Side feature: model can learn blue-side advantage explicitly.
#   - Recent filter: train on last N days to reduce patch drift.
#   - Optional enemy synergy features.
#   - Temporal split: TRAIN → CALIB → TEST (no leakage).
#
# Usage:
#   pip install scikit-learn joblib tqdm
#   python train_comp_model.py --db riot_comp.db --n-features 1048576 \
#     --use-roles --symmetrize --enemy-synergy --since-days 60 \
#     --calib-ratio 0.1 --test-ratio 0.1
#
# Output:
#   artifacts/comp_model.joblib x (Calibrated sklearn model + metrics)

import os, time, sqlite3, argparse, json
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from itertools import combinations, product

import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.pipeline import Pipeline
from joblib import dump

# -----------------------
# Data loading
# -----------------------

# Sample tuple:
# (ally_ids, enemy_ids, ally_roles, enemy_roles, label, game_start_ms, ally_is_blue)
Sample = Tuple[List[int], List[int], List[str], List[str], int, int, int]

def load_matches(db_path: str, queue: int = 420) -> Tuple[List[Sample], int]:
    """
    Returns samples for BLUE vs RED:
      ally_ids/enemy_ids are BLUE/RED champs, label=1 if BLUE won.
      ally_is_blue=1 for originals (used when symmetrizing).
    Skips matches without 5 per side or missing team rows.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT m.match_id, m.game_start
        FROM matches m
        JOIN teams t1 ON t1.match_id = m.match_id AND t1.team_id = 100
        JOIN teams t2 ON t2.match_id = m.match_id AND t2.team_id = 200
        WHERE m.queue_id = ?
    """, (queue,))
    rows = cur.fetchall()
    match_ids = [r[0] for r in rows]
    game_start_map = {r[0]: int(r[1] or 0) for r in rows}

    if not match_ids:
        conn.close()
        return [], 0

    # results
    cur.execute("""
        SELECT match_id, team_id, win
        FROM teams
        WHERE team_id IN (100,200)
    """)
    result_map = defaultdict(dict)
    for mid, tid, win in cur.fetchall():
        result_map[mid][tid] = int(win or 0)

    # participants
    cur.execute("""
        SELECT match_id, team_id, champion_id, COALESCE(team_pos,''), COALESCE(lane,''), COALESCE(role,'')
        FROM participants
    """)
    part_map = defaultdict(list)
    for mid, tid, cid, team_pos, lane, role in cur.fetchall():
        part_map[mid].append((int(tid), int(cid), team_pos.strip().upper(), lane.strip().upper(), role.strip().upper()))
    conn.close()

    samples: List[Sample] = []
    skipped = 0
    for mid in match_ids:
        parts = part_map.get(mid)
        if not parts:
            skipped += 1
            continue
        blue = [(cid, tp) for (tid, cid, tp, ln, rl) in parts if tid == 100]
        red  = [(cid, tp) for (tid, cid, tp, ln, rl) in parts if tid == 200]
        if len(blue) != 5 or len(red) != 5:
            skipped += 1
            continue
        if mid not in result_map or 100 not in result_map[mid]:
            skipped += 1
            continue
        label = int(result_map[mid][100])

        ally_ids   = [c for (c, _) in blue]
        ally_roles = [tp for (_, tp) in blue]
        enemy_ids  = [c for (c, _) in red]
        enemy_roles= [tp for (_, tp) in red]

        samples.append((ally_ids, enemy_ids, ally_roles, enemy_roles, label, game_start_map.get(mid, 0), 1))

    return samples, skipped

# -----------------------
# Feature building
# -----------------------

def comp_features(ally: List[int], enemy: List[int],
                  ally_roles: Optional[List[str]] = None,
                  enemy_roles: Optional[List[str]] = None,
                  ally_is_blue: int = 1,
                  use_roles: bool = True,
                  include_enemy_synergy: bool = False) -> Dict[str, float]:
    """
    Dict features for one comp matchup:
      - ally/enemy unigrams
      - ally synergy pairs (+ optional enemy synergy pairs)
      - ally vs enemy cross pairs
      - role-tagged unigrams (keeps champ↔role pairing)
      - side feature (learn blue advantage)
    """
    f: Dict[str, float] = {}

    # Preserve original order for role-tagged features
    ally = list(map(int, ally))
    enemy = list(map(int, enemy))

    # Sorted copies for deterministic pairs
    a_sorted = sorted(ally)
    e_sorted = sorted(enemy)

    # Unigrams
    for c in a_sorted: f[f"ally:c{c}"] = 1.0
    for c in e_sorted: f[f"enemy:c{c}"] = 1.0

    # Role-tagged unigrams (use original alignment)
    if use_roles and ally_roles and enemy_roles and len(ally_roles) == 5 and len(enemy_roles) == 5:
        for c, r in zip(ally, ally_roles):
            r = (r or "").upper()
            if r: f[f"allyrole:{r}:c{c}"] = 1.0
        for c, r in zip(enemy, enemy_roles):
            r = (r or "").upper()
            if r: f[f"enemyrole:{r}:c{c}"] = 1.0

    # Ally synergy
    for c1, c2 in combinations(a_sorted, 2):
        f[f"allypair:{c1}-{c2}"] = 1.0

    # Enemy synergy (optional)
    if include_enemy_synergy:
        for c1, c2 in combinations(e_sorted, 2):
            f[f"enemypair:{c1}-{c2}"] = 1.0

    # Ally vs enemy cross pairs
    for ca, ce in product(a_sorted, e_sorted):
        f[f"counter:{ca}-{ce}"] = 1.0

    # Side feature
    f["side:blue"] = 1.0 if ally_is_blue else 0.0

    return f

# -----------------------
# Utility
# -----------------------

def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (p >= lo) & (p < hi) if i < n_bins-1 else (p >= lo) & (p <= hi)
        if not np.any(mask): continue
        p_bin = p[mask]; y_bin = y[mask]
        ece += (p_bin.size / p.size) * abs(y_bin.mean() - p_bin.mean())
    return float(ece)

def symmetrize(samples: List[Sample]) -> List[Sample]:
    """Add a swapped-side copy: (RED vs BLUE, label'=1-label, ally_is_blue=0)."""
    aug: List[Sample] = []
    for ally, enemy, aroles, eroles, y, ts, _ in samples:
        aug.append((ally, enemy, aroles, eroles, y, ts, 1))
        aug.append((enemy, ally, eroles, aroles, 1 - y, ts, 0))
    return aug

# -----------------------
# Train / Evaluate
# -----------------------

def train(db: str,
          n_features: int = 1<<20,   # 1,048,576 hashed dims
          calib_ratio: float = 0.1,
          test_ratio: float = 0.1,
          since_days: int = 0,
          use_roles: bool = True,
          include_enemy_synergy: bool = False,
          symmetrize_data: bool = False,
          C: float = 1.0,
          max_iter: int = 300,
          random_state: int = 42,
          artifacts_dir: str = "artifacts"):

    print("Loading samples from DB…")
    samples, skipped = load_matches(db, queue=420)
    print(f"Loaded {len(samples)} samples ({skipped} skipped).")

    # Filter to recent window if requested
    if since_days and since_days > 0:
        now_ms = int(time.time() * 1000)
        cutoff = now_ms - since_days * 24 * 3600 * 1000
        pre = len(samples)
        samples = [s for s in samples if s[5] and s[5] >= cutoff]
        print(f"Filtered to last {since_days} days: {len(samples)} (from {pre})")

    if not samples:
        raise SystemExit("No samples after filtering.")

    # Temporal split: oldest → train, middle → calib, newest → test
    samples.sort(key=lambda x: x[5])  # by game_start_ms
    n_total = len(samples)
    n_test  = max(1, int(n_total * test_ratio))
    n_calib = max(1, int(n_total * calib_ratio))
    n_train = n_total - n_calib - n_test
    if n_train <= 0:
        raise SystemExit("Not enough samples for the requested calib/test ratios.")

    train_set = samples[:n_train]
    calib_set = samples[n_train:n_train+n_calib]
    test_set  = samples[n_train+n_calib:]

    # Optionally symmetrize AFTER splitting (to keep temporal integrity)
    if symmetrize_data:
        train_set = symmetrize(train_set)
        calib_set = symmetrize(calib_set)
        test_set  = symmetrize(test_set)
        print(f"Symmetrized: Train {len(train_set)} | Calib {len(calib_set)} | Test {len(test_set)}")

    print(f"Splits → Train: {len(train_set)} | Calib: {len(calib_set)} | Test: {len(test_set)}")

    # Build feature dicts
    def featurize(batch):
        feats, labels = [], []
        for ally, enemy, aroles, eroles, label, _, ally_is_blue in batch:
            feats.append(comp_features(
                ally, enemy, aroles, eroles,
                ally_is_blue=ally_is_blue,
                use_roles=use_roles,
                include_enemy_synergy=include_enemy_synergy
            ))
            labels.append(int(label))
        return feats, np.array(labels, dtype=np.int8)

    feats_tr, y_tr = featurize(train_set)
    feats_ca, y_ca = featurize(calib_set)
    feats_te, y_te = featurize(test_set)

    # Diagnostics: blue win rate on TEST
    base_rate = float(y_te.mean())
    print(f"TEST base rate (blue win%): {base_rate:.3f}")

    # Pipeline: hasher -> logistic
    pipe = Pipeline(steps=[
        ("hash", FeatureHasher(n_features=n_features, input_type="dict", alternate_sign=True)),
        ("clf", LogisticRegression(
            solver="saga",
            penalty="l2",
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            n_jobs=None,
            verbose=0,
        )),
    ])

    print("Fitting base model…")
    pipe.fit(feats_tr, y_tr)

    # Calibrate on calib set (prefit; deprecation warning on sklearn>=1.6 is OK)
    print("Calibrating probabilities (isotonic)…")
    calib = CalibratedClassifierCV(estimator=pipe, cv="prefit", method="isotonic")
    calib.fit(feats_ca, y_ca)

    # Evaluate on TEST
    print("Evaluating on TEST…")
    p_te = calib.predict_proba(feats_te)[:, 1]
    auc   = roc_auc_score(y_te, p_te)
    ll    = log_loss(y_te, p_te, labels=[0,1])
    brier = brier_score_loss(y_te, p_te)
    ece   = expected_calibration_error(p_te, y_te, n_bins=10)

    metrics = {
        "AUC": float(auc),
        "LogLoss": float(ll),
        "Brier": float(brier),
        "ECE@10": float(ece),
        "N_train": int(len(train_set)),
        "N_calib": int(len(calib_set)),
        "N_test": int(len(test_set)),
        "n_features": int(n_features),
        "use_roles": bool(use_roles),
        "enemy_synergy": bool(include_enemy_synergy),
        "symmetrize": bool(symmetrize_data),
        "since_days": int(since_days),
        "C": float(C),
        "max_iter": int(max_iter),
    }
    print("Metrics (TEST):", json.dumps(metrics, indent=2))

    # Save artifact
    os.makedirs(artifacts_dir, exist_ok=True)
    artifact_path = os.path.join(artifacts_dir, "comp_model.joblib")
    dump({"model": calib, "cfg": metrics}, artifact_path)
    print(f"Saved → {artifact_path}")
    print(f"Mean predicted P(win) on TEST: {float(p_te.mean()):.3f}")

def main():
    ap = argparse.ArgumentParser(description="Train composition-only LoL win predictor")
    ap.add_argument("--db", default="riot_comp.db")
    ap.add_argument("--n-features", type=int, default=1<<20, help="hashed feature dimension (e.g., 2^20=1048576)")
    ap.add_argument("--calib-ratio", type=float, default=0.1, help="fraction for calibration split")
    ap.add_argument("--test-ratio", type=float, default=0.1, help="fraction for test split")
    ap.add_argument("--since-days", type=int, default=0, help="filter to last N days (0 = disabled)")
    ap.add_argument("--use-roles", action="store_true", help="include role-tagged features when teamPosition is available")
    ap.add_argument("--enemy-synergy", action="store_true", help="add enemy synergy pair features")
    ap.add_argument("--symmetrize", action="store_true", help="add swapped-side copies of all samples")
    ap.add_argument("--C", type=float, default=1.0, help="inverse regularization for LogisticRegression")
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train(db=args.db,
          n_features=args.n_features,
          calib_ratio=args.calib_ratio,
          test_ratio=args.test_ratio,
          since_days=args.since_days,
          use_roles=bool(args.use_roles),
          include_enemy_synergy=bool(args.enemy_synergy),
          symmetrize_data=bool(args.symmetrize),
          C=args.C,
          max_iter=args.max_iter,
          random_state=args.seed)

if __name__ == "__main__":
    main()
