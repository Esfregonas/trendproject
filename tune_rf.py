# tune_rf.py
from __future__ import annotations
import json, random
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from model import purged_kfold_indices  # já existe no teu projeto

def sample_params(rng: random.Random) -> dict:
    return {
        "n_estimators": rng.randrange(200, 801, 50),           # 200..800 step 50
        "max_depth":    rng.choice([None] + list(range(3, 21))),
        "min_samples_leaf": rng.randrange(1, 11),               # 1..10
        "max_features": rng.choice(["sqrt", "log2", 0.5, 0.7, 1.0]),
        "class_weight": "balanced_subsample",
    }

def eval_params(X: pd.DataFrame, y: pd.Series, params: dict,
                n_splits: int, purge: int, embargo: float, seed: int) -> float:
    n = len(X)
    folds = list(purged_kfold_indices(n, n_splits=n_splits, purge=purge, embargo=embargo))
    aucs = []
    for f in folds:
        clf = RandomForestClassifier(
            n_jobs=-1, random_state=seed, **params
        )
        clf.fit(X.iloc[f.train_idx], y.iloc[f.train_idx])
        proba = clf.predict_proba(X.iloc[f.test_idx])[:, 1]
        auc = roc_auc_score(y.iloc[f.test_idx], proba)
        aucs.append(auc)
    return float(np.mean(aucs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_iter", type=int, default=20)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--purge", type=int, default=5)
    ap.add_argument("--embargo", type=float, default=0.01)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    outdir = Path("data"); outdir.mkdir(exist_ok=True)

    print(">> a carregar X,y de data/")
    X = pd.read_parquet(outdir / "X.parquet").astype(float)
    y = pd.read_parquet(outdir / "y.parquet")["y"].astype(int)
    print(f"X: {X.shape} | y: {y.value_counts().to_dict()}")

    rng = random.Random(args.random_state)

    rows = []
    best = {"score": -np.inf, "params": None}
    for _ in tqdm(range(args.n_iter), desc="Random search"):
        params = sample_params(rng)
        score = eval_params(X, y, params, args.n_splits, args.purge, args.embargo, seed=args.random_state)
        rows.append({"score_auc": score, **params})
        if score > best["score"]:
            best = {"score": score, "params": params}

    df = pd.DataFrame(rows).sort_values("score_auc", ascending=False).reset_index(drop=True)
    df.to_csv(outdir / "rf_tuning.csv", index=False)

    with open(outdir / "rf_best.json", "w") as f:
        json.dump({"best_params": best["params"], "best_score_auc_cv": best["score"]}, f, indent=2)

    print("\n=== Melhor conjunto de hiperparâmetros ===")
    print(best["params"])
    print(f"AUC (média CV): {best['score']:.4f}")

    print("\nTop-5 combinações por AUC:")
    print(df.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
