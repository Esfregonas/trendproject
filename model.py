# model.py
"""
Modelagem + Validação Temporal + Importances
--------------------------------------------
- RandomForest + Purged K-Fold (time-aware) com purge/embargo.
- Calibração (isotónica) via CalibratedClassifierCV para probabilidades calibradas.
- Métricas por fold e médias: acc, f1@0.5, roc-auc, pr-auc, f1@thr* e o próprio thr*.
- Feature importances (Gini + Permutation) guardadas em data/feature_importance.csv.
- Aceita hiperparâmetros extra via **rf_kwargs (ex.: vindos de data/rf_best.json).

Uso típico (a partir do main.py):
    from model import run_baseline
    _ = run_baseline(
        X, y,
        n_splits=5, purge=5, embargo=0.01,
        # defaults sobreponíveis por **rf_kwargs:
        n_estimators=300, max_depth=None,
        # e.g., **{"n_estimators": 500, "max_depth": 3, "min_samples_leaf": 1}
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple, Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance

# métricas adicionais (no teu metrics.py)
from metrics import pr_auc, best_threshold_by_f1


# ============================================================================
#                       Purged K-Fold (validação temporal)
# ============================================================================

@dataclass
class PurgedFold:
    """Container com índices de treino e teste para um fold temporal."""
    train_idx: np.ndarray
    test_idx: np.ndarray


def _time_ordered_kfold(n: int, n_splits: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Divide sequencialmente [0..n) em n_splits blocos contíguos:
      - cada fold de teste é um bloco contínuo no tempo,
      - treino = todo o resto (fora do bloco).
    """
    if n_splits < 2:
        raise ValueError("n_splits deve ser >= 2")

    fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
    fold_sizes[: n % n_splits] += 1

    start = 0
    for fs in fold_sizes:
        stop = start + fs
        test_idx = np.arange(start, stop)
        train_idx = np.concatenate([np.arange(0, start), np.arange(stop, n)])
        yield train_idx, test_idx
        start = stop


def _apply_purge_and_embargo(train_idx: np.ndarray,
                             test_idx: np.ndarray,
                             purge: int,
                             embargo: int,
                             n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Purge: remove do treino a vizinhança do conjunto de teste (antes e depois).
    Embargo: remove do treino um bloco imediatamente a seguir ao teste.
    """
    # Purge (remove [min- purge, max+ purge] do treino)
    lo = max(int(test_idx.min()) - int(purge), 0)
    hi = min(int(test_idx.max()) + int(purge) + 1, n)
    mask = np.ones(n, dtype=bool)
    mask[lo:hi] = False
    train_idx = np.intersect1d(train_idx, np.where(mask)[0])

    # Embargo (remove uma janela logo a seguir ao teste)
    if embargo > 0:
        e_lo = int(test_idx.max()) + 1
        e_hi = min(e_lo + int(embargo), n)
        train_idx = np.setdiff1d(train_idx, np.arange(e_lo, e_hi))

    return train_idx, test_idx


def purged_kfold_indices(n: int,
                         n_splits: int = 5,
                         purge: int = 5,
                         embargo: float | int = 0.0) -> Iterator[PurgedFold]:
    """
    Gera folds temporais com purge + embargo.

    Args
    ----
    n : int
        Tamanho total da amostra (len(X)).
    n_splits : int
        Nº de folds.
    purge : int
        Nº de índices removidos de cada lado do bloco de teste.
    embargo : float|int
        Se float (0..1), usa fração do tamanho do teste; se int, nº absoluto.

    Yield
    -----
    PurgedFold(train_idx, test_idx)
    """
    for train_idx, test_idx in _time_ordered_kfold(n, n_splits):
        if isinstance(embargo, float) and 0 < embargo < 1:
            emb = int(round(len(test_idx) * float(embargo)))
        else:
            emb = int(embargo)
        tr, te = _apply_purge_and_embargo(train_idx, test_idx, purge=int(purge), embargo=emb, n=n)
        yield PurgedFold(train_idx=tr, test_idx=te)


# ============================================================================
#                       Baseline + Calibração + Importances
# ============================================================================

def run_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    purge: int = 5,
    embargo: float | int = 0.01,
    # Defaults “seguros” — podem ser sobrepostos por **rf_kwargs:
    n_estimators: int = 300,
    max_depth: Optional[int] = None,
    random_state: int = 42,
    **rf_kwargs,  # hiperparâmetros adicionais (min_samples_leaf, max_features, ...)
) -> Dict:
    """
    Treina RandomForest com Purged K-Fold + Calibração (isotónica), imprime métricas por fold,
    calcula importances (Gini + Permutation) e guarda CSV com importances.

    Retorna dicionário com métricas de CV (médias e por fold) e DataFrame de importances.
    """
    # Garantir tipos compatíveis
    X = X.astype(float).copy()
    y = y.astype(int).copy()
    n = len(X)
    if len(y) != n:
        raise ValueError("X e y desalinhados")

    cols = list(X.columns)
    folds = list(purged_kfold_indices(n, n_splits=n_splits, purge=purge, embargo=embargo))

    # fábrica do RF
def mk_rf() -> RandomForestClassifier:
    """
    Constrói o RandomForest evitando duplicar 'class_weight' quando também vem de rf_kwargs.
    Dá prioridade ao valor do tuning (rf_best.json); se não vier, usa 'balanced_subsample'.
    """
    params = dict(rf_kwargs)                 # cópia para não mutar o original
    cw = params.pop("class_weight", "balanced_subsample")
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state,
        class_weight=cw,
        **params,                            # restantes hiperparâmetros do tuning
    )


    # coletores
    rows: List[Dict] = []

    print("=== Baseline: RandomForest (calibrated isotonic) + Purged K-Fold ===")
    print(f"n_splits={n_splits} | purge={purge} | embargo={embargo}")

    for i, f in enumerate(folds, start=1):
        base = mk_rf()
        clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
        clf.fit(X.iloc[f.train_idx], y.iloc[f.train_idx])

        proba = clf.predict_proba(X.iloc[f.test_idx])[:, 1]
        pred_fixed = (proba >= 0.5).astype(int)

        acc = accuracy_score(y.iloc[f.test_idx], pred_fixed)
        f1f = f1_score(y.iloc[f.test_idx], pred_fixed, zero_division=0)
        # roc_auc requer as probabilidades
        if len(np.unique(y.iloc[f.test_idx])) > 1:
            auc = roc_auc_score(y.iloc[f.test_idx], proba)
        else:
            auc = float("nan")
        pra = pr_auc(y.iloc[f.test_idx], proba)

        thr_star, f1_star = best_threshold_by_f1(y.iloc[f.test_idx], proba)

        rows.append({
            "fold": i,
            "acc@0.5": acc,
            "f1@0.5": f1f,
            "roc_auc": auc,
            "pr_auc": pra,
            "thr*": float(thr_star),
            "f1@thr*": f1_star,
            "n_train": int(len(f.train_idx)),
            "n_test": int(len(f.test_idx)),
        })

        print(
            f"[Fold {i}] ACC@0.5={acc:.3f}  F1@0.5={f1f:.3f}  ROC-AUC={auc:.3f}  "
            f"PR-AUC={pra:.3f}  F1@thr*={f1_star:.3f} (thr*={thr_star:.3f})  "
            f"| n_train={len(f.train_idx)} n_test={len(f.test_idx)}"
        )

    # métricas agregadas
    metrics_df = pd.DataFrame(rows)
    means = metrics_df[["acc@0.5", "f1@0.5", "roc_auc", "pr_auc", "f1@thr*"]].mean(numeric_only=True).to_dict()
    print("\nMédias CV:", {k: round(v, 4) for k, v in means.items()})

    # Reajustar no conjunto inteiro para importances (sem calibração)
    final_clf = mk_rf()
    final_clf.fit(X, y)

    # Gini importance
    gini = pd.Series(final_clf.feature_importances_, index=cols, name="gini").sort_values(ascending=False)

    # Permutation importance (em amostra)
    perm = permutation_importance(
        final_clf, X, y,
        scoring="roc_auc",
        n_repeats=10,
        random_state=random_state,
        n_jobs=-1
    )
    perm_ser = pd.Series(perm.importances_mean, index=cols, name="perm").sort_values(ascending=False)

    # Guardar CSV com as duas medidas
    out = pd.DataFrame({"gini": gini, "perm": perm_ser})
    out.index.name = "feature"
    from pathlib import Path
    Path("data").mkdir(exist_ok=True)
    out.to_csv("data/feature_importance.csv")

    # Imprimir top-10 para inspeção rápida
    print("\nTop-10 importances (ordenado por perm):")
    top = out.sort_values("perm", ascending=False).head(10)
    # to_string simples para evitar problemas de escape
    print(top.to_string(float_format=lambda v: f"{v:.4f}"))

    return {
        "cv_metrics_mean": means,
        "cv_metrics_folds": metrics_df,
        "feature_importance": out,
    }
