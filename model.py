# model.py
"""
Baseline de modelagem para o nosso projeto:
- Usa RandomForest como primeiro modelo simples/robusto.
- Avalia com Purged K-Fold (López de Prado): evita vazamento temporal
  ao "purgar" observações perto da janela de teste e aplicar "embargo".
- Guarda tudo bem comentado para ser didático.

O objetivo aqui NÃO é maximizar performance já — é:
1) validar o pipeline fim-a-fim com um modelo estável,
2) obter métricas honestas (sem leakage),
3) ter um ponto de partida para futuros modelos.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,      # % de acertos (pode ser enganador em classes desbalanceadas)
    f1_score,            # média harmónica de precisão/recall (bom com classes 0/1)
    roc_auc_score,       # AUC-ROC com probabilidades (robusto ao threshold 0.5)
)
# --------------------------------------------------------------------------------------
# Purged K-Fold
# --------------------------------------------------------------------------------------
# Porquê?
# - Em séries temporais, amostras próximas no tempo "vazam" informação entre treino e teste.
# - A solução: quando definimos o bloco de TESTE, removemos (purge) vizinhanças
#   do treino à volta desse bloco, e opcionalmente aplicamos um "embargo"
#   (uma janela adicional depois do teste que também não entra no treino).
# - Isto imita o que acontece ao operar: não treinamos com dados que "tocam" o período avaliado.

@dataclass
class PurgedKFold:
    n_splits: int = 5   # nº de folds (divisões no tempo)
    purge: int = 5      # nº de barras removidas ANTES/DEPOIS da janela de teste
    embargo: float = 0.0  # fração do tamanho do teste embargada após o teste (ex.: 0.01 = 1%)

    def split(self, X: pd.DataFrame, y: pd.Series) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Gera pares (train_idx, test_idx) garantindo:
        - TESTE = bloco contínuo no tempo (ordem preservada).
        - TREINO = tudo o resto, exceto:
            * 'purge' antes e depois do bloco de teste
            * 'embargo' após o teste (percentagem do tamanho do teste)
        """
        n = len(X)
        # Particiona o índice em n_splits blocos consecutivos
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        indices = np.arange(n)

        start = 0
        for fold_size in fold_sizes:
            stop = start + fold_size
            test_idx = indices[start:stop]  # bloco de teste

            # remove (purge) área à volta do teste no conjunto de treino
            left = max(0, start - self.purge)
            right = min(n, stop + self.purge)

            train_mask = np.ones(n, dtype=bool)
            train_mask[left:right] = False  # aqui "limpamos" a zona próxima do teste

            # embargo: exclui período logo a seguir ao teste (ex.: 1% do tamanho do teste)
            emb = int(round(self.embargo * fold_size))
            if emb > 0:
                emb_end = min(n, stop + emb)
                train_mask[stop:emb_end] = False

            train_idx = indices[train_mask]
            yield train_idx, test_idx
            start = stop


# --------------------------------------------------------------------------------------
# Baseline com RandomForest
# --------------------------------------------------------------------------------------
# Porquê RF?
# - Robusto, não exige escalamento, lida bem com não linearidades e interações.
# - class_weight='balanced' compensa desbalanceamento 0/1 típico em meta-labels.
# - É um "baseline honesto": não precisa de tuning pesado para dar uma noção inicial.

def run_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    purge: int = 5,
    embargo: float = 0.0,
    n_estimators: int = 300,
    max_depth: int | None = None,
    random_state: int = 42,
) -> dict:
    """
    Treina/avalia RandomForest com Purged K-Fold e devolve médias das métricas.

    Parâmetros principais:
      - n_splits: nº de divisões temporais (k-fold).
      - purge:    nº de barras removidas (leakage control).
      - embargo:  fração do teste embargada depois do bloco de teste.
      - n_estimators/max_depth: capacidade do RF (podes afinar depois).

    Notas:
      - y esperado = meta-label (0/1). Se vierem -1, mapeamos para 0 (não-confirmado).
      - accuracy pode iludir; por isso mostramos também F1 e AUC (se possível).
    """
    # Garantir que y está em {0,1}
    y_enc = y.copy()
    if set(y_enc.unique()) - {0, 1}:
        y_enc = y_enc.replace({-1: 0})

    cv = PurgedKFold(n_splits=n_splits, purge=purge, embargo=embargo)

    accs, f1s, aucs = [], [], []

    for i, (tr, te) in enumerate(cv.split(X, y_enc), 1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y_enc.iloc[tr], y_enc.iloc[te]

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,                 # usa todos os núcleos disponíveis
            class_weight="balanced",   # compensa classes desiguais
            random_state=random_state,
        )
        clf.fit(Xtr, ytr)

        # Probabilidades para métricas threshold-free (AUC)
        proba = clf.predict_proba(Xte)[:, 1]
        # Previsões 0/1 simples com threshold 0.5 (podes ajustar depois)
        pred = (proba >= 0.5).astype(int)

        acc = accuracy_score(yte, pred)
        f1  = f1_score(yte, pred, zero_division=0)
        # AUC pode falhar se yte tiver só uma classe num fold; tratamos isso.
        try:
            auc = roc_auc_score(yte, proba)
        except ValueError:
            auc = float("nan")

        accs.append(acc); f1s.append(f1); aucs.append(auc)
        print(
            f"[Fold {i}] acc={acc:.3f}  f1={f1:.3f}  auc={auc:.3f}  "
            f"| n_train={len(tr)} n_test={len(te)}"
        )

    res = {
        "acc": float(np.nanmean(accs)),
        "f1":  float(np.nanmean(f1s)),
        "auc": float(np.nanmean(aucs)),
    }
    print("\nMédias CV:", res)
    return res


# --------------------------------------------------------------------------------------
# Atalho: carregar X/y gravados em disco (data/X.parquet, data/y.parquet)
# --------------------------------------------------------------------------------------
def run_baseline_from_disk(path_X: str, path_y: str, **kwargs) -> dict:
    """
    Conveniência para correr o baseline diretamente a partir dos ficheiros
    que o main.py guarda (X.parquet, y.parquet).
    """
    X = pd.read_parquet(path_X)
    y = pd.read_parquet(path_y)["y"]
    return run_baseline(X, y, **kwargs)
