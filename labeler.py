# labeler.py

"""
Módulo para aplicar a marcação triple‑barrier a um DataFrame de preços.
Produz labels de {1, –1, 0} consoante:
  • 1  – atinge primeiro a barreira de take‑profit,
  • –1 – atinge primeiro a barreira de stop‑loss,
  • 0  – atinge só a barreira temporal (time‑barrier).
"""

import pandas as pd
import numpy as np

def apply_triple_barrier(
    df: pd.DataFrame,
    pt: float,
    sl: float,
    max_bars: int
) -> pd.Series:
    """
    df       – DataFrame com coluna 'close', index datetime
    pt, sl   – limites de ganho/perda (e.g. 0.005 = 0.5%)
    max_bars – número máximo de barras para a time‑barrier

    Retorna uma Series de labels alinhada com df.index.
    """
    close = df['close'].values
    timestamps = df.index
    n = len(close)
    labels = pd.Series(0, index=timestamps)

    for i in range(n):
        # Varredura até max_bars à frente
        for j in range(1, max_bars + 1):
            if i + j >= n:
                # chegou ao fim dos dados
                break
            ret = (close[i + j] / close[i] - 1)
            if ret > pt:
                labels.iloc[i] = 1
                break
            elif ret < -sl:
                labels.iloc[i] = -1
                break
        # Se não bateu pt nem sl, label fica 0 (time barrier)
    return labels

def meta_labeling(
    labels: pd.Series
) -> pd.Series:
    """
    Gera os meta-labels para as observações cujo label primário ≠ 0.
    Retorna uma Series indexada pelos timestamps com valores em {0,1}:
      1 se o rótulo primário for +1 (take-profit),
      0 se for -1 (stop-loss).
    """
    # Filtramos só os sinais não-zero
    mask = labels != 0
    meta = labels[mask].copy()
    # Transformamos +1 → 1 e –1 → 0
    meta.loc[:] = (meta.loc[:] == 1).astype(int)
    return meta