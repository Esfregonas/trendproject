# fracdiff.py
"""
Fractional Differentiation (fixed-width) — AFML (López de Prado, cap. 5)
Mantém memória de baixa frequência mas promove estacionaridade de alta frequência.

Funções principais:
- get_weights_ffd(d, thres): pesos truncados (fixed-width) para fracdiff
- fracdiff_ffd(series, d=0.5, thres=1e-4): série fracionada (NaN até atingir largura)
"""

from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd


def get_weights_ffd(d: float, thres: float = 1e-4, max_size: int = 10000) -> np.ndarray:
    """
    Gera pesos w_k para fractional differentiation com janela FIXA (truncada).
    Recorrência: w_0=1; w_k = w_{k-1} * (-(d - k + 1)/k)
    Trunca quando |w_k| < thres ou quando atinge max_size.

    Retorna em ordem "do mais antigo para o mais recente":
      w = [w_{m-1}, ..., w_1, w_0]
    """
    if d < 0:
        raise ValueError("d deve ser >= 0")
    w: List[float] = [1.0]
    k = 1
    while k < max_size:
        w_k = w[-1] * (-(d - (k - 1)) / k)
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    w = np.array(w[::-1], dtype=float)  # mais antigo -> mais recente
    return w


def fracdiff_ffd(series: pd.Series, d: float = 0.5, thres: float = 1e-4) -> pd.Series:
    """
    Aplica fractional differentiation com janela fixa (pesos truncados).
    - Mantém NaN até haver amostras suficientes para a convolução
    - Ignora janelas com NaN (produz NaN nesse ponto)
    """
    s = pd.Series(series).astype(float)
    idx = s.index
    w = get_weights_ffd(d, thres=thres)
    m = len(w)

    out = np.full(len(s), np.nan, dtype=float)
    x = s.values
    for t in range(m - 1, len(s)):
        window = x[t - m + 1 : t + 1]
        if np.isnan(window).any():
            continue
        out[t] = float(np.dot(w, window))  # pesos alinhados: antigo..recente
    name = f"fd_{series.name or 'x'}_d{str(d).replace('.', '')}"
    return pd.Series(out, index=idx, name=name)
