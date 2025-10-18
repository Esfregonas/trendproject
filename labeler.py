# labeler.py
"""
Labeling (López de Prado): Triple-Barrier + Meta-labeling
=========================================================

O que este módulo faz:
1) apply_triple_barrier  -> gera labels primários {-1, 0, +1}
   (+1: bateu take-profit; -1: bateu stop; 0: não bateu nenhuma até à vertical)
2) meta_labeling         -> reduz para {0,1} para meta-model (baseline deste projeto)
   (1 se label primário == +1; 0 se 0 ou -1)

Extras:
- Barra de progresso (tqdm) durante o cálculo das labels.
- Código claro e comentado para aprendizagem e manutenção.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- tqdm com fallback (se não estiver instalado, segue sem progress bar) -------------
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback simples
    def tqdm(iterable, **kwargs):
        return iterable


# ------------------------- helpers ----------------------------------------------------
def _first_crossing_index(
    high: pd.Series, low: pd.Series, up: float, dn: float
) -> int | None:
    """
    Devolve o deslocamento (0..n-1) da **primeira** barra que cruza:
      - 'high >= up' (upper barrier), ou
      - 'low  <= dn' (lower barrier),
    o que ocorrer mais cedo. Se nada cruzar, devolve None.
    """
    up_hits = np.flatnonzero(high.values >= up)
    dn_hits = np.flatnonzero(low.values  <= dn)

    if len(up_hits) == 0 and len(dn_hits) == 0:
        return None

    up_idx = up_hits[0] if len(up_hits) else None
    dn_idx = dn_hits[0] if len(dn_hits) else None

    if (up_idx is not None) and (dn_idx is not None):
        return int(min(up_idx, dn_idx))
    return int(up_idx if up_idx is not None else dn_idx)


# ------------------------- API pública ------------------------------------------------
def apply_triple_barrier(
    vb: pd.DataFrame,
    pt: float = 0.005,
    sl: float = 0.005,
    max_bars: int = 10,
) -> pd.Series:
    """
    Calcula labels {-1, 0, +1} sobre um DataFrame OHLCV (idealmente **volume bars**).

    Regras:
      +1 → primeira barreira atingida foi a superior (take-profit)
      -1 → primeira barreira atingida foi a inferior (stop-loss)
       0 → nenhuma barreira antes da vertical (max_bars)

    Parâmetros
    ----------
    vb : DataFrame com colunas ['open','high','low','close'] e índice temporal.
    pt : percentagem da barreira superior (ex.: 0.005 = +0.5%).
    sl : percentagem da barreira inferior (ex.: 0.005 = -0.5%).
    max_bars : número máximo de barras observadas (vertical barrier).

    Retorna
    -------
    pd.Series (dtype=int) indexada como `vb`, com valores em {-1, 0, +1}.
    """
    # cast explícito para evitar problemas de dtype
    close = vb["close"].astype(float)
    high  = vb["high"].astype(float)
    low   = vb["low"].astype(float)

    n = len(vb)
    out = np.zeros(n, dtype=np.int8)  # pequeno e suficiente

    # Loop claro + barra de progresso; para janelas muito longas, podemos vectorizar/numba
    for i in tqdm(range(n), desc="Triple-Barrier", mininterval=1.0):
        # níveis absolutos das barreiras a partir do preço atual
        up = close.iloc[i] * (1.0 + pt)
        dn = close.iloc[i] * (1.0 - sl)

        # janela futura até à vertical (exclui a barra i)
        start = i + 1
        stop  = min(n, i + 1 + max_bars)
        if start >= stop:
            out[i] = 0
            continue

        h = high.iloc[start:stop]
        l = low.iloc[start:stop]

        j = _first_crossing_index(h, l, up, dn)
        if j is None:
            out[i] = 0
        else:
            k = start + j  # índice absoluto do crossing
            # Nota: se ambas cruzarem na mesma barra, a ordem relativa decide;
            # aqui damos prioridade ao que efetivamente cruzou (>= up ou <= dn).
            if high.iloc[k] >= up and not (low.iloc[k] <= dn):
                out[i] = 1
            elif low.iloc[k] <= dn and not (high.iloc[k] >= up):
                out[i] = -1
            else:
                # Empate raro: se ambos cruzam na mesma vela, marcamos por proximidade.
                # (podes alterar esta regra consoante a tua convenção)
                out[i] = 1 if (high.iloc[k] - up) >= (dn - low.iloc[k]) else -1

    labels = pd.Series(out.astype(int), index=vb.index, name="tb_label")
    return labels


def meta_labeling(labels: pd.Series) -> pd.Series:
    """
    Meta-label usada no baseline deste projeto.

    Convenção simplificada:
      - 1 se label primário == +1 (movimento a favor)
      - 0 se label primário == 0 (neutro) ou -1 (contra)

    Observação: no meta-labeling "clássico" usaríamos também um vetor `side` (sinal do
    modelo primário) e marcaríamos 1 quando o retorno realizado confirmasse esse `side`.
    """
    y = labels.map({1: 1, 0: 0, -1: 0}).astype(int)
    y.name = "meta"
    return y
