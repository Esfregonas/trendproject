# weights.py
"""
Pesos de treino segundo AFML:
- Uniqueness por concorrência (nº de eventos sobrepostos)
- Time-decay para dar mais ênfase a eventos recentes
- Helpers para construir t1 (fim do evento) quando usamos max_bars fixo

Referências: López de Prado, 'Advances in Financial Machine Learning', cap. 4.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# t1 a partir de uma janela fixa (ex.: max_bars do triple-barrier)
# -------------------------------------------------------------------------
def get_t1_from_max_bars(index: pd.Index, max_bars: int) -> pd.Series:
    """
    Constroi 't1' (fim do evento) deslocando cada timestamp 'max_bars' à frente.
    O último evento termina no último timestamp disponível.
    Retorna Series t1 indexada pelos timestamps originais (event starts).
    """
    if max_bars <= 0:
        raise ValueError("max_bars deve ser > 0")
    idx = pd.Index(index)
    pos = pd.Series(range(len(idx)), index=idx)
    end_pos = (pos + max_bars).clip(upper=len(idx) - 1).astype(int)
    t1 = pd.Series(idx[end_pos.values], index=idx, name="t1")
    return t1


# -------------------------------------------------------------------------
# Concorrência (nº de eventos ativos em cada instante) — VERSÃO ROBUSTA
# -------------------------------------------------------------------------
def _concurrency_series(index: pd.Index, t1: pd.Series) -> pd.Series:
    """
    Constrói uma série de concorrência (nº de eventos ativos) para todo o timeline.
    Implementação robusta a fins (t1) que não existem no índice de trabalho
    (ex.: após dropna). Usa 'difference array' (+1 no início, -1 no fim) + cumsum.
    """
    idx = pd.Index(index)

    # Array de diferenças, inicializado a zero
    bumps = pd.Series(0, index=idx, dtype="int64")

    # +1 nos inícios (sempre existem porque são o próprio idx de eventos)
    starts = t1.index.intersection(idx)
    if len(starts):
        bumps.loc[starts] = bumps.loc[starts].add(1, fill_value=0).astype(int)

    # -1 nos fins (apenas quando o fim existir no índice alvo)
    # reindex ao idx para descartar fins fora do timeline atual, e pegar os valores (timestamps)
    t1_on_idx = t1.reindex(idx).dropna()
    ends = pd.Index(t1_on_idx.values).intersection(idx)
    if len(ends):
        bumps.loc[ends] = bumps.loc[ends].sub(1, fill_value=0).astype(int)

    # cumsum => concorrência
    conc = bumps.cumsum()
    conc.name = "concurrency"
    conc[conc < 1] = 1  # salvaguarda (evita divisão por zero)
    return conc


# -------------------------------------------------------------------------
# Uniqueness por evento = média de 1/concurrency no intervalo do evento
# -------------------------------------------------------------------------
def uniqueness_weights(index: pd.Index, t1: pd.Series, normalize: bool = True) -> pd.Series:
    """
    Calcula peso por 'average uniqueness' (AFML):
      w_i = mean_{t in [i, t1_i)} 1 / concurrency_t
    Retorna série de pesos indexada pelos inícios dos eventos.
    """
    idx = pd.Index(index)
    t1 = t1.reindex(idx).dropna()
    conc = _concurrency_series(idx, t1)

    w = pd.Series(0.0, index=t1.index)
    for start, end in t1.items():
        # fatia [start, end) — se 'end' == start, peso=1.0 por convenção
        if start == end:
            w.loc[start] = 1.0
            continue
        if end in conc.index:
            seg = conc.loc[start:end].iloc[:-1]  # exclui o instante do fim
        else:
            seg = conc.loc[start:]
        w.loc[start] = float((1.0 / seg).mean()) if len(seg) else 1.0

    if normalize and w.sum() > 0:
        w = w / w.sum()
    w.name = "w_uniqueness"
    return w


# -------------------------------------------------------------------------
# Time-decay (exponencial simples sobre a ordem temporal)
# -------------------------------------------------------------------------
def time_decay(weights: pd.Series, decay: float = 0.5, normalize: bool = True) -> pd.Series:
    """
    Aplica decaimento exponencial no tempo: mais recente => peso maior.
    'decay' é a razão por passo (0<decay<=1). 1.0 => sem decaimento.

    Implementação: pesos *= decay**rank, onde rank=0 para o mais recente.
    """
    if not (0 < decay <= 1.0):
        raise ValueError("decay deve estar em (0, 1]")
    w = weights.copy().astype(float)

    # Ordenação temporal (antigo->recente), aplicando expoente inverso
    order = np.arange(len(w))
    rank_from_recent = (len(w) - 1) - order
    decay_vec = np.power(decay, rank_from_recent)
    w = w.values * decay_vec
    w = pd.Series(w, index=weights.index, name="w_time_decay")

    if normalize and w.sum() > 0:
        w = w / w.sum()
    return w


# -------------------------------------------------------------------------
# Pipeline de pesos pronto a usar
# -------------------------------------------------------------------------
def build_sample_weights(
    index: pd.Index,
    max_bars: int,
    decay: Optional[float] = 0.5,
    normalize: bool = True,
) -> pd.Series:
    """
    Constrói pesos finais = uniqueness (concurrency) * time-decay (opcional).
    Retorna série alinhada ao 'index' (event starts).
    """
    t1 = get_t1_from_max_bars(index, max_bars=max_bars)
    w_uni = uniqueness_weights(index, t1, normalize=False)
    if decay is not None:
        w_td = time_decay(w_uni, decay=decay, normalize=False)
        w = w_uni * w_td
    else:
        w = w_uni

    if normalize and w.sum() > 0:
        w = w / w.sum()
    w.name = "w_sample"
    return w
