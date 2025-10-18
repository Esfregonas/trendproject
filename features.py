# features.py
"""
Conjunto de *features* utilizadas no projeto (versões simples e didáticas)
===========================================================================

Blocos disponíveis
------------------
1) Volatilidade / ATR
   - rolling_volatility(series, window)
   - atr(df, window)

2) Entropia e Complexidade
   - shannon_entropy(window_values, bins=10)
   - lz_complexity(window_values)
   As duas são pensadas para uso com rolling.apply / progress_apply.

3) Quebras estruturais (CUSUM)
   - cusum_efp(close, threshold)

4) Microestrutura (proxies simples)
   - roll_spread_estimator(close)
   - amihud_impact(df, window)

5) VPIN (versão pedagógica com *buckets* de volume)
   - vpin(vb, bucket_size=10)
   NOTE: inclui **tqdm** para barra de progresso (%).

6) Fracdiff (versão compacta, suficiente para prototipagem)
   - add_fracdiff_features(df_ohlc, d_list)

Notas
-----
- Estas implementações privilegiam clareza; quando (e se) for necessário,
  podemos trocar por versões vetorizadas/numba para acelerar mais.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# tqdm com fallback (mostra barras de progresso; se não houver, segue sem barra)
# ------------------------------------------------------------------------------
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


# ==============================================================================
# 1) Volatilidade / ATR
# ==============================================================================

def rolling_volatility(series: pd.Series, window: int = 10) -> pd.Series:
    """
    Volatilidade (desvio-padrão) anualizada de *log-returns* numa janela.
    """
    s = series.astype(float)
    ret = np.log(s).diff()
    vol = ret.rolling(window).std() * np.sqrt(252)  # anualiza
    vol.name = f"vol{window}"
    return vol


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    *Average True Range* (ATR) clássico.
    """
    high = df["high"].astype(float)
    low  = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    out = tr.rolling(window).mean()
    out.name = f"atr{window}"
    return out


# ==============================================================================
# 2) Entropia e Complexidade
#   (pensadas para rolling.apply / progress_apply)
# ==============================================================================

def shannon_entropy(x: Iterable[float], bins: int = 10) -> float:
    """
    Entropia de Shannon de uma janela de valores (histogram-based).
    """
    arr = np.asarray(list(x), dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan

    # utiliza returns para reduzir escala
    r = np.diff(arr)
    if r.size == 0:
        return np.nan

    hist, _ = np.histogram(r[~np.isnan(r)], bins=bins, density=True)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def lz_complexity(x: Iterable[float]) -> float:
    """
    Complexidade Lempel-Ziv (LZ76) via binarização por mediana.
    Implementação sucinta para janelas curtas (rolling windows).
    """
    arr = np.asarray(list(x), dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan

    # Binariza por comparação à mediana (evita escala)
    med = np.nanmedian(arr)
    bits = (arr > med).astype(np.int8)

    # LZ76: conta o nº de *novas* sub-strings enquanto percorre a sequência
    s = "".join("1" if b else "0" for b in bits)
    i, k, l = 0, 1, 1
    c = 1  # complexity counter
    n = len(s)

    while True:
        if i + k > n - 1 or l + k > n:
            c += 1
            break
        if s[i:i + k] == s[l:l + k]:
            k += 1
        else:
            if k > 1:
                i += 1
            else:
                c += 1
                l += 1
                i = 0
            k = 1
            if l == i:
                l += 1
            if l >= n:
                break
    # normalização simples pela escala do log (opcional)
    return float(c / (n / np.log2(max(n, 2))))


# ==============================================================================
# 3) Quebras estruturais (CUSUM Filter simples)
# ==============================================================================

def cusum_efp(close: pd.Series, threshold: float = 2.5) -> pd.Series:
    """
    Sinal (0/1) de evento quando o CUSUM padronizado excede 'threshold'.
    """
    x = close.astype(float)
    r = x.diff().fillna(0.0)
    mu = r.rolling(50).mean()
    sd = r.rolling(50).std().replace(0, np.nan)

    z = (r - mu) / sd
    z = z.fillna(0.0)

    s_pos = 0.0
    s_neg = 0.0
    flags = np.zeros(len(z), dtype=np.int8)

    for i, zi in enumerate(z.values):
        s_pos = max(0.0, s_pos + zi)
        s_neg = min(0.0, s_neg + zi)
        if s_pos > threshold or s_neg < -threshold:
            flags[i] = 1
            s_pos = 0.0
            s_neg = 0.0

    out = pd.Series(flags.astype(float), index=close.index, name="cusum_flag")
    return out


# ==============================================================================
# 4) Microestrutura (proxies simples)
# ==============================================================================

def roll_spread_estimator(price: pd.Series) -> float:
    """
    Estimador de Roll (spread) a partir da autocovariância dos retornos.
    (valor único — usa a série inteira)
    """
    p = price.astype(float)
    r = p.diff().dropna()
    if len(r) < 3:
        return np.nan
    cov = r[:-1].cov(r[1:])
    if cov >= 0 or np.isnan(cov):
        return np.nan
    spread = 2.0 * np.sqrt(-cov)
    return float(spread)


def amihud_impact(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Amihud illiquidity: E[ |ret| / dollar_volume ] numa janela.
    (usa close e volume; se tiver preço médio, melhor)
    """
    close = df["close"].astype(float)
    vol   = df["volume"].astype(float)
    ret = close.pct_change().abs()
    dollar = (close * vol).replace(0, np.nan)
    illiq = (ret / dollar).rolling(window).mean()
    illiq.name = f"amihud{window}"
    return illiq


# ==============================================================================
# 5) VPIN (versão pedagógica com buckets de volume) com barra de progresso
# ==============================================================================

@dataclass
class _Bucket:
    buy: float = 0.0
    sell: float = 0.0
    vol: float = 0.0

def vpin(vb: pd.DataFrame, bucket_size: int = 10) -> pd.Series:
    """
    VPIN simplificado com buckets de volume (igual-volume).

    Ideia:
      - Assumimos *sinal* da ordem pelo sinal do retorno (proxy).
      - Acumulamos volume (buy/sell) até atingir um *target_volume*.
      - Para cada bucket: VPIN_k = |buy - sell| / (buy + sell)
      - Devolvemos uma série alinhada ao índice do vb (preenche NaN fora do fecho do bucket).

    Parâmetros
    ----------
    vb : DataFrame de volume bars (colunas: ['close','volume', ...]).
    bucket_size : int
        Número médio de barras por bucket (controla o *target_volume*).
        target_volume = mean(vb['volume']) * bucket_size

    Nota:
      - Esta versão é pedagógica e usa um loop com **tqdm** para visibilidade.
      - Para produção, podemos substituir por uma versão vetorizada.
    """
    close = vb["close"].astype(float)
    vol   = vb["volume"].astype(float)
    idx   = vb.index

    if len(vb) < bucket_size:
        return pd.Series(index=idx, dtype=float, name="vpin")

    target_vol = float(vol.mean() * bucket_size)  # volume alvo por bucket
    if target_vol <= 0:
        return pd.Series(index=idx, dtype=float, name="vpin")

    # sinal por variação de preço (proxy de agressão)
    sign = np.sign(close.diff().fillna(0.0).values)  # {-1,0,+1}

    out = np.full(len(vb), np.nan, dtype=float)
    bucket = _Bucket()

    for i in tqdm(range(len(vb)), desc="VPIN", mininterval=0.5):
        v = float(vol.iloc[i])
        s = float(sign[i])

        buy = v if s > 0 else 0.0
        sell = v if s < 0 else 0.0

        bucket.buy  += buy
        bucket.sell += sell
        bucket.vol  += v

        if bucket.vol >= target_vol:  # fecha bucket
            imbalance = abs(bucket.buy - bucket.sell) / max(bucket.vol, 1e-12)
            out[i] = imbalance  # marca no fecho do bucket

            # reinicia (carrega excesso para o próximo)
            overflow = bucket.vol - target_vol
            # atribui overflow como neutro (poderíamos carregar *proporcionalmente*)
            bucket = _Bucket(vol=overflow)

    return pd.Series(out, index=idx, name=f"vpin{bucket_size}")


# ==============================================================================
# 6) Fracdiff (versão compacta)
# ==============================================================================

def _get_fracdiff_weights(d: float, size: int) -> np.ndarray:
    """
    Pesos para diferenciação fracionária (série curta).
    """
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w, dtype=float)


def _fracdiff_one(series: pd.Series, d: float, window: int = 100) -> pd.Series:
    """
    Diferenciação fracionária simples com janela limitada (para manter estável).
    """
    s = series.astype(float).copy()
    w = _get_fracdiff_weights(d, window)[::-1]  # mais antigo -> mais recente
    out = pd.Series(index=s.index, dtype=float)
    vals = s.values

    for i in range(window, len(s)):
        out.iloc[i] = np.dot(w, vals[i - window:i])
    return out


def add_fracdiff_features(df_ohlc: pd.DataFrame, d_list: List[float], window: int = 100) -> pd.DataFrame:
    """
    Aplica fracdiff a colunas OHLC e concatena em um DataFrame de features.
    """
    cols = ["open", "high", "low", "close"]
    base = df_ohlc[cols].astype(float).copy()

    out = {}
    for d in d_list:
        for c in cols:
            out[f"{c}_fd{d}"] = _fracdiff_one(base[c], d=d, window=window)

    return pd.DataFrame(out, index=base.index)
