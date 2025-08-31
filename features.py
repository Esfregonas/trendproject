# features.py

"""
Módulo para extrair features financeiras:
  1) Fractional differentiation (mantém memória e torna estacionário);
  2) Volatility-based thresholds (para calibrar pt/sl dinâmicos);
  3) Microstructural features (se tiveres dados de book/ticks);
  4) Entropy-based features.
  5) Structural-break features;
  6) Classic VPIN em volume-bars.
"""

import pandas as pd
import numpy as np

# 1) Fractional Differentiation

def fracdiff(series: pd.Series, d: float, thresh: float = 1e-5) -> pd.Series:
    """
    Retorna uma série fracionalmente diferenciada com parâmetro d
    (0 <= d <= 1), preservando o máximo de memória possível.
    """
    # 1. Calcular coeficientes w_k = (-1)^k * comb(d, k)
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
        k += 1
    w = np.array(w[::-1]).reshape(-1, 1)  # pesos invertidos

    # 2. Aplicar filtro convolucional
    df = pd.Series(series.values)
    out = np.full_like(df, np.nan)
    for i in range(len(w), len(df)):
        window = df.iloc[i - len(w) + 1 : i + 1].values
        out[i] = np.dot(w.T, window)
    out = pd.Series(out, index=series.index)
    return out.dropna()

def add_fracdiff_features(df: pd.DataFrame, d_list: list) -> pd.DataFrame:
    """
    Para cada coluna de preço em df, aplica fracdiff com cada d em d_list
    e concatena ao DataFrame original.
    Exemplo: d_list = [0.2, 0.4, 0.6, 0.8]
    """
    feats = pd.DataFrame(index=df.index)
    for col in ['open','high','low','close']:
        for d in d_list:
            feats[f'{col}_fd{d}'] = fracdiff(df[col], d)
    return feats

# 2) Volatility-based features

def rolling_volatility(series: pd.Series, window: int) -> pd.Series:
    """Desvio-padrão rolante dos retornos percentuais."""
    return series.pct_change().rolling(window).std()


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average True Range rolante para medir volatilidade.
    """
    high_low = df['high'] - df['low']
    high_prev = (df['high'] - df['close'].shift()).abs()
    prev_low = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_prev, prev_low], axis=1).max(axis=1)
    return tr.rolling(window).mean()

# 3) Microstructural features

def bid_ask_spread(ticks: pd.DataFrame) -> pd.Series:
    """
    ticks deve ter colunas ['bid','ask'], index datetime.
    Retorna a série spread = ask - bid.
    """
    spread = ticks['ask'] - ticks['bid']
    return spread

def roll_spread_estimator(prices: pd.Series) -> float:
    """
    Aplica o estimador de Roll (1984) ao array de preços de transação.
    Retorna a medida de spread.
    """
    # covariance entre retornos defasados
    ret = prices.pct_change().dropna()
    cov = ret.autocorr(lag=1)  # cov(ret_t, ret_{t-1})
    return 2 * np.sqrt(-cov)

def amihud_impact(bars: pd.DataFrame) -> pd.Series:
    """
    bars: DF com 'close' e 'volume'. Retorna série I = |return| / volume.
    """
    ret = bars['close'].pct_change().abs()
    impact = ret / bars['volume']
    return impact

# 4) Entropy-based features

def shannon_entropy(series: pd.Series, bins: int = 10) -> float:
    probs, _ = np.histogram(series, bins=bins, density=True)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def lz_complexity(series: pd.Series) -> float:
    s = ''.join(np.where(series.diff().fillna(0) >= 0, '1', '0'))
    i, comps = 0, set()
    while i < len(s):
        for j in range(i + 1, len(s) + 1):
            if s[i:j] not in comps:
                comps.add(s[i:j])
                i = j
                break
    return len(comps) / len(s) if len(s) > 0 else 0.0

# 5) Structural-break features (CUSUM)

def cusum_efp(series: pd.Series, threshold: float = 1.0) -> pd.Series:
    s = (series - series.mean()) / series.std()
    cusum_pos = s.cumsum().clip(lower=0)
    cusum_neg = (-s).cumsum().clip(lower=0)
    return ((cusum_pos > threshold) | (cusum_neg > threshold)).astype(int)

# 6) Classic VPIN em volume-bars

def vpin(bars: pd.DataFrame, bucket_size: int) -> pd.Series:
    signs = np.sign(bars['close'].diff().fillna(0))
    signed_vol = signs * bars['volume']
    num = signed_vol.abs().rolling(window=bucket_size).sum()
    den = bars['volume'].rolling(window=bucket_size).sum()
    return num / den.dropna()


# Mais avançado:
# def vpin(ticks, bucket_volume): ...
def tick_rule(prices: pd.Series) -> pd.Series:
    """
    Classifica cada transação como +1 (uptick) ou -1 (downtick)
    com base na variação de preço.
    """
    delta = prices.diff().fillna(0)
    # uptick se Δ≥0, downtick se Δ<0
    return np.where(delta >= 0, 1, -1)

def rolling_order_imbalance(ticks: pd.DataFrame, window: int = 200) -> pd.Series:
    """
    Proxy de VPIN: numa janela de `window` ticks calcula
      imbalance = sum(|signed_vol|) / sum(volume)

    signed_vol = tick_rule(price) * size
    """
    # 1) Sinais +1/−1
    signs = tick_rule(ticks['price'])
    # 2) Volume assinado
    signed_vol = signs * ticks['size']
    # 3) Rolling sums
    num = pd.Series(signed_vol).abs().rolling(window).sum()
    den = ticks['size'].rolling(window).sum()
    return num / den

# def hasbrouck_lambda(book_df): ...