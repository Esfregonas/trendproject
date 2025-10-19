# bars.py
"""
Funções para converter dados brutos em volume-bars e dollar-bars.

Este módulo fornece:
- volume_bars(df, volume_threshold)
- dollar_bars(df, dollar_threshold, price_mode)
- compute_threshold(df, kind, threshold, multiplier)  -> valor do limiar + texto explicativo
- build_bars(df, kind, threshold, multiplier, **kwargs) -> dispatcher que escolhe o tipo de barra

Assume DataFrame temporal com colunas: ['open','high','low','close','volume'].
"""

import pandas as pd
from typing import Tuple


# ============================================================================
#                                Volume-Bars
# ============================================================================

def volume_bars(
    df: pd.DataFrame,
    volume_threshold: float
) -> pd.DataFrame:
    """
    Constrói volume-bars a partir de um DataFrame de barras temporais
    com colunas ['open','high','low','close','volume'].
    Cada barra resultante terá volume acumulado ≥ volume_threshold.

    Notas:
    - O timestamp de cada barra é o 'end_time' da última barra temporal agregada.
    - Se não se atingir o limiar no final, a barra incompleta é descartada.
    """
    bars = []
    cum_vol = 0.0
    o = h = l = c = None

    for ts, row in df.iterrows():
        v = float(row["volume"])

        # início da barra
        if o is None:
            o = row["open"]
            h = row["high"]
            l = row["low"]

        # atualizar extremos
        h = max(h, row["high"])
        l = min(l, row["low"])

        cum_vol += v
        c = row["close"]

        # atingir o limiar -> fecha barra
        if cum_vol >= volume_threshold:
            bars.append(
                {
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": cum_vol,
                    "end_time": ts,  # guardamos timestamp final
                }
            )
            # reset
            cum_vol = 0.0
            o = h = l = c = None

    if not bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    out = (
        pd.DataFrame(bars)
        .set_index("end_time")[["open", "high", "low", "close", "volume"]]
    )
    return out


# ============================================================================
#                                Dollar-Bars
# ============================================================================

def _price_proxy(row: pd.Series, mode: str) -> float:
    """
    Proxy de preço para dollar-bars.
    mode:
      - 'close': usa close
      - 'hlc3' : (high+low+close)/3
      - 'ohlc4': (open+high+low+close)/4
    """
    if mode == "hlc3":
        return float((row["high"] + row["low"] + row["close"]) / 3.0)
    if mode == "ohlc4":
        return float((row["open"] + row["high"] + row["low"] + row["close"]) / 4.0)
    return float(row["close"])


def dollar_bars(
    df: pd.DataFrame,
    dollar_threshold: float,
    price_mode: str = "close",
) -> pd.DataFrame:
    """
    Constrói dollar-bars: cada barra acumula *valor negociado*
    (preço_proxy × volume) ≥ dollar_threshold.

    price_mode controla o proxy de preço usado:
      - 'close' (default), 'hlc3', 'ohlc4'.
    """
    bars = []
    cum_val = 0.0
    cum_vol = 0.0
    o = h = l = c = None

    for ts, row in df.iterrows():
        p = _price_proxy(row, price_mode)
        v = float(row["volume"])
        dv = p * v  # dollar value desta barra temporal

        # início da barra
        if o is None:
            o = row["open"]
            h = row["high"]
            l = row["low"]

        # atualizar extremos
        h = max(h, row["high"])
        l = min(l, row["low"])

        cum_val += dv
        cum_vol += v
        c = row["close"]

        # atingir o limiar -> fecha barra
        if cum_val >= dollar_threshold:
            bars.append(
                {
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": cum_vol,          # mantém volume acumulado
                    "dollar_value": cum_val,    # e o valor em dólares
                    "end_time": ts,
                }
            )
            # reset
            cum_val = 0.0
            cum_vol = 0.0
            o = h = l = c = None

    if not bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "dollar_value"])

    out = (
        pd.DataFrame(bars)
        .set_index("end_time")[["open", "high", "low", "close", "volume", "dollar_value"]]
    )
    return out


# ============================================================================
#                      Helpers para threshold & dispatcher
# ============================================================================

def compute_threshold(
    df: pd.DataFrame,
    kind: str = "volume",
    threshold: float | None = None,
    multiplier: float = 1.0,
    price_mode: str = "close",
) -> Tuple[float, str]:
    """
    Calcula um threshold para o tipo de barra escolhido.

    Se 'threshold' for fornecido, devolve-o (com fonte='config').
    Caso contrário, devolve média × 'multiplier' (com fonte='dinâmico').

    Para:
      - kind == 'volume' -> mean(volume) × multiplier
      - kind == 'dollar' -> mean(price_proxy*volume) × multiplier
    """
    if threshold is not None:
        return float(threshold), "config"

    if kind == "dollar":
        # média do valor negociado por barra temporal
        pv = df.apply(lambda r: _price_proxy(r, price_mode) * float(r["volume"]), axis=1)
        base = float(pv.mean())
        return max(1.0, multiplier * base), f"{multiplier:.2f}×média($)"

    # default: volume
    base = float(df["volume"].mean())
    return max(1.0, multiplier * base), f"{multiplier:.2f}×média(vol)"


def build_bars(
    df: pd.DataFrame,
    kind: str = "volume",
    threshold: float | None = None,
    multiplier: float = 1.0,
    price_mode: str = "close",
):
    """
    Dispatcher conveniente: calcula o threshold quando necessário e
    constrói o tipo de barra pedido. Retorna (bars_df, threshold, fonte_str).

    kind ∈ {'volume','dollar'}
    """
    kind = (kind or "volume").lower()
    thr, source = compute_threshold(df, kind=kind, threshold=threshold, multiplier=multiplier, price_mode=price_mode)

    if kind == "dollar":
        out = dollar_bars(df, dollar_threshold=thr, price_mode=price_mode)
    else:
        out = volume_bars(df, volume_threshold=thr)

    return out, thr, source
