# bars.py
"""
Funções para converter dados brutos em volume‑bars e dollar‑bars.
"""

"""
Módulo para converter barras temporais (OHLCV) em volume‑bars,
onde cada barra acumula um volume mínimo configurável.
"""

import pandas as pd

def volume_bars(
    df: pd.DataFrame,
    volume_threshold: float
) -> pd.DataFrame:
    """
    Constrói volume‑bars a partir de um DataFrame de barras
    temporais com colunas ['open','high','low','close','volume'].
    Cada barra resultante terá volume ≥ volume_threshold.
    """
    bars = []
    cum_vol = 0.0
    o = h = l = c = None

    for ts, row in df.iterrows():
        v = float(row['volume'])
        # início de barra
        if o is None:
            o = row['open']
            h = row['high']
            l = row['low']

        # atualizar high/low
        h = max(h, row['high'])
        l = min(l, row['low'])

        cum_vol += v
        c = row['close']

        # se atingir o volume mínimo, fechar barra
        if cum_vol >= volume_threshold:
            bars.append({
                'open': o,
                'high': h,
                'low': l,
                'close': c,
                'volume': cum_vol,
                'end_time': ts   # guardamos o timestamp aqui
            })
            # reset para próxima barra
            cum_vol = 0.0
            o = h = l = c = None

    # se não há barras completas, devolve vazio
    if not bars:
        return pd.DataFrame(columns=['open','high','low','close','volume'])

    # construir DataFrame e indexar por end_time
    out = pd.DataFrame(bars).set_index('end_time')[['open','high','low','close','volume']]
    return out