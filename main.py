# main.py

from data_fetcher import connect_ib, fetch_data
from ib_insync    import Stock, util
from bars         import volume_bars
from labeler      import apply_triple_barrier, meta_labeling
from features     import (
    add_fracdiff_features,
    rolling_order_imbalance,
    roll_spread_estimator,
    amihud_impact,
    rolling_volatility,
    atr,
    shannon_entropy,
    lz_complexity,
    cusum_efp,
    vpin
)
import pandas as pd

def main():
    # 0) Conexão e dados brutos
    ib       = connect_ib()
    contract = Stock('AAPL', 'SMART', 'USD')
    df       = fetch_data(ib, contract,
                          duration='180 D',
                          bar_size='1 day')

    # 1) Threshold dinâmico de volume: 2× média diária
    avg_daily_vol = df['volume'].mean()
    threshold     = 2 * avg_daily_vol
    print(f"Média diária de volume: {avg_daily_vol:.0f}, threshold: {threshold:.0f}\n")

    # 2) Volume-bars usando esse threshold
    vb = volume_bars(df, volume_threshold=threshold)
    print(f"Barras diárias: {df.shape}, volume-bars: {vb.shape}\n")

    # 3) Novas features recomendadas pelo livro
    # 3.1) Volatility-based
    vol10 = rolling_volatility(vb['close'], window=10)
    atr14 = atr(vb, window=14)

    # 3.2) Entropy-based (rolling)
    ent20 = vb['close'].rolling(20).apply(shannon_entropy, raw=False)
    lz20  = vb['close'].rolling(20).apply(lz_complexity,   raw=False)

    # 3.3) Structural-break (CUSUM)
    sb_flag = cusum_efp(vb['close'], threshold=2.5)

    # 3.4) Classic VPIN em volume-bars
    vpin10 = vpin(vb, bucket_size=10)

    # Conferir head dessas features
    print("vol10 head:\n",   vol10.head(),   "\n")
    print("atr14 head:\n",   atr14.head(),   "\n")
    print("ent20 head:\n",   ent20.head(),   "\n")
    print("lz20 head:\n",    lz20.head(),    "\n")
    print("sb_flag head:\n", sb_flag.head(), "\n")
    print("vpin10 head:\n",  vpin10.head(),  "\n")

    return  # interrompe aqui

    # 4) Labels e meta-labels
    labels = apply_triple_barrier(vb, pt=0.005, sl=0.005, max_bars=10)
    meta   = meta_labeling(labels)
    print("Labels primários:\n", labels.value_counts(), "\n")
    print("Meta-labels:\n",    meta.value_counts(),   "\n")

    # 5) Fracdiff features
    d_list = [0.2, 0.4, 0.6, 0.8]
    feats  = add_fracdiff_features(vb[['open','high','low','close']], d_list)
    feats, meta = feats.dropna(), meta.reindex(feats.dropna().index)
    print("Fracdiff features shape:", feats.shape, "\n")

    # 6) Microstructural features via TickByTickLast
    ticks    = ib.reqTickByTickAllLast(contract, startDateTime='', numberOfTicks=1000)
    ticks_df = util.df(ticks)
    ticks_df.index = pd.to_datetime(ticks_df['time'], unit='s', utc=True)

    # 7) Order-flow imbalance (proxy VPIN)
    imbalance = rolling_order_imbalance(ticks_df, window=200)
    print("Order-flow imbalance head:\n", imbalance.head(), "\n")

    # 8) Calcular restantes micro-features a partir de preços de transação
    roll   = roll_spread_estimator(ticks_df['price'])
    impact = amihud_impact(vb)
    print("Roll spread (proxy):", roll, "\n")
    print("Amihud impact head:\n", impact.head(), "\n")

    # 9) Adicionar imbalance ao DataFrame de features
    imbalance = imbalance.reindex(feats.index)
    feats['order_imbalance'] = imbalance
    print("Features agora incluem:", feats.columns.tolist(), "\n")

    # 10) Concatenar todas as features num único DataFrame
    #    (inclui: fracdiff, micro, vol, entropy, CUSUM e VPIN)
    X = pd.concat(
        [feats, vol10, atr14, ent20, lz20, sb_flag, vpin10],
        axis=1
    ).dropna()

    # 11) Alinhar X (features) com meta-labels
    #     join='inner' garante que só fiquem os índices que
    #     existem em ambos X e meta
    X, meta = X.align(meta, join='inner', axis=0)

    # 12) Conferir o resultado final
    print("Shape final de X:", X.shape)
    print("Colunas finais de X:", X.columns.tolist())
    print("y (meta) distribuições:\n", meta.value_counts())

if __name__ == "__main__":
    main()
