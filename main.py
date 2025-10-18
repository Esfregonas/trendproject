# main.py
"""
Pipeline com logger + barras de progresso
-----------------------------------------
1) Lê config (config.yaml) e liga ao TWS/Gateway da IB.
2) Busca OHLCV (time bars) e converte para volume bars.
3) Extrai features (rápidas + opcionais pesadas, com barra de progresso nas rolling).
4) Cria labels (Triple-Barrier) + meta-labels.
5) Constrói X/y, grava em data/ e corre baseline (RF + Purged K-Fold).

Usamos:
- log(...) para mostrar cada passo com hora (flush=True).
- tqdm.pandas() para percentagem nas rolling (entropia/LZ) com .progress_apply().
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from time import perf_counter as now
from datetime import datetime

# -------------------- logger simples com hora --------------------
def log(msg: str) -> None:
    """Imprime mensagem com timestamp (sem buffer)."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# -------------------- tqdm para progresso nas rolling ------------
try:
    from tqdm import tqdm
    tqdm.pandas()  # habilita Series/DataFrame.progress_apply()
except Exception:
    # se tqdm não existir, .progress_apply() vira .apply() silenciosamente
    class _DummyTQDM:
        @staticmethod
        def pandas(): ...
    tqdm = _DummyTQDM()  # type: ignore

# -------------------- parâmetros (ajusta aqui) -------------------
DEBUG_FAST = False           # True => só features rápidas (vol10, atr14, CUSUM)
TB_PT = 0.005                # take-profit  (ex.: 0.5%)
TB_SL = 0.005                # stop-loss    (ex.: 0.5%)
TB_MAX_BARS = 10             # janela máxima (vertical barrier)
VOLUME_THRESHOLD_MULT = 1.0  # 1.0× média de volume (ajusta se necessário)

# Quando DEBUG_FAST=False podes ligar/desligar features pesadas:
FEATURE_ENTROPY = True       # rolling Shannon entropy (com barra %)
FEATURE_LZ       = True      # rolling Lempel-Ziv complexity (com barra %)
FEATURE_VPIN     = True      # VPIN clássico

# -------------------- módulos do projeto -------------------------
from data_fetcher import load_cfg, connect_ib, make_contract, fetch_data
from bars        import volume_bars
from features    import (
    rolling_volatility, atr, shannon_entropy, lz_complexity, cusum_efp, vpin
)
from labeler     import apply_triple_barrier, meta_labeling
from model       import run_baseline


def main():
    log(">> main() começou")

    # 0) Config + ligação IB
    cfg = load_cfg("config.yaml")
    ib = connect_ib(
        host=cfg["ib"]["host"],
        port=cfg["ib"]["port"],
        client_id=cfg["ib"]["clientId"],
    )

    try:
        # 1) Contrato e OHLCV (time bars)
        contract = make_contract(cfg["data"]["contract"])
        h = cfg["data"]["history"]

        log("A pedir dados históricos à IB...")
        df = fetch_data(
            ib=ib,
            contract=contract,
            duration=h["durationStr"],        # ex.: "10 D"
            bar_size=h["barSizeSetting"],     # ex.: "1 min"
            what_to_show=h.get("whatToShow"), # None => TRADES (stocks) / MIDPOINT (FX)
            use_rth=h.get("useRTH", True),
        )
        log(f"Dados brutos (OHLCV) recebidos: {df.shape}")
        if df.empty:
            log("Sem dados — ajusta 'durationStr'/'barSizeSetting' no config.yaml.")
            return
        print(df.head(), "\n")  # amostra (ok manter como print normal)

        # 2) Volume bars
        avg_vol   = df["volume"].mean()
        threshold = VOLUME_THRESHOLD_MULT * avg_vol
        log(f"Média de volume: {avg_vol:.0f} | threshold usado: {threshold:.0f}")

        log("A construir volume bars...")
        vb = volume_bars(df, volume_threshold=threshold)
        log(f"Volume bars construídas: {vb.shape}")
        if vb.empty:
            log("Volume bars vazias — baixa o threshold ou aumenta a janela no YAML.")
            return

        # 3) Features
        log("A calcular features...")
        t0 = now()
        features_dict: dict[str, pd.Series] = {}

        # 3.1) rápidas
        features_dict["vol10"]   = rolling_volatility(vb["close"], window=10)
        features_dict["atr14"]   = atr(vb, window=14)
        features_dict["sb_flag"] = cusum_efp(vb["close"], threshold=2.5).astype(float)

        # 3.2) pesadas (com barra de progresso nas rolling)
        if not DEBUG_FAST:
            if FEATURE_ENTROPY:
                # tqdm.pandas() permite .progress_apply() -> mostra % no terminal
                features_dict["ent20"] = (
                    vb["close"].rolling(20)
                    .progress_apply(shannon_entropy, raw=False)
                )
            if FEATURE_LZ:
                features_dict["lz20"] = (
                    vb["close"].rolling(20)
                    .progress_apply(lz_complexity, raw=False)
                )
            if FEATURE_VPIN:
                # Se a tua função vpin() tiver loops, podes pôr tqdm lá dentro (em features.py)
                features_dict["vpin10"] = vpin(vb, bucket_size=10)

        log(f"Features calculadas em {now()-t0:.3f}s")
        # heads de inspeção (podem ter NaN no início por causa das janelas)
        for name, ser in features_dict.items():
            print(f"{name} head:\n", ser.head(), "\n")

        # 4) Labels + Meta-labels
        log("A calcular labels (Triple-Barrier)...")
        labels = apply_triple_barrier(vb, pt=TB_PT, sl=TB_SL, max_bars=TB_MAX_BARS)
        log("Labels calculadas. A criar meta-labels...")
        meta = meta_labeling(labels)
        log(f"Labels primários: {labels.value_counts().to_dict()} | Meta: {meta.value_counts().to_dict()}")

        # 5) X/y
        log("A montar X e alinhar com y...")
        feats = pd.concat(features_dict, axis=1)
        X = feats.dropna()
        y = meta.reindex(X.index).dropna()
        idx = X.index.intersection(y.index)
        X, y = X.loc[idx], y.loc[idx]
        log(f"Shape X: {X.shape} | Distribuição y: {y.value_counts().to_dict()}")
        # verificação explícita das colunas
        log(f"Colunas em X: {list(X.columns)} | Amostras em X: {len(X)}")

        # 6) Guardar
        log("A guardar X/y em disco...")
        out = Path("data"); out.mkdir(exist_ok=True)
        X.to_parquet(out / "X.parquet"); X.to_csv(out / "X.csv")
        y.to_frame("y").to_parquet(out / "y.parquet"); y.to_frame("y").to_csv(out / "y.csv")
        log(f"Artefactos gravados em: {out.resolve()}")

        # 7) Baseline
        log("=== Baseline: RandomForest + Purged K-Fold ===")
        _ = run_baseline(
            X, y,
            n_splits=5,
            purge=5,
            embargo=0.01,
            n_estimators=300,
            max_depth=None,
        )
        log("Baseline concluído.")

    finally:
        log("A encerrar ligação à IB...")
        ib.disconnect()
        log("Ligação IB encerrada. Fim do script.")


# --- entry point ---
if __name__ == "__main__":
    main()
