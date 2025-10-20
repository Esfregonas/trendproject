# main.py
"""
Pipeline principal (didático):
1) Lê config.yaml e liga ao TWS/Gateway (IB).
2) Busca OHLCV históricos conforme a config.
3) Constrói volume-bars OU dollar-bars conforme 'bars.kind' na config.
4) Extrai features (volatilidade, ATR, entropia, LZ, CUSUM, VPIN).
5) Calcula labels (triple-barrier) e meta-labels.
6) Monta matriz X e vetor y, grava em data/.
7) Lê data/rf_best.json (se existir) e corre baseline com esses parâmetros
   (com calibração isotónica e métricas PR-AUC / F1@thr* no model.py).
8) (NOVO) Constrói pesos AFML (uniqueness + time-decay por half-life) e passa-os ao treino.

Nota: manter TWS aberto e API ativa (porta 7497).
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from datetime import datetime

import pandas as pd

# --- helpers do teu módulo data_fetcher (com explicações in-line) ---
from data_fetcher import (
    load_cfg,      # lê o config.yaml
    connect_ib,    # abre ligação ao TWS/Gateway
    make_contract, # fabrica Stock/FX/Future a partir do dict de config
    fetch_data,    # pede OHLCV e devolve DataFrame
)

# --- módulos do projeto ---
from bars import build_bars
from features import (
    rolling_volatility,
    atr,
    shannon_entropy,
    lz_complexity,
    cusum_efp,
    vpin,
)
from labeler import apply_triple_barrier, meta_labeling
from weights import build_sample_weights  # <- AFML weights (uniqueness + time-decay)
from model import run_baseline


# ===================== opções rápidas (podes ajustar) ===================== #
BARS_THRESHOLD_MULT = 1.00  # 1.0 = 100% da média (quando não há 'threshold' absoluto)
# ========================================================================= #


def log(msg: str) -> None:
    """Print com timestamp (simples para acompanhar o progresso)."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def main():
    t0 = perf_counter()
    log(">> main() começou")

    # 0) Ler configuração e abrir ligação à IB
    cfg = load_cfg("config.yaml")
    ib = connect_ib(
        host=cfg["ib"]["host"],
        port=cfg["ib"]["port"],
        client_id=cfg["ib"]["clientId"],
    )

    try:
        # 1) Contrato e dados brutos (evita hardcode: lê tudo da config)
        contract = make_contract(cfg["data"]["contract"])
        h        = cfg["data"]["history"]

        log("A pedir dados históricos à IB…")
        df = fetch_data(
            ib=ib,
            contract=contract,
            duration=h["durationStr"],            # ex.: "10 D"
            bar_size=h["barSizeSetting"],         # ex.: "1 min"
            what_to_show=h.get("whatToShow"),
            use_rth=h.get("useRTH", True),
        )
        print(f"\nDados brutos (OHLCV) recebidos: {df.shape}")
        print(df.head(), "\n")
        if df.empty:
            log("Sem dados — verifica duration/barSize na config.yaml.")
            return

        # 2) Tipo de barra e threshold (absoluto ou dinâmico=média×multiplier)
        bars_cfg = (cfg.get("bars", {}) or {})
        kind     = str(bars_cfg.get("kind", "volume")).lower()          # 'volume' ou 'dollar'
        thr_cfg  = bars_cfg.get("threshold", None)
        price_md = str(bars_cfg.get("price_mode", "close")).lower()

        # 3) Construir barras
        log(f"A construir {kind}-bars…")
        bars_df, thr_value, thr_source = build_bars(
            df,
            kind=kind,
            threshold=float(thr_cfg) if thr_cfg is not None else None,
            multiplier=float(bars_cfg.get("multiplier", BARS_THRESHOLD_MULT)),
            price_mode=price_md,
        )
        print(f"Barras originais: {df.shape} | {kind}-bars: {bars_df.shape}")
        print(f"Threshold usado: {thr_value:.0f} (fonte: {thr_source})\n")
        if bars_df.empty:
            log(f"{kind}-bars vazias — ajusta 'threshold' ou 'multiplier' na config.")
            return

        # 4) Features (primeiro conjunto)
        log("A calcular features…")
        t_feats = perf_counter()

        vol10   = rolling_volatility(bars_df["close"], window=10)       # volatilidade
        atr14   = atr(bars_df, window=14)                               # ATR clássico
        ent20   = bars_df["close"].rolling(20).apply(shannon_entropy, False)
        lz20    = bars_df["close"].rolling(20).apply(lz_complexity,   False)
        sb_flag = cusum_efp(bars_df["close"], threshold=2.5)            # quebras
        vpin10  = vpin(bars_df, bucket_size=10)                         # fluxo ordens (proxy)

        print(f"Features calculadas em {perf_counter()-t_feats:.3f}s\n")
        print("vol10 head:\n",   vol10.head(),   "\n")
        print("atr14 head:\n",   atr14.head(),   "\n")
        print("sb_flag head:\n", sb_flag.head(), "\n")
        print("ent20 head:\n",   ent20.head(),   "\n")
        print("lz20  head:\n",   lz20.head(),    "\n")
        print("vpin10 head:\n",  vpin10.head(),  "\n")

        # 5) Labels e meta-labels
        log("A calcular labels (Triple-Barrier)…")
        MAX_BARS = 10
        labels = apply_triple_barrier(bars_df, pt=0.005, sl=0.005, max_bars=MAX_BARS)
        log("Labels calculadas. A criar meta-labels…")
        meta = meta_labeling(labels)
        print("Labels primários (contagem):", labels.value_counts().to_dict())
        print("Meta-labels (contagem):",      meta.value_counts().to_dict(), "\n")

        # 6) Montar matriz de features X
        feats = pd.concat(
            {
                "vol10":   vol10,
                "atr14":   atr14,
                "sb_flag": sb_flag.astype(float),  # 0/1 -> float
                "ent20":   ent20,
                "lz20":    lz20,
                "vpin10":  vpin10,
            },
            axis=1,
        )

        # Limpar NaNs e alinhar com y (meta)
        X = feats.dropna()
        y = meta.reindex(X.index).dropna()

        # Interseção defensiva (se por acaso houver desalinhamento residual)
        idx = X.index.intersection(y.index)
        X, y = X.loc[idx], y.loc[idx]
        print(f"Shape X depois do dropna/alinhamento: {X.shape}")
        print("Distribuição y (meta):", y.value_counts().to_dict(), "\n")

        # 6.1) (NOVO) Pesos AFML (uniqueness + time-decay), alinhados a X/y
        #      AFML: half-life em nº de eventos -> decay = 0.5 ** (1/HL)
        w_cfg = (cfg.get("weights") or {})
        use_w = w_cfg.get("enabled", True)  # por defeito ATIVO

        def hl_to_decay(hl: float | None) -> float:
            if hl is None:
                return 1.0  # sem time-decay
            hl = float(hl)
            if hl <= 0:
                return 1.0
            return float(0.5 ** (1.0 / hl))

        w = None
        if use_w:
            hl = w_cfg.get("half_life_events", None)          # HL em nº de eventos (ou None)
            decay = hl_to_decay(hl)
            max_b = int(w_cfg.get("max_bars", MAX_BARS))      # idealmente igual ao triple-barrier
            w = build_sample_weights(X.index, max_bars=max_b, decay=decay)
            tag = f"HL={hl}" if hl is not None else "no-decay"
            print(f"Pesos AFML: enabled=True | {tag} | decay={decay:.6f} | max_bars={max_b}")
            print("w.head():\n", w.head(), "\n")
        else:
            print("Pesos AFML desativados (weights.enabled=false no config).")

        # 7) Guardar artefactos (X, y)
        out = Path("data"); out.mkdir(exist_ok=True)
        X.to_parquet(out / "X.parquet");        X.to_csv(out / "X.csv")
        y.to_frame("y").to_parquet(out / "y.parquet"); y.to_frame("y").to_csv(out / "y.csv")
        print(f"Artefactos gravados em: {out.resolve()}")

        # 8) Ler hiperparâmetros do tuning (se existirem) e correr baseline
        rf_kwargs: dict = {}
        best_path = out / "rf_best.json"
        if best_path.exists():
            with open(best_path) as f:
                rf_kwargs = json.load(f).get("best_params", {}) or {}
            print("\n[INFO] A usar hiperparâmetros do tuning:", rf_kwargs)

        log("=== Baseline: RandomForest (calibrated) + Purged K-Fold ===")
        results = run_baseline(
            X, y,
            n_splits=5,
            purge=5,
            embargo=0.01,
            sample_weight=w,        # passa pesos (ou None)
            **rf_kwargs,
        )

        # Guardar métricas estruturadas do baseline
        results_dir = Path("data"); results_dir.mkdir(exist_ok=True)
        results["cv_metrics_folds"].to_csv(results_dir / "cv_metrics_folds.csv", index=False)
        with open(results_dir / "cv_metrics_mean.json", "w") as f:
            json.dump(results["cv_metrics_mean"], f, indent=2)

        # --- Guardar probabilidades out-of-fold, se presentes
        oof = results.get("oof_prob", None)
        if oof is not None:
            oof.to_frame("oof_prob").to_parquet(results_dir / "oof_prob.parquet")
            print("Guardado:", results_dir / "oof_prob.parquet")

        print("\nResumo (CV médias):")
        print(results["cv_metrics_mean"])
        print("\nPrimeiras linhas de cv_metrics_folds.csv:")
        print(results["cv_metrics_folds"].head().to_string(index=False))

        log(f"Baseline concluído. Tempo total: {perf_counter()-t0:.2f}s")

    finally:
        # fecha SEMPRE a ligação, mesmo que haja erro acima
        ib.disconnect()
        log("Ligação IB encerrada. Fim do script.")


# --- entry point ---
if __name__ == "__main__":
    main()
