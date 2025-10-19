# main.py
"""
Pipeline principal (didático):
1) Lê config.yaml e liga ao TWS/Gateway (IB).
2) Busca OHLCV históricos conforme a config.
3) Constrói volume-bars com threshold dinâmico (ou absoluto se vier na config).
4) Extrai features (volatilidade, ATR, entropia, LZ, CUSUM, VPIN).
5) Calcula labels (triple-barrier) e meta-labels.
6) Monta matriz X e vetor y, grava em data/.
7) Lê data/rf_best.json (se existir) e corre baseline com esses parâmetros
   (agora com calibração isotónica e métricas adicionais: PR-AUC e F1@thr*).

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
from bars import volume_bars
from features import (
    rolling_volatility,
    atr,
    shannon_entropy,
    lz_complexity,
    cusum_efp,
    vpin,
)
from labeler import apply_triple_barrier, meta_labeling
from model import run_baseline  # -> agora devolve também métricas por fold/médias


# ===================== opções rápidas (podes ajustar) ===================== #
# multiplicador da média de volume para definir o threshold das volume-bars
VOLUME_THRESHOLD_MULT = 1.0  # 1.0 = 100% da média; põe 0.5/2.0 se precisares
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
            duration=h["durationStr"],            # ex.: "5 D"
            bar_size=h["barSizeSetting"],         # ex.: "5 mins"
            what_to_show=h.get("whatToShow"),     # None => TRADES/MIDPOINT por defeito
            use_rth=h.get("useRTH", True),
        )
        print(f"\nDados brutos (OHLCV) recebidos: {df.shape}")
        print(df.head(), "\n")
        if df.empty:
            log("Sem dados — verifica duration/barSize na config.yaml.")
            return

        # 2) Threshold das volume-bars
        #    Se houver 'bars.threshold' na config usamos esse absoluto;
        #    caso contrário, usamos média*VOLUME_THRESHOLD_MULT (dinâmico).
        cfg_threshold = (cfg.get("bars", {}) or {}).get("threshold", None)
        if cfg_threshold:
            threshold = float(cfg_threshold)
            thr_src = "config"
        else:
            avg_vol = float(df["volume"].mean())
            threshold = max(1.0, VOLUME_THRESHOLD_MULT * avg_vol)
            thr_src = f"{VOLUME_THRESHOLD_MULT:.1f}×média"
        log(f"Média de volume: {df['volume'].mean():.0f} | "
            f"threshold usado: {threshold:.0f} ({thr_src})")

        # 3) Construir volume-bars
        log("A construir volume-bars…")
        vb = volume_bars(df, volume_threshold=threshold)
        print(f"Barras originais: {df.shape} | volume-bars: {vb.shape}\n")
        if vb.empty:
            log("Volume-bars vazias — baixa o threshold ou aumenta a janela.")
            return

        # 4) Features (primeiro conjunto)
        log("A calcular features…")
        t_feats = perf_counter()

        vol10   = rolling_volatility(vb["close"], window=10)          # volatilidade
        atr14   = atr(vb, window=14)                                  # ATR clássico
        ent20   = vb["close"].rolling(20).apply(shannon_entropy, False)
        lz20    = vb["close"].rolling(20).apply(lz_complexity,   False)
        sb_flag = cusum_efp(vb["close"], threshold=2.5)               # quebras
        vpin10  = vpin(vb, bucket_size=10)                            # fluxo ordens (proxy)

        print(f"Features calculadas em {perf_counter()-t_feats:.3f}s\n")
        print("vol10 head:\n",   vol10.head(),   "\n")
        print("atr14 head:\n",   atr14.head(),   "\n")
        print("sb_flag head:\n", sb_flag.head(), "\n")
        print("ent20 head:\n",   ent20.head(),   "\n")
        print("lz20  head:\n",   lz20.head(),    "\n")
        print("vpin10 head:\n",  vpin10.head(),  "\n")

        # 5) Labels e meta-labels
        log("A calcular labels (Triple-Barrier)…")
        labels = apply_triple_barrier(vb, pt=0.005, sl=0.005, max_bars=10)
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
            **rf_kwargs,   # se vieram do JSON, sobrepõem os defaults
        )

        # --- Guardar métricas estruturadas do baseline (NOVIDADE) ---
        # Ficheiros: cv_metrics_folds.csv (todas as folds) e cv_metrics_mean.json (médias)
        results_dir = Path("data"); results_dir.mkdir(exist_ok=True)
        results["cv_metrics_folds"].to_csv(results_dir / "cv_metrics_folds.csv", index=False)

        with open(results_dir / "cv_metrics_mean.json", "w") as f:
            json.dump(results["cv_metrics_mean"], f, indent=2)

        # Nota: feature_importance.csv já é gravado pelo próprio model.py;
        # manter esta linha se quiseres sobrepor: results["feature_importance"].to_csv(results_dir / "feature_importance.csv")

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
