"""
data_fetcher.py
----------------
Módulo para ligar à Interactive Brokers (IB) e recolher dados históricos
em formato de barras. Mantém o estilo 'didático' com explicações in-line.

Compatível com o ficheiro de configuração `config.yaml` na raiz do projeto.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from ib_insync import IB, util, Stock, Forex  # (Future pode ser adicionado depois)
import yaml
import logging


# ---------------------------------------------------------------------
# 0) Utilitário: ler configuração do ficheiro YAML
# ---------------------------------------------------------------------
def load_cfg(path: str | Path = "config.yaml") -> dict:
    """
    Lê e devolve o dicionário de configuração (host/porta, contrato, janela, etc.)
    """
    return yaml.safe_load(Path(path).read_text())


# ---------------------------------------------------------------------
# 1) Ligação ao TWS/Gateway
# ---------------------------------------------------------------------
def connect_ib(host: str = "127.0.0.1", port: int = 7497, client_id: int = 7) -> IB:
    """
    Estabelece e devolve uma ligação ativa ao TWS/Gateway da IB.

    Dicas:
      - No TWS: File > Global Configuration > API > Settings
        * Enable ActiveX and Socket Clients (ligado)
        * Socket port = 7497
        * Read-Only API (desligado)
        * Allow connections from localhost (ligado)
    """
    ib = IB()
    ib.connect(host, port, clientId=client_id, timeout=5)
    if not ib.isConnected():
        raise RuntimeError("Falha na ligação ao IB. Verifica TWS/Gateway e a porta 7497.")
    return ib


# ---------------------------------------------------------------------
# 2) Fabricação de 'contract' (Stock/Forex) a partir de dict
# ---------------------------------------------------------------------
def make_contract(contract_cfg: dict):
    """
    Cria um contrato IB (Stock, Forex, …) a partir de um dicionário de config.

    Exemplo de `contract_cfg` (retirado do config.yaml):
      {
        "type": "Stock", "symbol": "AAPL", "exchange": "SMART", "currency": "USD"
      }
      ou
      { "type": "Forex", "symbol": "EURUSD" }
    """
    ctype = contract_cfg["type"].lower()
    if ctype == "stock":
        return Stock(
            contract_cfg["symbol"],
            contract_cfg.get("exchange", "SMART"),
            contract_cfg.get("currency", "USD"),
        )
    elif ctype == "forex":
        # Em FX, a IB usa 'CASH'; usa-se 'MIDPOINT' para históricos
        return Forex(contract_cfg["symbol"])
    else:
        raise ValueError("contract.type deve ser 'Stock' ou 'Forex' (Future podemos adicionar depois).")


# ---------------------------------------------------------------------
# 3) Função genérica para pedir barras históricas e devolver DataFrame
# ---------------------------------------------------------------------
def fetch_data(
    ib: IB,
    contract,
    duration: str = "1 D",
    bar_size: str = "1 min",
    what_to_show: Optional[str] = None,
    use_rth: bool = True,
) -> pd.DataFrame:
    """
    Pede barras históricas à IB e devolve um DataFrame com colunas:
    ['open', 'high', 'low', 'close', 'volume'], indexado por timestamp (UTC).

    Parâmetros:
      ib           – instância ativa obtida via connect_ib()
      contract     – objeto IB (Stock, Forex, …)
      duration     – janela (ex.: '1 D', '2 W', '6 M', '1 Y')
      bar_size     – granularidade (ex.: '5 secs', '1 min', '5 mins', '1 hour')
      what_to_show – 'TRADES' (ações), 'MIDPOINT' (forex), 'BID', 'ASK', etc.
                     Se None, escolhemos automaticamente em função do tipo do contrato.
      use_rth      – True para horas regulares; False inclui pré/pós-mercado (stocks)

    Passos internos:
      1) Envia pedido ao TWS (reqHistoricalData)
      2) Converte a lista de barras em DataFrame (util.df)
      3) Seleciona/renomeia colunas, ordena por tempo e garante timezone UTC
    """
    # 1) Escolher 'what_to_show' se não vier definido: em FX usa-se MIDPOINT
    if what_to_show is None:
        what_to_show = "MIDPOINT" if getattr(contract, "secType", "") == "CASH" else "TRADES"

    logging.info(f"Pedido histórico: {contract.localSymbol if hasattr(contract, 'localSymbol') else contract} "
                 f"| {duration} @ {bar_size} ({what_to_show}) | RTH={use_rth}")

    # 2) Pedir barras ao TWS (nota: keepUpToDate=False retorna histórico e termina)
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow=what_to_show,
        useRTH=use_rth,
        formatDate=1,
        keepUpToDate=False,
    )

    # 3) Converter para DataFrame
    df = util.df(bars)  # tem colunas típicas: date, open, high, low, close, volume, barCount, average
    if df.empty:
        logging.warning("⚠️  A IB devolveu 0 linhas (tenta aumentar 'duration' ou ajustar 'bar_size').")
        # devolvemos um DF com colunas esperadas para não partir o resto do pipeline
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # 4) Indexar por 'date' e reter only OHLCV
    #    (em alguns mercados o 'volume' pode ser 0 — é normal em MIDPOINT de FX)
    df = (
        df.set_index("date")[["open", "high", "low", "close", "volume"]]
        .sort_index()
    )
    # 5) Garantir timezone UTC
    df.index = pd.to_datetime(df.index, utc=True)

    logging.info(f"Históricos recebidos: {df.shape[0]} linhas × {df.shape[1]} colunas")
    return df


# ------------------------------------
