# data_fetcher.py

"""
Módulo para ligar à IB e recolher dados de mercado (ticks, nível-2, barras).
"""

from ib_insync import IB, util, Stock, Future
import pandas as pd

def connect_ib(host: str = '127.0.0.1',
               port: int = 7497,
               client_id: int = 1) -> IB:
    """
    Estabelece e devolve uma ligação ativa ao TWS/Gateway da IB.
    """
    ib = IB()
    ib.connect(host, port, clientId=client_id)
    return ib

def fetch_data(
    ib: IB,
    contract,
    duration: str = '1 D',
    bar_size: str = '1 min',
    what_to_show: str = 'TRADES',
    use_rth: bool = True
) -> pd.DataFrame:
    """
    Puxa barras históricas de IB e devolve um DataFrame com as colunas:
    ['open', 'high', 'low', 'close', 'volume'], indexado pela data UTC.

    Parâmetros:
      ib           – instância ativa de IB retornada por connect_ib()
      contract     – um objeto Stock, Future, etc.
      duration     – ex: '1 D', '2 W', '6 M'
      bar_size     – ex: '5 secs', '1 min', '5 mins'
      what_to_show – 'TRADES', 'MIDPOINT', 'BID', 'ASK', etc.
      use_rth      – True para horas de negociação regulares

    Exemplo:
      ib = connect_ib()
      df = fetch_data(ib,
                      Stock('AAPL','SMART','USD'),
                      duration='2 D',
                      bar_size='5 secs')
    """
    # 1) Pedir dados históricos
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow=what_to_show,
        useRTH=use_rth,
        formatDate=1
    )
    # 2) Converter para DataFrame
    df = util.df(bars)
    # 3) Ajustar colunas e índice
    df = df.set_index('date')[['open', 'high', 'low', 'close', 'volume']]
    # 4) Garantir o tipo datetime e ordenação
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    return df
