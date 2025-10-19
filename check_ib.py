from ib_insync import IB
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path('config.yaml').read_text())
ib = IB()
ib.connect(cfg['ib']['host'], cfg['ib']['port'], clientId=cfg['ib']['clientId'], timeout=5)

print("Conectado?", ib.isConnected())
# serverVersion é propriedade do cliente:
print("Server version:", ib.client.serverVersion())
# opcional: hora do TWS para confirmar vida no socket
print("TWS time:", ib.reqCurrentTime())

ib.disconnect()
