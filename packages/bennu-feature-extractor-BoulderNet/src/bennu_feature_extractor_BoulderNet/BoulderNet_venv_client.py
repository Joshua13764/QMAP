from .venv_client_base import IVenvClient

import requests

class BoulderNetVenvClient(IVenvClient):
    def __init__(self, logger, config):
        self._logger = logger
        self._config = config
        self.base = f"http://{config.host}:{config.port}"
        self._logger.info(f"ServerClient initialized with base URL: {self.base}")

    def test_ping(self):
        r = requests.get(f"{self.base}/ping", timeout=5)
        data = r.json()
        self._logger.info(f"Ping response: {data}")
        return data

    def test_echo(self, text):
        r = requests.post(f"{self.base}/echo", json={"text": text}, timeout=10)
        data = r.json()
        self._logger.info(f"Echo response: {data}")
        return data
    
    def get_base_url(self):
        return self.base
    
    def test(self):
        self.test_ping()
        self.test_echo("ClientTestString")