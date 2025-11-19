import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class DownloadSessionFactory:
    @staticmethod
    def create_with_retries() -> requests.Session:
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
