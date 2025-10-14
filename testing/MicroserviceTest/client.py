# client.py
import requests

BASE = "http://127.0.0.1:5000"

def test_ping():
    r = requests.get(f"{BASE}/ping", timeout=5)
    print("Ping response:", r.json())

def test_echo(text):
    r = requests.post(f"{BASE}/echo", json={"text": text}, timeout=10)
    print("Echo response:", r.json())

if __name__ == "__main__":
    test_ping()
    test_echo("hello from Python 3.13")
