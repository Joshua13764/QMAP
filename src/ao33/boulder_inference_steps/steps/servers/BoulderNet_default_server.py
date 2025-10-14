# server.py
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from fastapi import FastAPI
from pathlib import Path
import uvicorn

app = FastAPI()


@app.get("/ping")
def ping():
    return {"pong": True, "message": "Service alive"}


# Request body dataclass (FastAPI supports dataclasses for request bodies)
@dataclass
class EchoRequest:
    text: str


@app.post("/echo")
def echo(req: EchoRequest):
    return {"received": req.text, "length": len(req.text)}


# Server config dataclass with dataclasses_json helpers
@dataclass_json
@dataclass
class ServerConfig:
    host: str
    port: int


def load_config(path: str = "server_config.json") -> ServerConfig:
    """Load a JSON config file into a ServerConfig dataclass using dataclasses_json."""
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    return ServerConfig.from_json(raw)

if __name__ == "__main__":
    cfg = load_config()
    uvicorn.run(app, host=cfg.host, port=cfg.port)
