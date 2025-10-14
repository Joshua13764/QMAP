# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from jsondataclasses import jsondataclass
import json

app = FastAPI()

@app.get("/ping")
def ping():
    return {"pong": True, "message": "Service alive"}

class EchoRequest(BaseModel):
    text: str

@app.post("/echo")
def echo(req: EchoRequest):
    return {"received": req.text, "length": len(req.text)}


@jsondataclass
class ServerConfig:
    host: str
    port: int

def load_config(path: str = "server_config.json") -> ServerConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ServerConfig(data)

if __name__ == "__main__":
    import uvicorn

    cfg = load_config()
    uvicorn.run(app, host=cfg.host, port=cfg.port)
