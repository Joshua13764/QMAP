import logging
from dataclasses import dataclass
from pathlib import Path

import uvicorn
from dataclasses_json import dataclass_json
from fastapi import FastAPI, Request
from rich.console import Console
from rich.logging import RichHandler


# === Configure logging ===
def setup_logger(log_dir: Path, name: str = "server") -> logging.Logger:
    log_dir.mkdir(exist_ok=True, parents=True)
    log_file = log_dir / f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler (colorful output)
    console = Console(force_terminal=True)
    console_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=False,
        markup=True,
        log_time_format="%Y-%m-%d %H:%M:%S",
    )
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.propagate = False
    logger.info(f"[cyan]Logger initialized → {log_file}[/]")
    return logger


# === FastAPI app ===
app = FastAPI()
logger = setup_logger(Path("logs"))  # logs/server.log


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware that logs all incoming HTTP requests and their responses."""
    logger.info(f"➡️  {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"⬅️  {response.status_code} {request.url.path}")
    return response


@app.get("/ping")
def ping():
    logger.info("Received /ping request")
    return {"pong": True, "message": "Service alive"}


@dataclass
class EchoRequest:
    text: str


@app.post("/echo")
def echo(req: EchoRequest):
    logger.info(f"Echo request received: {req.text!r}")
    return {"received": req.text, "length": len(req.text)}


# === Config dataclass ===
@dataclass_json
@dataclass
class ServerConfig:
    host: str
    port: int


def load_config(path: str = "server_config.json") -> ServerConfig:
    """Load a JSON config file into a ServerConfig dataclass."""
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    return ServerConfig.from_json(raw)


if __name__ == "__main__":
    cfg = load_config()
    logger.info(f"🚀 Starting server at http://{cfg.host}:{cfg.port}")
    uvicorn.run(app, host=cfg.host, port=cfg.port)
