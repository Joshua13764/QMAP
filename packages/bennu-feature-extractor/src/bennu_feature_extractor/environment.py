from logger_factory import get_logger
from pathlib import Path

class Environment():

    def __init__(self, log_path : Path = Path("./logs")) -> None:
        
        self.Logger = get_logger("Main log", log_dir=log_path)