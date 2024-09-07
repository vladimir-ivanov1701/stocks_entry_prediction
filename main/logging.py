import logging
from constants import LOG_FORMAT


class MyLogger(logging.Logger):
    def __init__(self):
        super().__init__()
        self.setLevel(logging.DEBUG)
        self.handler = logging.FileHandler(f"{__name__}.log", mode="w")
        self.formatter = logging.Formatter(LOG_FORMAT)
        self.handler.setFormatter(self.formatter)
        self.addHandler(self.handler)
