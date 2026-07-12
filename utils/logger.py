import sys
import logging
from typing import Optional


def get_logger(name="custom", **kwargs):
    logger = Logger.instance(name, **kwargs)
    return logger


class Logger(logging.Logger):
    instances = {}

    def __init__(
        self,
        name: str = "custom",
        log_file: Optional[str] = None,
        log_level: int = logging.INFO,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.log_file = log_file

        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s %(filename)s:%(lineno)d] %(message)s"
        )
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(log_level)
        self.handlers.append(stream_handler)

        if log_file is not None:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            self.handlers.append(file_handler)

        self.propagate = False

    @classmethod
    def instance(cls, name: str = "custom", **kwargs) -> "Logger":
        instance_dict = cls.instances
        if name not in cls.instances:
            instance = cls(name, **kwargs)
            instance_dict[name] = instance
        else:
            instance = instance_dict[name]
            if (
                kwargs.get("log_file", None) is not None
                and getattr(instance, "log_file") is None
            ):
                instance = cls(name, **kwargs)
                instance_dict[name] = instance

        return instance_dict[name]
