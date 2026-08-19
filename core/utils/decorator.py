import numpy as np
from functools import wraps
from time import perf_counter
from typing import Callable, Any, TypeVar, ParamSpec

from core.io.logger import setup_logger

# Generic Typage
P = ParamSpec("P")
R = TypeVar("R")

# Default logger (if self.logger not found)
default_logger = setup_logger()


def log_time() -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            logger = getattr(args[0], 'logger', default_logger)  # self.logger if available
            start = perf_counter()
            result = function(*args, **kwargs)
            end = perf_counter()
            logger.debug(f'[TIMER] "{function.__name__}" executed in {end - start:.4f} seconds')
            return result
        return wrapper
    return decorator


def format_arg(arg: Any) -> str:
    if isinstance(arg, np.ndarray):
        return f"ndarray(shape={arg.shape}, dtype={arg.dtype})"
    elif isinstance(arg, (list, tuple)) and len(arg) > 10:
        return f"{type(arg).__name__}(len={len(arg)})"
    elif isinstance(arg, dict):
        return f"dict(keys={list(arg.keys())})"
    else:
        return repr(arg)


def log_call() -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            logger = getattr(args[0], 'logger', default_logger)
            formatted_args = ", ".join(format_arg(a) for a in args)
            formatted_kwargs = ", ".join(f"{k}={format_arg(v)}" for k, v in kwargs.items())
            logger.debug(
                "[CALL] \"{name}\"\n"
                "                       - args:   [{args}]\n"
                "                       - kwargs: {{ {kwargs} }}".format(
                    name=function.__name__,
                    args=formatted_args if formatted_args else "None",
                    kwargs=formatted_kwargs if formatted_kwargs else "None"
                )
            )
            return function(*args, **kwargs)
        return wrapper
    return decorator


def log_section(message: str):
    """Décorateur pour logger [START] et [END] autour d'une fonction avec un message personnalisé."""
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            logger = getattr(args[0], 'logger', default_logger)
            logger.info(f"[START] {message}")
            result = function(*args, **kwargs)
            logger.info(f"[END] {message}")
            return result
        return wrapper
    return decorator


def log_init():
    """Décorateur pour logger [INIT] <setup.name> initialized à la fin d'une fonction."""
    def decorator(function):
        @wraps(function)
        def wrapper(self, *args, **kwargs):
            result = function(self, *args, **kwargs)
            logger = getattr(self, 'logger', default_logger)
            setup_name = getattr(getattr(self, 'setup', None), 'name', 'UnknownSetup')
            logger.info(f"[INIT] {setup_name.capitalize()} initialized")
            return result
        return wrapper
    return decorator
