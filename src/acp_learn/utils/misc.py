from typing import Any
from functools import wraps


def get_metric_value(metric_dict: dict[str, Any], metric_name: str | None) -> float | None:
    if metric_name is None:
        return None
    if metric_name in metric_dict:
        return float(metric_dict[metric_name])
    return None


def task_wrapper(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        return fn(*args, **kwargs)
    return inner
