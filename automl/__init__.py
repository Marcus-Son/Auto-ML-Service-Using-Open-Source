from enum import Enum

class TaskType(Enum):
    REGRESSION = "Regression"
    CLASSIFICATION = "Classification"
    TIMESERIES = "Time‑Series"

__all__ = ["TaskType"]
