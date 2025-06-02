# automl/detector.py
"""
데이터셋과 타깃 컬럼을 넘기면 TaskType(Enum) 을 리턴합니다.

우선순위
1) 타임스탬프 컬럼 + 규칙적 간격  →  TIMESERIES
2) 실수형 타깃 & 고유값이 충분히 다양 →  REGRESSION
3) 나머지                                       →  CLASSIFICATION
"""
from typing import List
from pathlib import Path
import pandas as pd
import numpy as np
from automl import TaskType


# ----------------------------- 내부 유틸 ----------------------------- #
_MIN_NUM_CLASSES = 15          # 고유값 n↑ ⇒ 회귀로 간주
_CATEGORICAL_MAX_UNIQUE = 0.2  # 클래스 비중이 전체 20% 미만이면 분류로 간주


def _has_regular_datetime(col: pd.Series) -> bool:
    """단조·규칙적 타임스탬프 여부"""
    if not pd.api.types.is_datetime64_any_dtype(col):
        return False
    col = col.dropna().sort_values()
    if col.empty or not col.is_monotonic_increasing:
        return False
    return col.diff().dropna().nunique() == 1


def _is_numeric_regression(y: pd.Series) -> bool:
    """‘수치형 + 고유값 다양’ 여부 ↔ 회귀"""
    if not pd.api.types.is_numeric_dtype(y):
        return False
    # 정수형인데 고유값이 너무 적으면 분류일 수 있음
    if pd.api.types.is_integer_dtype(y):
        return y.nunique() > _MIN_NUM_CLASSES
    # 실수형이면 거의 회귀
    return True


# ------------------------------ API 함수 ----------------------------- #
def detect(df: pd.DataFrame, target: str) -> TaskType:
    """
    Parameters
    ----------
    df : pandas.DataFrame
    target : str
        예측하고자 하는 타깃 컬럼명

    Returns
    -------
    TaskType
        REGRESSION | CLASSIFICATION | TIMESERIES
    """
    if target not in df.columns:
        raise ValueError(f"'{target}' 컬럼이 데이터프레임에 없습니다.")

    y = df[target]

    # 1) 시계열 판별 (타깃 말고 다른 열에 날짜 컬럼이 있는 경우도 고려)
    for col in df.columns:
        if col == target:
            continue
        if _has_regular_datetime(df[col]):
            return TaskType.TIMESERIES

    # 2) 회귀 판별
    if _is_numeric_regression(y):
        return TaskType.REGRESSION

    # 3) 분류 (fallback)
    return TaskType.CLASSIFICATION