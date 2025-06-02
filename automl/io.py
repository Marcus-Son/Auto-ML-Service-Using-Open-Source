import pandas as pd
from pathlib import Path
from typing import Union

def load(src: Union[str, Path, object]) -> pd.DataFrame:
    if hasattr(src, "read"):
        name = getattr(src, "name", "uploaded")
        if name.endswith(".csv"):
            return pd.read_csv(src)
        elif name.endswith(".parquet"):
            return pd.read_parquet(src)
        raise ValueError("지원하지 않는 포맷")
    src = Path(src)
    if src.suffix == ".csv":
        return pd.read_csv(src)
    if src.suffix == ".parquet":
        return pd.read_parquet(src)
    raise ValueError("csv / parquet 만 지원")