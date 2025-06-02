import pandas as pd
from automl import detector, TaskType

def test_regression():
    df = pd.DataFrame({"x": range(100), "y": range(100)})
    assert detector.detect(df, "y") == TaskType.REGRESSION
