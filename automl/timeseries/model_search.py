# automl/timeseries/model_search.py

import streamlit as st
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

# Prophet/ARIMA 등 패키지 사용 예시 (옵션)
try:
    from prophet import Prophet
except ImportError:
    Prophet = None

def get_model_candidates():
    candidates = {
        "Ridge": (Ridge(), {'alpha': [0.1, 1.0, 10.0]}),
        "Lasso": (Lasso(), {'alpha': [0.01, 0.1, 1.0]}),
        "RandomForest": (RandomForestRegressor(), {'n_estimators': [50, 100], 'max_depth': [None, 5, 10]}),
        "GradientBoosting": (GradientBoostingRegressor(), {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}),
    }
    if LGBMRegressor is not None:
        candidates["LightGBM"] = (LGBMRegressor(), {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]})
    if XGBRegressor is not None:
        candidates["XGBoost"] = (XGBRegressor(), {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]})
    return candidates

def search_and_evaluate(X_train, y_train, X_valid, y_valid, X_test, y_test, scoring='neg_mean_squared_error'):
    X_train = X_train.fillna(0)
    X_valid = X_valid.fillna(0)
    X_test = X_test.fillna(0)
    st.header("🔎 자동 모델 서치 & 리더보드 (TimeSeries)")
    leaderboard = []
    best_models = {}

    candidates = get_model_candidates()
    for name, (model, param_grid) in candidates.items():
        st.write(f"**{name}**: 하이퍼파라미터 튜닝 중...")
        best_mse = np.inf
        best_model = None
        best_params = {}
        # 간단한 grid search (cross-validation 아님)
        from itertools import product
        keys = list(param_grid.keys())
        vals = list(param_grid.values())
        for param_combo in product(*vals):
            params = dict(zip(keys, param_combo))
            model.set_params(**params)
            try:
                model.fit(X_train, y_train)
                valid_pred = model.predict(X_valid)
                mse = mean_squared_error(y_valid, valid_pred)
                if mse < best_mse:
                    best_mse = mse
                    best_model = type(model)(**params)
                    best_model.fit(np.vstack([X_train, X_valid]), np.concatenate([y_train, y_valid]))  # 재학습
                    best_params = params
            except Exception as e:
                st.write(f"{name} 파라미터 조합 실패: {params} | {e}")
                continue
        # 평가
        test_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, test_pred)
        mae = mean_absolute_error(y_test, test_pred)
        r2 = r2_score(y_test, test_pred)
        leaderboard.append({
            'Model': name,
            'Params': best_params,
            'MSE': mse,
            'MAE': mae,
            'R2': r2
        })
        best_models[name] = best_model

    leaderboard_df = pd.DataFrame(leaderboard).sort_values("MSE")
    st.subheader("🏆 Leaderboard (Test 기준 MSE 순)")
    st.dataframe(leaderboard_df, use_container_width=True)

    # 최적 모델 선정
    top_model_name = leaderboard_df.iloc[0]['Model']
    st.success(f"**최적의 모델: {top_model_name}**")
    best_model = best_models[top_model_name]
    test_pred = best_model.predict(X_test)
    test_mse = mean_squared_error(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    st.subheader("✅ Test 세트 평가")
    st.write(f"Test MSE: {test_mse:.4f}")
    st.write(f"Test MAE: {test_mae:.4f}")
    st.write(f"Test R2: {test_r2:.4f}")
    st.write(f"최적 하이퍼파라미터: {leaderboard_df.iloc[0]['Params']}")

    return leaderboard_df, best_model

# (옵션) Prophet 등 직접 적용 예시 (날짜, y)
def run_prophet(df, time_col, target, test_size=0.2):
    if Prophet is None:
        st.warning("Prophet 설치 필요: pip install prophet")
        return None, None
    st.write("Prophet: 기본 시계열 예측")
    dfp = df[[time_col, target]].rename(columns={time_col: "ds", target: "y"})
    dfp = dfp.dropna()
    n = len(dfp)
    train_df = dfp.iloc[:int(n*(1-test_size))]
    test_df = dfp.iloc[int(n*(1-test_size)):]
    m = Prophet()
    m.fit(train_df)
    forecast = m.predict(test_df)
    y_pred = forecast['yhat'].values
    y_true = test_df['y'].values
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    st.write(f"Prophet - Test MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return m, forecast