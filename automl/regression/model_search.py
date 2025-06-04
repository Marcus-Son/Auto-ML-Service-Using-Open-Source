import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def get_model_candidates():
    return {
        "LinearRegression": (LinearRegression(), {}),
        "Ridge": (Ridge(), {'alpha': [0.1, 1.0, 10.0]}),
        "Lasso": (Lasso(), {'alpha': [0.01, 0.1, 1.0]}),
        "ElasticNet": (ElasticNet(), {'alpha': [0.01, 0.1, 1.0], 'l1_ratio': [0.2, 0.5, 0.8]}),
        "RandomForest": (RandomForestRegressor(), {'n_estimators': [50, 100], 'max_depth': [None, 5, 10]}),
        "GradientBoosting": (GradientBoostingRegressor(), {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}),
        "SVR": (SVR(), {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}),
    }

def search_and_evaluate(X_train, y_train, X_valid, y_valid, X_test, y_test, scoring='neg_mean_squared_error'):
    st.header("🔎 자동 모델 서치 & 리더보드")
    leaderboard = []
    best_models = {}

    candidates = get_model_candidates()
    for name, (model, param_grid) in candidates.items():
        st.write(f"**{name}**: 하이퍼파라미터 튜닝 중...")
        if param_grid:
            gs = GridSearchCV(model, param_grid, cv=3, scoring=scoring, n_jobs=-1)
            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_
            best_params = gs.best_params_
            valid_pred = best_model.predict(X_valid)
        else:
            model.fit(X_train, y_train)
            best_model = model
            best_params = {}
            valid_pred = best_model.predict(X_valid)

        mse = mean_squared_error(y_valid, valid_pred)
        mae = mean_absolute_error(y_valid, valid_pred)
        r2 = r2_score(y_valid, valid_pred)
        leaderboard.append({
            'Model': name,
            'Params': best_params,
            'MSE': mse,
            'MAE': mae,
            'R2': r2
        })
        best_models[name] = best_model

    leaderboard_df = pd.DataFrame(leaderboard).sort_values("MSE")
    st.subheader("🏆 Leaderboard (Validation 기준 MSE 순)")
    st.dataframe(leaderboard_df, use_container_width=True)

    # 최적 모델 선정 및 테스트 세트 평가
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