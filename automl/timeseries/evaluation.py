# automl/timeseries/evaluation.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance

try:
    import shap
except ImportError:
    shap = None

def timeseries_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

def show_metrics(y_true, y_pred):
    st.subheader("📊 성능 지표 (Test Set)")
    metrics = timeseries_metrics(y_true, y_pred)
    st.write({k: f"{v:.4f}" for k, v in metrics.items()})
    st.markdown("""
    - **MSE** (평균제곱오차): 낮을수록 좋음
    - **MAE** (평균절대오차): 실제 오차의 평균
    - **RMSE**: MSE의 제곱근, 이상치 민감
    - **MAPE**: 직관적 백분율 오차
    - **R2**: 1에 가까울수록 좋음 (설명력)
    """)

def plot_trend(y_true, y_pred, time_idx=None, title="실제값 vs 예측값"):
    st.subheader("📈 실제값/예측값 추세 비교")
    fig, ax = plt.subplots(figsize=(14, 4))
    if time_idx is not None:
        ax.plot(time_idx, y_true, label="실제값")
        ax.plot(time_idx, y_pred, label="예측값", alpha=0.7)
    else:
        ax.plot(y_true, label="실제값")
        ax.plot(y_pred, label="예측값", alpha=0.7)
    ax.legend()
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)

def plot_residuals(y_true, y_pred, time_idx=None):
    st.subheader("📉 잔차(Residuals) 분석")
    residuals = y_true - y_pred
    fig, axs = plt.subplots(1, 2, figsize=(14,4))
    if time_idx is not None:
        axs[0].plot(time_idx, residuals)
    else:
        axs[0].plot(residuals)
    axs[0].axhline(0, color='red', linestyle='--')
    axs[0].set_title("잔차 시계열")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("잔차")
    sns.histplot(residuals, kde=True, ax=axs[1])
    axs[1].set_title("잔차 분포")
    st.pyplot(fig)
    plt.close(fig)

def plot_predicted_vs_actual(y_true, y_pred):
    st.subheader("📊 예측값 vs 실제값 산점도")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_true, y=y_pred, ax=ax)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax.set_xlabel("실제값")
    ax.set_ylabel("예측값")
    ax.set_title("실제값 vs 예측값")
    st.pyplot(fig)
    plt.close(fig)

def plot_feature_importance(model, X_test):
    st.subheader("🧑‍🔬 Permutation Feature Importance (PFI)")
    try:
        result = permutation_importance(model, X_test, model.predict(X_test), n_repeats=20, random_state=42)
        importance = pd.Series(result.importances_mean, index=X_test.columns)
        topk = importance.abs().sort_values(ascending=False).head(15)
        fig, ax = plt.subplots()
        topk.plot.barh(ax=ax)
        ax.set_title("Permutation Feature Importance (상위 15)")
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Permutation Importance 계산 실패: {e}")

def plot_shap(model, X_test, y_test=None):
    st.subheader("🔎 SHAP Explainability")
    if shap is None:
        st.info("SHAP 설치 필요: `pip install shap`")
        return
    try:
        # NaN/inf 처리
        if isinstance(X_test, pd.DataFrame):
            X_test_plot = X_test.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
            if X_test_plot.shape[0] == 0:
                st.warning("SHAP 분석을 위한 유효한 샘플이 없습니다.")
                return
        else:
            X_test_plot = X_test

        # SHAP summary plot (additivity check 끔)
        explainer = shap.Explainer(model, X_test_plot)
        shap_values = explainer(X_test_plot, check_additivity=False)
        st.info("전체 feature 영향력 시각화")
        fig = plt.figure()
        try:
            shap.plots.beeswarm(shap_values, show=False)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"SHAP summary plot 실패: {e}")
        st.markdown("**SHAP summary plot:** 전체 feature 영향력")
        st.markdown("---")

        # force plot: 자동 샘플 선택 (가장 큰 에러, 아니면 0번)
        idx_auto = 0
        if y_test is not None and X_test_plot.shape[0] == len(y_test):
            y_pred = model.predict(X_test_plot)
            errors = np.abs(np.array(y_test) - y_pred)
            idx_auto = int(np.argmax(errors))

        st.markdown(f"**SHAP force plot (자동 선택: 샘플 #{idx_auto})**")

        try:
            expected_value = explainer.expected_value
            if isinstance(expected_value, (np.ndarray, list)):
                expected_value = expected_value[0]
            shap_val = shap_values[idx_auto].values if hasattr(shap_values[idx_auto], 'values') else np.array(shap_values[idx_auto])
            feature_val = X_test_plot.iloc[idx_auto].values if hasattr(X_test_plot, 'iloc') else X_test_plot[idx_auto]

            # force plot (matplotlib=False, show=False) **check_additivity 빼라**
            force_html = shap.plots.force(
                expected_value,
                shap_val,
                feature_val,
                matplotlib=False,
                show=False
            )
            import streamlit.components.v1 as components
            components.html(shap.getjs(), height=0)
            components.html(force_html, height=300)
        except Exception as e:
            st.warning(f"SHAP force plot 실패: {e}")
    except Exception as e:
        st.warning(f"SHAP 설명 실패: {e}")

def advanced_evaluation(model, X_test, y_test, time_idx=None):
    y_pred = model.predict(X_test)
    show_metrics(y_test, y_pred)
    plot_trend(y_test, y_pred, time_idx)
    plot_predicted_vs_actual(y_test, y_pred)
    plot_residuals(y_test, y_pred, time_idx)
    plot_feature_importance(model, X_test)
    plot_shap(model, X_test)

    # 에러가 큰 구간(상위 5) 표시
    st.subheader("에러가 큰 시점 Top 5")
    y_true_arr = np.array(y_test)
    errors = np.abs(y_true_arr - y_pred)
    top_idx = np.argsort(-errors)[:5]
    if time_idx is not None:
        time_arr = np.array(time_idx)
        error_df = pd.DataFrame({
            "시점": time_arr[top_idx],
            "실제값": y_true_arr[top_idx],
            "예측값": y_pred[top_idx],
            "에러": errors[top_idx]
        })
    else:
        error_df = pd.DataFrame({
            "실제값": y_true_arr[top_idx],
            "예측값": y_pred[top_idx],
            "에러": errors[top_idx]
        })
    st.write(error_df)

def evaluate(model, X_test, y_test, time_idx=None):
    advanced_evaluation(model, X_test, y_test, time_idx)