import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance

# SHAP 및 LIME 등 XAI
try:
    import shap
except ImportError:
    shap = None

def regression_metrics(y_true, y_pred):
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
    metrics = regression_metrics(y_true, y_pred)
    st.write({k: f"{v:.4f}" for k, v in metrics.items()})

    # 해설 예시
    st.markdown(f"""
    - **MSE** (평균제곱오차): 낮을수록 좋음
    - **MAE** (평균절대오차): 실제 오차의 평균. 해석이 직관적
    - **RMSE**: MSE와 유사, 이상치에 민감
    - **MAPE**: 예측값/실제값 비율로 직관적 정확도
    - **R2**: 1에 가까울수록 좋음 (설명력)
    """)

def plot_residuals(y_true, y_pred):
    st.subheader("📉 잔차(Residuals) 분석")
    residuals = y_true - y_pred
    fig, axs = plt.subplots(1, 2, figsize=(12,4))
    sns.scatterplot(x=y_pred, y=residuals, ax=axs[0])
    axs[0].axhline(0, color='red', linestyle='--')
    axs[0].set_title("예측값 vs 잔차")
    axs[0].set_xlabel("예측값")
    axs[0].set_ylabel("잔차")
    sns.histplot(residuals, kde=True, ax=axs[1])
    axs[1].set_title("잔차 분포")
    st.pyplot(fig)

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
    except Exception as e:
        st.warning(f"Permutation Importance 계산 실패: {e}")

def plot_shap(model, X_test):
    st.subheader("🔎 SHAP Explainability")
    if shap is None:
        st.info("SHAP 설치 필요: `pip install shap`")
        return
    try:
        explainer = shap.Explainer(model, X_test)
        shap_values = explainer(X_test)
        st.info("상업 AutoML에서도 가장 자주 쓰이는 전체/개별 예측 해석")
        fig1 = shap.plots.beeswarm(shap_values, show=False)
        st.pyplot(bbox_inches='tight', pad_inches=0)
        st.markdown("**SHAP summary plot:** 전체 feature 영향력")
        st.pyplot(fig1)
        st.markdown("---")
        idx = st.slider("개별 예측 샘플(index)", 0, X_test.shape[0]-1, 0)
        st.markdown("**개별 예측 SHAP force plot**")
        fig2 = shap.plots.force(shap_values[idx], matplotlib=True, show=False)
        st.pyplot(fig2)
    except Exception as e:
        st.warning(f"SHAP 설명 실패: {e}")

def plot_predicted_vs_actual(y_true, y_pred):
    st.subheader("📈 예측값 vs 실제값 산점도")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_true, y=y_pred, ax=ax)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax.set_xlabel("실제값")
    ax.set_ylabel("예측값")
    ax.set_title("실제값 vs 예측값")
    st.pyplot(fig)

def advanced_evaluation(model, X_test, y_test):
    # 전통적 성능지표
    y_pred = model.predict(X_test)
    show_metrics(y_test, y_pred)
    plot_predicted_vs_actual(y_test, y_pred)
    plot_residuals(y_test, y_pred)
    plot_feature_importance(model, X_test)
    plot_shap(model, X_test)

    # 상업 AutoML식 리포트 안내
    st.markdown("""
    ---
    ## 상업 AutoML 서비스 리포트 스타일
    - **에러 분석**: 이상치 샘플 자동 탐지/표시 (아래 표)
    - **예측 신뢰구간(Confidence Interval)**: (이론적/추정치, 추가 구현 가능)
    - **Top N 중요 feature** + 자동 해석 코멘트
    - **전체 요약 PDF/HTML 저장 (Streamlit 내 추가 구현 가능)**
    """)

    # 이상치 예시
    st.subheader("에러가 큰 샘플 Top 5")
    errors = np.abs(y_test - y_pred)
    top_idx = np.argsort(-errors)[:5]
    st.write(pd.DataFrame({
        "실제값": y_test.iloc[top_idx].values,
        "예측값": y_pred[top_idx],
        "에러": errors.iloc[top_idx]
    }))

    # (추가로: PDF/HTML 리포트, Confidence Interval, Counterfactual 등 가능)


# ----- Main API -----
def evaluate(model, X_test, y_test):
    """상업 AutoML 스타일 회귀 평가+XAI"""
    advanced_evaluation(model, X_test, y_test)