import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, auc,
    roc_auc_score, precision_recall_curve
)

try:
    import shap
except ImportError:
    shap = None

def classification_metrics(y_true, y_pred):
    """기본 분류 성능지표 dict 리턴"""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, average="weighted"),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0)
    }
    # ROC AUC (이진 분류만)
    if len(np.unique(y_true)) == 2:
        try:
            metrics["ROC_AUC"] = roc_auc_score(y_true, y_pred)
        except Exception:
            metrics["ROC_AUC"] = np.nan
    return metrics

def show_metrics(y_true, y_pred):
    st.subheader("📊 성능 지표 (Test Set)")
    metrics = classification_metrics(y_true, y_pred)
    st.write({k: f"{v:.4f}" for k, v in metrics.items()})

    st.markdown("""
    - **Accuracy**: 전체 예측 정답 비율 (직관적 지표)
    - **F1**: 정밀도-재현율 균형, 불균형/실무에서 가장 많이 사용
    - **Precision/Recall**: 클래스별 오탐/미탐 민감
    - **ROC_AUC**: 이진 분류에서 예측 확률의 분별력 (1에 가까울수록 좋음)
    """)

def plot_confusion_matrix(y_true, y_pred, labels=None):
    st.subheader("🔢 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels if labels is not None else np.unique(y_true),
                yticklabels=labels if labels is not None else np.unique(y_true))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

def plot_roc_curve(y_true, y_score):
    st.subheader("📈 ROC Curve")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC={auc_val:.2f}")
    ax.plot([0,1], [0,1], 'r--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

def plot_precision_recall_curve(y_true, y_score):
    st.subheader("📈 Precision-Recall Curve")
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    st.pyplot(fig)

def plot_feature_importance(model, X_test):
    st.subheader("🧑‍🔬 Permutation Feature Importance (PFI)")
    try:
        from sklearn.inspection import permutation_importance
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
        st.info("상업 AutoML에서도 자주 쓰이는 전체/개별 해석")
        # 전체 summary plot
        fig1 = shap.plots.beeswarm(shap_values, show=False)
        st.pyplot(bbox_inches='tight', pad_inches=0)
        st.markdown("**SHAP summary plot:** 전체 feature 영향력")
        st.pyplot(fig1)
        st.markdown("---")
        # 개별 force plot (첫번째 샘플)
        idx = st.slider("개별 예측 샘플(index)", 0, X_test.shape[0]-1, 0)
        st.markdown("**개별 예측 SHAP force plot**")
        fig2 = shap.plots.force(shap_values[idx], matplotlib=True, show=False)
        st.pyplot(fig2)
    except Exception as e:
        st.warning(f"SHAP 설명 실패: {e}")

def plot_top_errors(y_true, y_pred, X_test, top_n=5):
    st.subheader(f"예측 오류(실패) Top {top_n}")
    errors = (y_true != y_pred)
    wrong_idx = np.where(errors)[0][:top_n]
    st.write(X_test.iloc[wrong_idx])
    st.write("실제값:", pd.Series(y_true).iloc[wrong_idx].values)
    st.write("예측값:", pd.Series(y_pred).iloc[wrong_idx].values)

def advanced_evaluation(model, X_test, y_test):
    y_pred = model.predict(X_test)
    show_metrics(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred)
    # 이진 분류일 때만 ROC/PR 곡선
    if len(np.unique(y_test)) == 2:
        try:
            y_score = model.predict_proba(X_test)[:, 1]
            plot_roc_curve(y_test, y_score)
            plot_precision_recall_curve(y_test, y_score)
        except Exception:
            pass
    plot_feature_importance(model, X_test)
    plot_shap(model, X_test)
    plot_top_errors(y_test, y_pred, X_test)

    st.markdown("""
    ---
    ## 상업 AutoML 리포트 스타일
    - 주요 성능지표 및 해설
    - Confusion Matrix/PR/ROC 곡선
    - Permutation/SHAP feature importance
    - 예측실패 샘플 자동 표시
    """)

def evaluate(model, X_test, y_test):
    """상업 AutoML 스타일 분류 평가+XAI"""
    advanced_evaluation(model, X_test, y_test)