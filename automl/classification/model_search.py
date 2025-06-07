import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# XGBoost, LightGBM optional
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

def get_model_candidates():
    candidates = {
        "LogisticRegression": (LogisticRegression(max_iter=1000), {
            "C": [0.1, 1.0, 10.0],
            "solver": ["liblinear"]
        }),
        "RandomForest": (RandomForestClassifier(), {
            "n_estimators": [50, 100],
            "max_depth": [None, 5, 10]
        }),
        "GradientBoosting": (GradientBoostingClassifier(), {
            "n_estimators": [50, 100],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5]
        }),
        "SVM": (SVC(probability=True), {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"]
        }),
        "DecisionTree": (DecisionTreeClassifier(), {
            "max_depth": [None, 5, 10]
        }),
        "KNeighbors": (KNeighborsClassifier(), {
            "n_neighbors": [3, 5, 7]
        }),
        "NaiveBayes": (GaussianNB(), {})
    }
    # XGBoost, LightGBM 추가 (설치되어 있을 때만)
    if XGBClassifier is not None:
        candidates["XGBoost"] = (XGBClassifier(eval_metric='mlogloss', use_label_encoder=False), {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1]
        })
    if LGBMClassifier is not None:
        candidates["LightGBM"] = (LGBMClassifier(), {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1]
        })
    return candidates

def search_and_evaluate(X_train, y_train, X_valid, y_valid, X_test, y_test, scoring='f1_weighted'):
    st.header("🔎 자동 분류 모델 서치 & 리더보드")
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
            valid_pred_proba = None
            try:
                valid_pred_proba = best_model.predict_proba(X_valid)[:, 1]
            except Exception:
                valid_pred_proba = None
        else:
            model.fit(X_train, y_train)
            best_model = model
            best_params = {}
            valid_pred = best_model.predict(X_valid)
            valid_pred_proba = None
            try:
                valid_pred_proba = best_model.predict_proba(X_valid)[:, 1]
            except Exception:
                valid_pred_proba = None

        # 평가: multi-class 지원
        acc = accuracy_score(y_valid, valid_pred)
        f1 = f1_score(y_valid, valid_pred, average="weighted")
        prec = precision_score(y_valid, valid_pred, average="weighted", zero_division=0)
        rec = recall_score(y_valid, valid_pred, average="weighted", zero_division=0)
        if valid_pred_proba is not None and len(np.unique(y_valid)) == 2:
            roc_auc = roc_auc_score(y_valid, valid_pred_proba)
        else:
            roc_auc = np.nan
        leaderboard.append({
            'Model': name,
            'Params': best_params,
            'Accuracy': acc,
            'F1': f1,
            'Precision': prec,
            'Recall': rec,
            'ROC_AUC': roc_auc
        })
        best_models[name] = best_model

    leaderboard_df = pd.DataFrame(leaderboard).sort_values("F1", ascending=False)
    st.subheader("🏆 Leaderboard (Validation F1 기준)")
    st.dataframe(leaderboard_df, use_container_width=True)

    # 최적 모델 선정 및 테스트 세트 평가
    top_model_name = leaderboard_df.iloc[0]['Model']
    st.success(f"**최적의 모델: {top_model_name}**")
    best_model = best_models[top_model_name]
    test_pred = best_model.predict(X_test)
    try:
        test_pred_proba = best_model.predict_proba(X_test)[:, 1]
    except Exception:
        test_pred_proba = None
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average="weighted")
    test_prec = precision_score(y_test, test_pred, average="weighted", zero_division=0)
    test_rec = recall_score(y_test, test_pred, average="weighted", zero_division=0)
    if test_pred_proba is not None and len(np.unique(y_test)) == 2:
        test_roc_auc = roc_auc_score(y_test, test_pred_proba)
    else:
        test_roc_auc = np.nan

    st.subheader("✅ Test 세트 평가")
    st.write(f"Test Accuracy: {test_acc:.4f}")
    st.write(f"Test F1: {test_f1:.4f}")
    st.write(f"Test Precision: {test_prec:.4f}")
    st.write(f"Test Recall: {test_rec:.4f}")
    if not np.isnan(test_roc_auc):
        st.write(f"Test ROC AUC: {test_roc_auc:.4f}")
    st.write(f"최적 하이퍼파라미터: {leaderboard_df.iloc[0]['Params']}")

    return leaderboard_df, best_model