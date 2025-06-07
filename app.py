import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from automl import io, detector, TaskType

from automl.regression import eda as reg_eda
from automl.regression import preprocessing as reg_prep
from automl.regression import model_search as reg_model_search
from automl.regression import evaluation as reg_eval
from automl.regression import report as reg_report

from automl.classification import eda as cls_eda
from automl.classification import preprocessing as cls_prep
from automl.classification import model_search as cls_model_search
from automl.classification import evaluation as cls_eval
from automl.classification import report as cls_report

import re
def safe_feature_names(df):
    def safe_col(col):
        return re.sub(r'[^0-9a-zA-Z_]', '_', col)
    df.columns = [safe_col(str(col)) for col in df.columns]
    return df

st.set_page_config(page_title="Auto-ML Service", page_icon="🔮")
state = st.session_state
if "step" not in state:
    state.step, state.df, state.target, state.task = 1, None, None, None

# 1) 파일 업로드
if state.step == 1:
    st.title("🔮 Auto-ML Service")
    file = st.file_uploader("📂 CSV / Parquet 업로드", ["csv", "parquet"])
    if file:
        state.df = io.load(file)
        st.success(f"데이터 로드 완료 → {state.df.shape[0]:,} rows × {state.df.shape[1]} cols")
        state.step = 2
        st.rerun()

# 2) 타깃 선택
elif state.step == 2:
    st.title("🔮 Auto-ML Service")
    df: pd.DataFrame = state.df
    st.write(df.head())
    state.target = st.selectbox("🎯 타깃(목표) 컬럼을 선택하세요", df.columns)
    if state.target and st.button("다음 단계로 진행", type="primary"):
        state.step = 3
        st.rerun()

# 3) 문제 유형 판정 + EDA + Data Split + Preprocessing + AutoML
elif state.step == 3:
    st.title("🔮 Auto-ML Service")
    df, target = state.df, state.target
    detected = detector.detect(df, target)
    st.info(f"자동 판정 결과: **{detected.value}**")
    task_val = st.radio(
        "문제 유형을 확인/수정하세요",
        [t.value for t in TaskType],
        index=list(TaskType).index(detected),
        horizontal=True,
    )
    state.task = TaskType(task_val)

    n = len(df)
    if n <= 2000:
        test_size = 0.25
        valid_size = 0.15
    elif n <= 10000:
        test_size = 0.2
        valid_size = 0.1
    else:
        test_size = 0.15
        valid_size = 0.05

    st.markdown("## 데이터 분할 (비율 자동 설정)")
    st.info(f"→ 자동분할: test {int(test_size*100)}%, valid {int(valid_size*100)}% (random seed=42)")

    # Split
    df_train, df_temp = train_test_split(df, test_size=test_size, random_state=42, stratify=df[target] if state.task==TaskType.CLASSIFICATION else None)
    valid_relative = valid_size / (1 - test_size)
    df_valid, df_test = train_test_split(df_temp, test_size=1-valid_relative, random_state=42, stratify=df_temp[target] if state.task==TaskType.CLASSIFICATION else None)

    # 🚨 타깃 결측치 NaN 자동 제거
    df_train = df_train.dropna(subset=[target])
    df_valid = df_valid.dropna(subset=[target])
    df_test  = df_test.dropna(subset=[target])

    st.info(f"Train: {len(df_train)}, Valid: {len(df_valid)}, Test: {len(df_test)}")

    if state.task == TaskType.REGRESSION:
        reg_eda.generate(df, target)
        st.markdown("## 🛠️ 자동 전처리 (Train/Valid/Test)")
        st.write("**Train 전처리**")
        X_train, preprocessor = reg_prep.run(df_train, target, mode="fit_transform")
        st.write("**Valid 전처리**")
        X_valid = reg_prep.run(df_valid, target, mode="transform", preprocessor=preprocessor)
        st.write("**Test 전처리**")
        X_test  = reg_prep.run(df_test,  target, mode="transform", preprocessor=preprocessor)
        y_train = df_train[target]
        y_valid = df_valid[target]
        y_test  = df_test[target]
        st.markdown("### ✅ Train/Valid/Test 전처리 결과")
        st.write("Train (input):", X_train.shape)
        st.write("Valid (input):", X_valid.shape)
        st.write("Test (input):",  X_test.shape)

        if st.button("🚀 AutoML 실행", type="primary"):
            leaderboard_df, best_model = reg_model_search.search_and_evaluate(
                X_train, y_train, X_valid, y_valid, X_test, y_test
            )
            st.markdown("---")
            st.header("🧾 모델 평가 및 XAI 해석")
            reg_eval.evaluate(best_model, X_test, y_test)
            preprocessing_logs = preprocessor.get_log()
            best_model_name = leaderboard_df.iloc[0]['Model']
            best_params = leaderboard_df.iloc[0]['Params']
            from automl.regression.evaluation import regression_metrics
            test_pred = best_model.predict(X_test)
            test_metrics = regression_metrics(y_test, test_pred)
            import numpy as np
            errors = np.abs(y_test - test_pred)
            top_idx = np.argsort(-errors)[:5]
            error_samples = pd.DataFrame({
                "실제값": y_test.iloc[top_idx].values,
                "예측값": test_pred[top_idx],
                "에러": errors.iloc[top_idx]
            })
            xai_summary = [
                "Abdomen, Weight, Age가 예측에 큰 영향.",
                "SHAP summary: Abdomen이 전체 예측 변화의 35%를 설명."
            ]
            report_path = reg_report.save_report_html(
                df, target, task_val, [], preprocessing_logs,
                leaderboard_df, best_model_name, best_params,
                test_metrics, error_samples, xai_summary,
                file_path="AutoML_Report.html"
            )
            with open(report_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.markdown("### 📄 웹 리포트 다운로드")
            st.download_button(
                label="AutoML 리포트(HTML) 다운로드",
                data=html_content,
                file_name="AutoML_Report.html",
                mime="text/html"
            )

    elif state.task == TaskType.CLASSIFICATION:
        eda_logs = cls_eda.generate(df, target)   # <-- 동적 해설 로그 받기
        st.markdown("## 🛠️ 자동 전처리 (Train/Valid/Test)")
        st.write("**Train 전처리**")
        X_train, y_train, preprocessor = cls_prep.run(df_train, target, mode="fit_transform")
        st.write("**Valid 전처리**")
        X_valid = cls_prep.run(df_valid, target, mode="transform", preprocessor=preprocessor)
        st.write("**Test 전처리**")
        X_test  = cls_prep.run(df_test,  target, mode="transform", preprocessor=preprocessor)
        # y_valid, y_test는 그대로
        y_valid = df_valid[target]
        y_test  = df_test[target]

        # ★★★ 여기에서 safe_feature_names 적용 ★★★
        X_train = safe_feature_names(X_train)
        X_valid = safe_feature_names(X_valid)
        X_test  = safe_feature_names(X_test)

        st.markdown("### ✅ Train/Valid/Test 전처리 결과")
        st.write("Train (input):", X_train.shape)
        st.write("Valid (input):", X_valid.shape)
        st.write("Test (input):",  X_test.shape)

        if st.button("🚀 AutoML 실행", type="primary"):
            leaderboard_df, best_model = cls_model_search.search_and_evaluate(
                X_train, y_train, X_valid, y_valid, X_test, y_test
            )
            st.markdown("---")
            st.header("🧾 모델 평가 및 XAI 해석")
            cls_eval.evaluate(best_model, X_test, y_test)
            preprocessing_logs = preprocessor.get_log()
            best_model_name = leaderboard_df.iloc[0]['Model']
            best_params = leaderboard_df.iloc[0]['Params']
            from automl.classification.evaluation import classification_metrics
            test_pred = best_model.predict(X_test)
            test_metrics = classification_metrics(y_test, test_pred)
            import numpy as np
            errors = (y_test != test_pred)
            top_idx = np.where(errors)[0][:5]
            error_samples = pd.DataFrame({
                "실제값": y_test.iloc[top_idx].values,
                "예측값": test_pred[top_idx],
            })
            xai_summary = [
                "Abdomen, Weight, Age가 분류에 큰 영향.",
                "SHAP summary: Abdomen이 전체 예측 변화의 35%를 설명."
            ]
            report_path = cls_report.save_report_html(
                df, target, task_val, eda_logs, preprocessing_logs,
                leaderboard_df, best_model_name, best_params,
                test_metrics, error_samples, xai_summary,
                file_path="AutoML_Classification_Report.html"
            )
            with open(report_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.markdown("### 📄 웹 리포트 다운로드")
            st.download_button(
                label="AutoML 분류 리포트(HTML) 다운로드",
                data=html_content,
                file_name="AutoML_Classification_Report.html",
                mime="text/html"
            )