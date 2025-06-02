# app.py ─ 단계별 UI (버튼 최소화) + Regression EDA
import streamlit as st
import pandas as pd
from automl import io, detector, TaskType
from automl.regression import run as run_reg
from automl.regression import eda as reg_eda
# from automl.classification import run as run_clf
# from automl.timeseries import run as run_ts

st.set_page_config(page_title="Auto-ML Service", page_icon="🔮")

# ───────────────── 세션 상태 초기화 ───────────────── #
state = st.session_state
if "step" not in state:
    state.step, state.df, state.target, state.task = 1, None, None, None

# ==================== 1) 파일 업로드 ==================== #
if state.step == 1:
    st.title("🔮 Auto-ML Service")
    file = st.file_uploader("📂 CSV / Parquet 업로드", ["csv", "parquet"])
    if file:
        state.df = io.load(file)
        st.success(f"데이터 로드 완료 → {state.df.shape[0]:,} rows × {state.df.shape[1]} cols")
        state.step = 2
        st.rerun()

# ==================== 2) 타깃 선택 ==================== #
elif state.step == 2:
    st.title("🔮 Auto-ML Service")
    df: pd.DataFrame = state.df
    st.write(df.head())           # 미리보기

    state.target = st.selectbox("🎯 타깃(목표) 컬럼을 선택하세요", df.columns)
    if state.target:
        state.step = 3
        st.rerun()

# ==================== 3) 문제 유형 판정 + EDA + AutoML ==================== #
elif state.step == 3:
    st.title("🔮 Auto-ML Service")

    df, target = state.df, state.target
    detected = detector.detect(df, target)
    st.info(f"자동 판정 결과: **{detected.value}**")

    # 필요하면 사용자가 수정 가능
    task_val = st.radio(
        "문제 유형을 확인/수정하세요",
        [t.value for t in TaskType],
        index=list(TaskType).index(detected),
        horizontal=True,
    )
    state.task = TaskType(task_val)

    # ---------- 회귀라면 즉시 EDA 자동 출력 ----------
    if state.task == TaskType.REGRESSION:
        reg_eda.generate(df, target)

    st.markdown("---")
    if st.button("🚀 AutoML 실행", type="primary"):
        if state.task == TaskType.REGRESSION:
            run_reg(df, target)
        # elif state.task == TaskType.CLASSIFICATION:
        #     run_clf(df, target)
        # else:
        #     run_ts(df, target)