from . import eda, preprocessing, model_search, evaluation, report
import streamlit as st


def run(df, target):
    st.header("🔧 [Regression] 파이프라인")
    # EDA
    with st.expander("EDA", expanded=False):
        eda.generate(df, target)

    # Preprocessing
    X_train, X_test, y_train, y_test, pipe = preprocessing.build(df, target)

    # Model Search
    best_model, lb = model_search.search(X_train, y_train)

    # Evaluation
    y_pred = best_model.predict(X_test)
    metrics = evaluation.calc(y_test, y_pred)

    st.subheader("📊 성능 요약")
    st.write(metrics)

    # Report Download
    rep_bytes = report.create(df, target, lb, metrics)
    st.download_button("리포트 다운로드", rep_bytes, file_name="regression_report.html")
