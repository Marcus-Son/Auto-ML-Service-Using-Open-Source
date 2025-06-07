# automl/timeseries/eda.py

"""
고도화된 Time Series EDA (AutoML 스타일)
--------------------------------------
- 시계열 인덱스/타깃 진단, 정상성/추세/계절성/이상치 자동 탐지
- 결측·중복, feature별 상관/Autocorr, 변동성 진단
- 피크/트렌드/패턴 시각화, 동적 해설/경고, 전체 로그 반환
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

try:
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.tsa.seasonal import STL
except ImportError:
    adfuller = acf = pacf = STL = None

_SAMPLE_SIZE = 10000

@st.cache_data(show_spinner=False)
def sample_df(df: pd.DataFrame, max_rows=_SAMPLE_SIZE) -> pd.DataFrame:
    if len(df) > max_rows:
        return df.sample(n=max_rows, random_state=42)
    return df.copy()

def _overview(df: pd.DataFrame, time_col: str, eda_logs: list):
    st.subheader("0️⃣ 데이터 개요")
    st.info("시계열 인덱스와 타깃 분포, 전체 구조 확인. 정상성·추세 등 후속 분석의 기준이 됨.")
    st.write(f"총 {df.shape[0]:,} 행 × {df.shape[1]} 컬럼")
    st.write("Datetime Index:", pd.api.types.is_datetime64_any_dtype(df[time_col]))
    st.write(df.head())
    eda_logs.append(f"총 {df.shape[0]:,}개 샘플, 시계열 컬럼({time_col})이 datetime 타입: {pd.api.types.is_datetime64_any_dtype(df[time_col])}")

def _missing_and_duplicates(df: pd.DataFrame, time_col: str, eda_logs: list):
    st.subheader("1️⃣ 결측 & 중복")
    miss = df.isna().mean().sort_values(ascending=False)
    miss_cols = miss[miss > 0]
    dup_cnt = df.duplicated(subset=[time_col]).sum()
    if miss_cols.empty:
        st.success("✅ 결측치 없음")
        eda_logs.append("결측치 없음")
    else:
        st.warning("결측치 있는 컬럼: " + ", ".join([f"{col}({ratio:.1%})" for col, ratio in miss_cols.items()]))
        st.write(miss_cols)
        eda_logs.append("결측치 있는 컬럼: " + ", ".join([f"{col}({ratio:.1%})" for col, ratio in miss_cols.items()]))
    st.write("중복 Timestamp 행:", dup_cnt)
    if dup_cnt > 0:
        st.caption("→ Timestamp 중복: 시계열 데이터 정합성에 문제. 집계/중복제거 필요.")
        eda_logs.append(f"중복 timestamp: {dup_cnt}개 존재")
    else:
        eda_logs.append("Timestamp 중복 없음")

def _target_dist(df: pd.DataFrame, target: str, eda_logs: list):
    st.subheader("2️⃣ 타깃(목표값) 분포")
    # 수치형 강제 변환
    df[target] = pd.to_numeric(df[target], errors="coerce")
    # 변환 후 NaN 체크
    if df[target].dropna().empty:
        st.error(f"[{target}] 컬럼이 모두 NaN이거나 변환에 실패했습니다. 실제 값 예시: {df[target].head().to_list()}")
        eda_logs.append(f"타깃({target})이 변환 후에도 NaN")
        st.stop()
    fig, ax = plt.subplots()
    sns.histplot(df[target].dropna(), kde=True, ax=ax)
    ax.set_title(f"Target Distribution – {target}")
    st.pyplot(fig, use_container_width=True)
    sk = stats.skew(df[target].dropna())
    kt = stats.kurtosis(df[target].dropna())
    st.write(f"Skewness: {sk:.2f}   Kurtosis: {kt:.2f}")
    msgs = []
    if abs(sk) > 1:
        msgs.append("⚠️ 타깃이 비대칭. 로그변환/정규화 고려.")
    if abs(kt) > 3:
        msgs.append("⚠️ 뾰족한 분포. 이상치 많을 수 있음.")
    if not msgs:
        st.success("타깃 분포가 비교적 정규분포에 가까움.")
    else:
        for m in msgs:
            st.warning(m)
    eda_logs.append(f"타깃 분포 – Skewness: {sk:.2f}, Kurtosis: {kt:.2f}")

def _trend_seasonality(df: pd.DataFrame, time_col: str, target: str, eda_logs: list):
    st.subheader("3️⃣ 추세/계절성 분해(STL)")
    if STL is None:
        st.info("statsmodels 설치 필요 (pip install statsmodels)")
        eda_logs.append("STL 계절성 분해: statsmodels 설치 필요")
        return
    # 수치형 변환
    df[target] = pd.to_numeric(df[target], errors="coerce")
    s_df = df[[time_col, target]].dropna().copy()
    s_df = s_df.sort_values(time_col)
    s_df.set_index(time_col, inplace=True)
    try:
        period = pd.infer_freq(s_df.index)
        if period is None:
            period = max(2, int(len(s_df) / 20))
        stl = STL(s_df[target], period=period if isinstance(period, int) else 7)
        res = stl.fit()
        fig, axs = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
        axs[0].plot(s_df[target])
        axs[0].set_title("원본")
        axs[1].plot(res.trend)
        axs[1].set_title("추세(Trend)")
        axs[2].plot(res.seasonal)
        axs[2].set_title("계절성(Seasonality)")
        axs[3].plot(res.resid)
        axs[3].set_title("잔차(Residual)")
        st.pyplot(fig, use_container_width=True)
        st.write("추세/계절성/잔차 분해 완료")
        eda_logs.append("STL로 추세/계절성/잔차 분해 성공")
    except Exception as e:
        st.warning(f"STL 분해 실패: {e}")
        eda_logs.append(f"STL 분해 실패: {e}")

def _stationarity(df: pd.DataFrame, time_col: str, target: str, eda_logs: list):
    st.subheader("4️⃣ 정상성(Stationarity) 진단")
    if adfuller is None:
        st.info("statsmodels 설치 필요 (pip install statsmodels)")
        eda_logs.append("정상성 진단: statsmodels 설치 필요")
        return
    # 수치형 변환
    df[target] = pd.to_numeric(df[target], errors="coerce")
    y = df[target].dropna()
    if y.empty:
        st.error(f"[{target}] 컬럼이 모두 NaN이거나 변환에 실패했습니다.")
        eda_logs.append(f"타깃({target})이 변환 후에도 NaN")
        st.stop()
    result = adfuller(y)
    p = result[1]
    st.write(f"ADF test p-value: {p:.4f}")
    if p < 0.05:
        st.success("정상성(Stationary) 데이터: 시계열 모델(ARIMA 등) 적용 가능.")
        eda_logs.append(f"ADF p-value={p:.4f}: 정상성 통과")
    else:
        st.warning("비정상성(Non-stationary) 데이터: 차분/변환 필요. (p>0.05)")
        eda_logs.append(f"ADF p-value={p:.4f}: 비정상성 (차분/변환 필요)")

def _autocorr(df: pd.DataFrame, target: str, eda_logs: list):
    st.subheader("5️⃣ 자기상관(Autocorrelation) 분석")
    if acf is None:
        st.info("statsmodels 설치 필요 (pip install statsmodels)")
        eda_logs.append("자기상관 분석: statsmodels 설치 필요")
        return
    # 수치형 변환
    df[target] = pd.to_numeric(df[target], errors="coerce")
    y = df[target].dropna()
    if y.empty:
        st.error(f"[{target}] 컬럼이 모두 NaN이거나 변환에 실패했습니다.")
        eda_logs.append(f"타깃({target})이 변환 후에도 NaN")
        st.stop()
    n_lag = min(40, max(10, int(len(y)/20)))
    acf_vals = acf(y, nlags=n_lag)
    fig, ax = plt.subplots()
    ax.bar(range(len(acf_vals)), acf_vals)
    ax.set_title("ACF (자기상관)")
    st.pyplot(fig, use_container_width=True)
    eda_logs.append(f"ACF(자기상관) 분석 수행 (nlags={n_lag})")

def _outlier_overview(df: pd.DataFrame, target: str, eda_logs: list):
    st.subheader("6️⃣ 이상치 탐색")
    # 결측치 row도 길이 맞추기 위해 직접 계산
    y = pd.to_numeric(df[target], errors="coerce")
    # IQR 계산 (결측치 제외)
    y_no_na = y.dropna()
    q1, q3 = np.percentile(y_no_na, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    # 전체 행에 대해 True/False, 결측치면 False
    out_idx = (y < lower) | (y > upper)
    out_idx = out_idx.fillna(False)  # 결측치는 이상치 아님 처리
    st.write(f"이상치 비율: {(out_idx.mean() * 100):.2f}%")
    if out_idx.mean() > 0.05:
        st.warning("이상치 비율 5%↑: robust scaler/모델 추천")
        eda_logs.append(f"이상치 비율 {out_idx.mean():.2%}")
    else:
        st.success("이상치 비율 5% 미만")
        eda_logs.append("이상치 비율 5% 미만")
    st.write("상위 5개 이상치 샘플:")
    st.write(df.loc[out_idx].head())

def _feature_corr(df: pd.DataFrame, target: str, eda_logs: list):
    df[target] = pd.to_numeric(df[target], errors="coerce")
    num_cols = [c for c in df.select_dtypes(include="number").columns if c != target]
    if not num_cols: return
    st.subheader("7️⃣ Feature별 타깃 상관관계")
    corr = df[num_cols + [target]].corr()[target][num_cols]
    top_corr = corr.abs().sort_values(ascending=False).head(5)
    st.write(top_corr)
    eda_logs.append("상위 상관 feature: " + ", ".join(top_corr.index))

def _seasonal_plot(df: pd.DataFrame, time_col: str, target: str, eda_logs: list):
    st.subheader("8️⃣ 월/요일/시즌별 평균")
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        st.info("Datetime 타입 컬럼 필요")
        eda_logs.append("Datetime 컬럼 아님")
        return
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df['month'] = df[time_col].dt.month
    df['weekday'] = df[time_col].dt.dayofweek
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    try:
        df.groupby('month')[target].mean().plot(ax=ax[0], title='월별 평균')
        df.groupby('weekday')[target].mean().plot(ax=ax[1], title='요일별 평균')
        st.pyplot(fig, use_container_width=True)
        eda_logs.append("월/요일별 평균 시각화")
    except Exception as e:
        st.warning(f"월/요일별 평균 시각화 실패: {e}")
        eda_logs.append(f"월/요일별 평균 시각화 실패: {e}")

def _full_profile(df: pd.DataFrame, eda_logs: list):
    with st.expander("🔍 전체 Profiling Report (optional)", expanded=False):
        st.info("전체 변수·관계·분포 일괄 진단(대용량 데이터는 시간이 오래 걸릴 수 있음)")
        try:
            import ydata_profiling
            profile = ydata_profiling.ProfileReport(df, title="Profiling", minimal=True)
            st.components.v1.html(profile.to_html(), height=800, scrolling=True)
            eda_logs.append("ydata_profiling 리포트 생성")
        except ImportError:
            st.info("`pip install ydata-profiling` 필요")
            eda_logs.append("ydata_profiling 설치 필요")

def generate(df: pd.DataFrame, time_col: str, target: str):
    """Streamlit 자동 EDA + 동적 해설 로그 반환"""
    eda_logs = []
    df = sample_df(df)
    _overview(df, time_col, eda_logs)
    _missing_and_duplicates(df, time_col, eda_logs)
    _target_dist(df, target, eda_logs)
    _trend_seasonality(df, time_col, target, eda_logs)
    _stationarity(df, time_col, target, eda_logs)
    _autocorr(df, target, eda_logs)
    _outlier_overview(df, target, eda_logs)
    _feature_corr(df, target, eda_logs)
    _seasonal_plot(df, time_col, target, eda_logs)
    _full_profile(df, eda_logs)
    return eda_logs