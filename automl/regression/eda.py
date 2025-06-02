# automl/regression/eda.py
"""
고도화된 Regression EDA
-----------------------
각 EDA 섹션별로 결과 기반 동적 해설(자동 해석) 추가
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

try:
    import ydata_profiling
except ImportError:
    ydata_profiling = None
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
except ImportError:
    IsolationForest = LocalOutlierFactor = None
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
except ImportError:
    vif = None

_SAMPLE_SIZE = 10000

@st.cache_data(show_spinner=False)
def sample_df(df: pd.DataFrame, max_rows=_SAMPLE_SIZE) -> pd.DataFrame:
    if len(df) > max_rows:
        return df.sample(n=max_rows, random_state=42)
    return df.copy()

def _overview(df: pd.DataFrame):
    st.subheader("0️⃣ 데이터 개요")
    st.info("데이터의 전체 구조, 변수 종류, 용량을 한눈에 확인합니다. 이상치/결측 탐색이나 전처리 범위 파악에 중요합니다.")
    col1, col2 = st.columns(2)
    with col1:
        st.write("행/열:", df.shape)
        st.write("데이터 타입:", dict(df.dtypes.value_counts()))
        st.write("메모리 사용량:", f"{df.memory_usage(deep=True).sum()/1e6:.2f} MB")
    with col2:
        st.write(df.head(5))
    st.caption(f"이 데이터는 {df.shape[0]:,}개의 행과 {df.shape[1]}개의 컬럼으로 구성되어 있습니다. "
               f"{', '.join([f'{k}: {v}' for k, v in dict(df.dtypes.value_counts()).items()])} 타입 변수들이 존재합니다.")

def _missing_and_duplicates(df: pd.DataFrame):
    st.subheader("1️⃣ 결측 & 중복")
    st.info("결측치는 전처리 전략(삭제/대치) 설계, 중복 데이터는 모델링 편향 방지에 꼭 체크해야 합니다.")
    miss = df.isna().mean().sort_values(ascending=False)
    dup_cnt = df.duplicated().sum()
    if miss.max() == 0:
        st.success("✅ 결측치가 없습니다.")
        st.caption("→ 추가 전처리 없이 바로 모델링이 가능합니다.")
    else:
        fig, ax = plt.subplots(figsize=(6, 3))
        miss[miss > 0].plot.bar(ax=ax)
        ax.set_ylabel("Missing Ratio")
        st.pyplot(fig, use_container_width=True)
        # 동적 해설
        most_missing = miss[miss > 0].head(3)
        txt = ", ".join([f"{col} ({ratio:.1%})" for col, ratio in most_missing.items()])
        st.warning(f"결측치 상위 컬럼: {txt}")
        if most_missing.iloc[0] > 0.5:
            st.caption(f"• '{most_missing.index[0]}'는 결측이 50%를 넘어 삭제를 고려하세요.")
        elif most_missing.iloc[0] > 0.1:
            st.caption("• 결측 10~50%: 대치(imputation) 또는 피처 엔지니어링을 권장합니다.")
        else:
            st.caption("• 결측 10% 이하: 평균/중앙값/최빈값 등으로 손쉽게 대치 가능합니다.")
    st.write("🔁 중복 행:", dup_cnt)
    if dup_cnt > 0:
        st.caption(f"• 중복 데이터 {dup_cnt}건: 데이터 정합성 확보를 위해 제거 권장")
    else:
        st.caption("• 중복 데이터 없음")

def _stats_and_target_dist(df: pd.DataFrame, target: str):
    st.subheader("2️⃣ 기술통계 & 타깃 분포")
    st.info("기술통계로 데이터의 분포/이상값/대표값을 파악하고, 타깃의 분포가 비대칭/뾰족한지 확인합니다. 예측모델 선택(예: 로그변환)에도 영향.")
    st.write(df.describe().T)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df[target].dropna(), kde=True, ax=ax)
        ax.set_title(f"Target Distribution – {target}")
        st.pyplot(fig, use_container_width=True)
    with col2:
        sk = stats.skew(df[target].dropna())
        kt = stats.kurtosis(df[target].dropna())
        st.write(f"Skewness: {sk:.2f} &nbsp;&nbsp; Kurtosis: {kt:.2f}")
        msgs = []
        if abs(sk) > 1:
            msgs.append("⚠️ 타깃이 비대칭(skew>1). 로그/Box-Cox 변환 추천.")
        if abs(kt) > 3:
            msgs.append("⚠️ 뾰족한 분포(kurtosis>3). 이상치가 많을 수 있습니다.")
        if not msgs:
            st.success("타깃 분포가 비교적 정규분포에 가깝습니다.")
        else:
            for m in msgs:
                st.warning(m)

def _categorical_eda(df: pd.DataFrame, target: str):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if not len(cat_cols):
        return
    st.subheader("3️⃣ 범주형 변수 분석")
    st.info("범주형 변수(예: 지역, 카테고리)별로 타깃의 평균·분포를 시각화합니다. high-cardinality 변수는 엔코딩 시 주의 필요.")
    for col in cat_cols:
        st.write(f"##### {col} (유니크: {df[col].nunique()})")
        vc = df[col].value_counts().head(30)
        fig, ax = plt.subplots()
        vc.plot.bar(ax=ax)
        ax.set_ylabel("Count")
        st.pyplot(fig, use_container_width=True)
        grp = df.groupby(col)[target].agg(['mean', 'count']).sort_values('mean')
        st.write("타깃 평균 상위/하위 (Top10):")
        st.dataframe(grp.head(10).style.background_gradient('Blues', subset=['mean']), use_container_width=True)
        # 동적 해설
        if grp['count'].max() > 0.5 * len(df):
            st.caption(f"• '{col}'의 일부 값이 데이터의 절반 이상을 차지합니다. 균형 불균형 여부 체크 필요.")
        if df[col].nunique() > 20:
            st.caption(f"• '{col}' 변수는 고유값이 {df[col].nunique()}개로 많아 고차원 원-핫인코딩에 주의.")
        # Boxplot
        if df[col].nunique() < 20:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], y=df[target], ax=ax)
            ax.set_title(f"{target} by {col}")
            st.pyplot(fig, use_container_width=True)

    hc_cols = [c for c in cat_cols if df[c].nunique() > 20]
    for col in hc_cols:
        st.write(f"High-cardinality 변수(Top 20): {col}")
        top_vals = df[col].value_counts().head(20)
        fig, ax = plt.subplots()
        top_vals.plot.bar(ax=ax)
        st.pyplot(fig, use_container_width=True)

def _outlier_overview(df: pd.DataFrame, target: str):
    num_cols = df.select_dtypes(include="number").columns
    if not num_cols.any():
        return
    st.subheader("4️⃣ 이상치 탐색")
    st.info("여러 방법으로 이상치를 탐지합니다. 이상치는 모델의 성능 저하·불안정 원인이 될 수 있어 반드시 체크해야 합니다.")
    sample = sample_df(df[num_cols])
    # IQR
    outlier_ratio = {}
    for col in num_cols:
        q1, q3 = np.percentile(sample[col].dropna(), [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (sample[col] < lower) | (sample[col] > upper)
        outlier_ratio[col] = mask.mean()
    sr = pd.Series(outlier_ratio).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 3))
    sr.plot.bar(ax=ax)
    ax.set_ylabel("IQR Outlier Ratio")
    st.pyplot(fig, use_container_width=True)
    # 동적 해설
    high_outlier = sr[sr > 0.05]
    if not high_outlier.empty:
        st.warning("이상치 비율이 5%를 넘는 변수: " +
            ', '.join([f"{col}({ratio:.1%})" for col, ratio in high_outlier.items()]))
    else:
        st.success("모든 변수의 이상치(IQR 기준)가 5% 미만입니다.")

    # Z-score
    st.write("**Z-score 기준 이상치 탐색 (|z| > 3):**")
    zscore_outlier = ((np.abs(stats.zscore(sample)) > 3).sum(axis=0) / len(sample))
    st.write(zscore_outlier)
    z_excess = zscore_outlier[zscore_outlier > 0.05]
    if not z_excess.empty:
        st.caption("Z-score 기준 5% 이상 이상치 변수: " + ', '.join(z_excess.index))

    # Isolation Forest
    if IsolationForest is not None:
        st.write("**Isolation Forest로 탐지된 이상치 비율:**")
        iso = IsolationForest(n_estimators=50, contamination=0.01, random_state=42)
        y_pred = iso.fit_predict(sample)
        outlier_rate = (y_pred == -1).mean()
        st.write(f"{outlier_rate:.2%}")
        if outlier_rate > 0.05:
            st.caption("Isolation Forest로 전체 이상치가 5%를 넘으므로 robust loss/model 또는 이상치 제거를 고려하세요.")
    else:
        st.info("IsolationForest 설치: pip install scikit-learn")

    # LOF
    if LocalOutlierFactor is not None:
        st.write("**LOF(Neighbor 기반)로 탐지된 이상치 비율:**")
        try:
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
            y_pred = lof.fit_predict(sample)
            outlier_rate = (y_pred == -1).mean()
            st.write(f"{outlier_rate:.2%}")
            if outlier_rate > 0.05:
                st.caption("LOF로 전체 이상치가 5%를 넘으므로 변수 scaling 또는 이상치 영향 완화 필요.")
        except Exception:
            st.info("LOF는 20개 이상의 샘플에서만 동작합니다.")
    else:
        st.info("LOF 설치: pip install scikit-learn")

def _multicollinearity_vif(df: pd.DataFrame):
    num_df = df.select_dtypes(include="number").dropna()
    if vif is None or num_df.shape[1] < 2:
        return
    st.subheader("5️⃣ 다중공선성(VIF)")
    st.info("입력 피처들끼리 너무 강한 상관(공선성)이 있으면 회귀계수 해석·일반화 성능이 나빠집니다. VIF > 10 변수는 주의!")
    sample = sample_df(num_df)
    X = sample.values
    vif_vals = [vif(X, i) for i in range(X.shape[1])]
    res = pd.DataFrame({
        "feature": num_df.columns,
        "VIF": vif_vals
    }).sort_values("VIF", ascending=False)
    st.dataframe(res, use_container_width=True)
    high_vif = res[res["VIF"] > 10]
    if not high_vif.empty:
        st.warning(
            "다중공선성이 강한 변수(VIF>10): " +
            ', '.join(high_vif['feature'].tolist()) +
            ". 해당 변수들은 회귀 해석력 저하·분산팽창 위험이 있으므로 제거/축소 권장!"
        )
    else:
        st.success("모든 변수의 VIF < 10. 다중공선성 우려가 없습니다.")

def _correlation(df: pd.DataFrame, target: str):
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        return
    st.subheader("6️⃣ 상관 행렬(Pearson)")
    st.info("수치형 변수간 상관관계와 타깃과의 연관성을 파악합니다. 상관 |r| > 0.8은 다중공선성 우려, 의미있는 feature engineering 힌트도 얻을 수 있습니다.")
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(min(12, 0.5 * corr.shape[1]), 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig, use_container_width=True)
    # multicollinearity hint
    high_pairs = [
        (i, j)
        for i in corr.columns
        for j in corr.columns
        if i != j and abs(corr.loc[i, j]) > 0.8
    ]
    if high_pairs:
        st.warning("• |r| > .8 변수쌍: " + ", ".join([f"{a}-{b}" for a, b in high_pairs]))
    else:
        st.success("모든 변수 쌍에서 |r| > .8 이상인 경우가 없습니다.")

def _scatter_top(df: pd.DataFrame, target: str, top_k=5):
    num_df = df.select_dtypes(include="number").drop(columns=[target], errors="ignore")
    if num_df.empty:
        return
    corrs = num_df.corrwith(df[target]).abs().sort_values(ascending=False)
    top_cols = corrs.head(top_k).index
    st.subheader(f"7️⃣ 타깃과 상관 상위 {top_k}개 피처")
    st.info("타깃과 상관관계가 가장 높은 피처와의 관계를 산점도로 시각화합니다. 비선형성/특이값 등 해석에도 유용합니다.")
    for col in top_cols:
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[col], y=df[target], ax=ax)
        ax.set_title(f"{col} vs {target}  (|r|={corrs[col]:.2f})")
        st.pyplot(fig, use_container_width=True)
        # 동적 해설
        if corrs[col] > 0.8:
            st.caption(f"{col}과(와) 타깃은 매우 강한 선형 상관관계(|r|={corrs[col]:.2f})를 보입니다.")
        elif corrs[col] > 0.5:
            st.caption(f"{col}과(와) 타깃은 중간 이상의 상관관계(|r|={corrs[col]:.2f})가 있습니다.")
        else:
            st.caption(f"{col}과(와) 타깃의 선형 상관은 약합니다(|r|={corrs[col]:.2f}).")

def _full_profile(df: pd.DataFrame):
    with st.expander("🔍 전체 Profiling Report (optional)", expanded=False):
        st.info("전체 변수·관계·분포를 일괄 진단합니다. 큰 데이터는 시간이 오래 걸릴 수 있습니다.")
        if ydata_profiling:
            profile = ydata_profiling.ProfileReport(df, title="Profiling", minimal=True)
            st.components.v1.html(profile.to_html(), height=800, scrolling=True)
        else:
            st.info("`pip install ydata-profiling` 후 전체 리포트를 볼 수 있습니다.")

def generate(df: pd.DataFrame, target: str):
    """Streamlit 화면에 완전 자동 EDA + 각 결과별 동적 해설"""
    df = sample_df(df)
    _overview(df)
    _missing_and_duplicates(df)
    _stats_and_target_dist(df, target)
    _categorical_eda(df, target)
    _outlier_overview(df, target)
    _multicollinearity_vif(df)
    _correlation(df, target)
    _scatter_top(df, target)
    _full_profile(df)