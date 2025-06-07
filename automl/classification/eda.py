"""
고도화된 Classification EDA (상업 AutoML 수준)
------------------------------------------
- 클래스 분포/imbalance, 클래스별 통계·분포차 검정, 주요 변수 중요도/해석, 이상치, 다중공선성, 차원축소, 동적 해설/경고
- t-SNE/UMAP, violin/swarm/stacked bar, chi2/ANOVA, mutual_info 등 풀세트
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2_contingency

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
    from sklearn.manifold import TSNE
except ImportError:
    TSNE = None
try:
    from umap import UMAP
except ImportError:
    UMAP = None
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
except ImportError:
    vif = None

from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from imblearn.over_sampling import SMOTE

_SAMPLE_SIZE = 10000

@st.cache_data(show_spinner=False)
def sample_df(df: pd.DataFrame, max_rows=_SAMPLE_SIZE) -> pd.DataFrame:
    if len(df) > max_rows:
        return df.sample(n=max_rows, random_state=42)
    return df.copy()

def _overview(df: pd.DataFrame, eda_logs: list):
    st.subheader("0️⃣ 데이터 개요")
    col1, col2 = st.columns(2)
    with col1:
        st.write("행/열:", df.shape)
        st.write("데이터 타입:", dict(df.dtypes.value_counts()))
        st.write("메모리 사용량:", f"{df.memory_usage(deep=True).sum()/1e6:.2f} MB")
    with col2:
        st.write(df.head(5))
    msg = f"데이터: {df.shape[0]:,}개 행, {df.shape[1]}개 컬럼, {dict(df.dtypes.value_counts())} 타입."
    st.caption(msg)
    eda_logs.append(msg)

def _missing_and_duplicates(df: pd.DataFrame, target: str, eda_logs: list):
    st.subheader("1️⃣ 결측 & 중복 (클래스별도 표시)")
    miss = df.isna().mean().sort_values(ascending=False)
    miss_by_class = df.groupby(target).apply(lambda x: x.isna().mean())
    dup_cnt = df.duplicated().sum()
    if miss.max() == 0:
        msg = "✅ 결측치 없음."
        st.success(msg)
        eda_logs.append(msg)
    else:
        fig, ax = plt.subplots(figsize=(6, 3))
        miss[miss > 0].plot.bar(ax=ax)
        ax.set_ylabel("Missing Ratio")
        st.pyplot(fig, use_container_width=True)
        st.write("클래스별 결측률(상위 5):")
        st.dataframe(miss_by_class.T.head(5))
        msg = f"결측치 상위 컬럼: {', '.join(miss[miss>0].index[:3])}"
        eda_logs.append(msg)
    st.write("🔁 중복 행:", dup_cnt)
    if dup_cnt > 0:
        cap = f"중복 데이터 {dup_cnt}건: 제거 권장"
        st.caption(cap)
        eda_logs.append(cap)
    else:
        eda_logs.append("중복 데이터 없음.")

def _target_dist(df: pd.DataFrame, target: str, eda_logs: list):
    st.subheader("2️⃣ 클래스 분포 (Imbalance 자동 경고·대응)")
    vc = df[target].value_counts(dropna=False)
    fig, ax = plt.subplots()
    vc.plot.bar(ax=ax, color=sns.color_palette('tab10', len(vc)))
    ax.set_ylabel("Count")
    ax.set_title("Target Class Distribution")
    st.pyplot(fig, use_container_width=True)
    st.write(vc.to_frame("Count").assign(비율=lambda x: x["Count"]/x["Count"].sum()).T)
    imbalance_ratio = vc.max() / vc.min() if len(vc) > 1 and vc.min() > 0 else 1.0
    if imbalance_ratio > 4:
        msg = f"클래스 imbalance 심각 (최다/최소 비율: {imbalance_ratio:.1f}) → SMOTE 등 샘플링/가중치 필요"
        st.error(msg)
        eda_logs.append(msg)
    elif imbalance_ratio > 1.5:
        msg = f"클래스 imbalance 주의 (최다/최소 비율: {imbalance_ratio:.1f})"
        st.warning(msg)
        eda_logs.append(msg)
    else:
        msg = "클래스 분포 균형 양호"
        st.success(msg)
        eda_logs.append(msg)

def _classwise_stats(df: pd.DataFrame, target: str, eda_logs: list):
    st.subheader("3️⃣ 클래스별 기술통계/대표값")
    classes = df[target].dropna().unique()
    num_cols = df.select_dtypes(include="number").columns
    stats_df = df.groupby(target)[num_cols].agg(['mean', 'std', 'min', 'max'])
    st.dataframe(stats_df)
    msg = f"클래스별 수치형 대표값, 차이 확인: 주요 피처 {list(num_cols)[:5]}"
    eda_logs.append(msg)
    # 클래스별 대표 샘플
    for cls in classes:
        cls_df = df[df[target]==cls]
        median_vals = cls_df[num_cols].median()
        closest = ((cls_df[num_cols] - median_vals).abs().sum(axis=1)).idxmin()
        st.write(f"Class {cls} 대표 샘플:")
        st.write(cls_df.loc[[closest]])

def _categorical_eda(df: pd.DataFrame, target: str, eda_logs=None):
    cat_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col != target]
    if not cat_cols: return
    st.subheader("4️⃣ 주요 범주형 변수 분포/클래스별 차이(카이제곱)")
    for col in cat_cols:
        fig, ax = plt.subplots()
        ct = pd.crosstab(df[col], df[target])
        ct.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title(f"{col} by {target}")
        st.pyplot(fig, use_container_width=True)
        st.write(ct)
        # chi2 독립성 검정
        chi2_val, p, dof, expected = chi2_contingency(ct.values)
        if isinstance(p, (np.ndarray, list)):  # p가 배열이면 첫 번째 값만 사용
            p_val = float(np.asarray(p).flat[0])
        else:
            p_val = float(p)
        if p_val < 0.05:
            st.warning(f"{col}: 클래스별 분포 차이 유의함 (chi2, p={p_val:.4f})")
            if eda_logs is not None:
                eda_logs.append(f"{col}: 클래스별 분포 차이 유의 (chi2, p={p_val:.4f})")
        else:
            st.info(f"{col}: 분포 차이 통계적으로 유의미하지 않음 (p={p_val:.4f})")
            if eda_logs is not None:
                eda_logs.append(f"{col}: 분포 차이 통계적으로 유의하지 않음 (chi2, p={p_val:.4f})")

def _num_feature_by_class(df: pd.DataFrame, target: str, eda_logs: list, top_k=5):
    st.subheader("5️⃣ 주요 수치형 피처별 클래스간 분포/차이(ANOVA/violin/swarm)")
    num_cols = [c for c in df.select_dtypes(include='number').columns if c != target]
    if len(num_cols) == 0:
        return
    scores, pvals = f_classif(df[num_cols].fillna(0), df[target])
    imp = pd.Series(scores, index=num_cols)
    top_cols = imp.sort_values(ascending=False).head(top_k).index
    for col in top_cols:
        fig, ax = plt.subplots()
        sns.violinplot(x=df[target], y=df[col], ax=ax, inner="quartile", palette="pastel")
        sns.swarmplot(x=df[target], y=df[col], color="k", size=2, ax=ax)
        ax.set_title(f"{col} by {target} (F={imp[col]:.2f}, p={pvals[list(num_cols).index(col)]:.4f})")
        st.pyplot(fig, use_container_width=True)
        if pvals[list(num_cols).index(col)] < 0.05:
            msg = f"{col}: 클래스간 분포 차이 유의 (ANOVA p={pvals[list(num_cols).index(col)]:.4f})"
            st.warning(msg)
        else:
            msg = f"{col}: 분포 차이 유의하지 않음 (p={pvals[list(num_cols).index(col)]:.4f})"
            st.info(msg)
        eda_logs.append(msg)

def _feature_importance(df: pd.DataFrame, target: str, eda_logs: list, top_k=10):
    st.subheader("6️⃣ Feature Importance Preview (정보이득 기반)")
    num_cols = [c for c in df.select_dtypes(include='number').columns if c != target]
    if len(num_cols) == 0: return
    X = df[num_cols].fillna(0)
    y = df[target]
    mi = mutual_info_classif(X, y, discrete_features=False)
    imp = pd.Series(mi, index=num_cols).sort_values(ascending=False)
    fig, ax = plt.subplots()
    imp.head(top_k).plot.bar(ax=ax)
    ax.set_title("Mutual Information Top Features")
    st.pyplot(fig, use_container_width=True)
    msg = "상위 중요 변수: " + ", ".join(imp.head(top_k).index)
    st.write(msg)
    eda_logs.append(msg)

def _outlier_overview(df: pd.DataFrame, target: str, eda_logs: list):
    num_cols = df.select_dtypes(include="number").columns
    if not num_cols.any():
        return
    st.subheader("7️⃣ 이상치 탐색 (수치형 기준)")
    sample = sample_df(df[num_cols])
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
    high_outlier = sr[sr > 0.05]
    if not high_outlier.empty:
        msg = "이상치 비율 5%↑: " + ', '.join([f"{col}({ratio:.1%})" for col, ratio in high_outlier.items()])
        st.warning(msg)
    else:
        msg = "모든 변수 이상치 비율 5% 미만"
        st.success(msg)
    eda_logs.append(msg)
    # 이상치 많은 클래스/샘플 하이라이트
    st.write("이상치 Top 5 샘플:")
    z = np.abs(stats.zscore(df[num_cols].fillna(0)))
    outlier_idx = np.where((z > 3).any(axis=1))[0][:5]
    st.write(df.iloc[outlier_idx])

def _multicollinearity_vif(df: pd.DataFrame, eda_logs: list):
    num_df = df.select_dtypes(include="number").dropna()
    if vif is None or num_df.shape[1] < 2:
        return
    st.subheader("8️⃣ 다중공선성(VIF)")
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
        msg = "VIF>10: " + ', '.join(high_vif['feature'].tolist())
        st.warning(msg)
        eda_logs.append(msg)
    else:
        msg = "VIF<10: 다중공선성 문제 없음"
        st.success(msg)
        eda_logs.append(msg)

def _correlation(df: pd.DataFrame, eda_logs: list):
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        return
    st.subheader("9️⃣ 상관 행렬(Pearson)")
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(min(12, 0.5 * corr.shape[1]), 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig, use_container_width=True)
    high_pairs = [
        (i, j)
        for i in corr.columns
        for j in corr.columns
        if i != j and abs(corr.loc[i, j]) > 0.8
    ]
    if high_pairs:
        msg = "상관 |r| > .8 변수쌍: " + ", ".join([f"{a}-{b}" for a, b in high_pairs])
        st.warning(msg)
        eda_logs.append(msg)
    else:
        msg = "모든 변수 쌍에서 |r| > .8 없음"
        st.success(msg)
        eda_logs.append(msg)

def _dim_reduction(df: pd.DataFrame, target: str,eda_logs: list):
    st.subheader("🔟 차원 축소(t-SNE/UMAP/Pairplot) 시각화")
    num_cols = [c for c in df.select_dtypes(include='number').columns if c != target]
    if len(num_cols) < 2: return
    msg = "차원축소/시각화 실행"
    eda_logs.append(msg)
    X = df[num_cols].fillna(0)
    y = df[target]
    if TSNE is not None:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)//10))
        X_emb = tsne.fit_transform(X)
        df_emb = pd.DataFrame(X_emb, columns=["x", "y"])
        df_emb[target] = y.values
        fig, ax = plt.subplots()
        sns.scatterplot(x="x", y="y", hue=target, data=df_emb, ax=ax, palette='tab10', alpha=0.8)
        ax.set_title("t-SNE Embedding by Class")
        st.pyplot(fig, use_container_width=True)
    if UMAP is not None:
        reducer = UMAP(n_components=2, random_state=42)
        X_emb = reducer.fit_transform(X)
        df_emb = pd.DataFrame(X_emb, columns=["x", "y"])
        df_emb[target] = y.values
        fig, ax = plt.subplots()
        sns.scatterplot(x="x", y="y", hue=target, data=df_emb, ax=ax, palette='tab10', alpha=0.8)
        ax.set_title("UMAP Embedding by Class")
        st.pyplot(fig, use_container_width=True)
    # Pairplot (비선형/복잡 패턴 확인)
    if len(num_cols) <= 8:
        st.write("수치형 변수 조합별 pairplot:")
        pairplot_fig = sns.pairplot(df, hue=target, vars=num_cols, palette='tab10', plot_kws={"alpha":0.5, "s":20})
        st.pyplot(pairplot_fig)

def _full_profile(df: pd.DataFrame):
    with st.expander("🔍 전체 Profiling Report (optional)", expanded=False):
        st.info("전체 변수·관계·분포 일괄 진단(대용량 데이터는 시간이 오래 걸릴 수 있음)")
        if ydata_profiling:
            profile = ydata_profiling.ProfileReport(df, title="Profiling", minimal=True)
            st.components.v1.html(profile.to_html(), height=800, scrolling=True)
        else:
            st.info("`pip install ydata-profiling` 후 전체 리포트 가능")

def generate(df: pd.DataFrame, target: str):
    """
    Streamlit 자동 EDA + 결과별 동적 해설 (Classification)
    """
    df = sample_df(df)
    eda_logs = []
    _overview(df, eda_logs)
    _missing_and_duplicates(df, target, eda_logs)
    _target_dist(df, target, eda_logs)
    _classwise_stats(df, target, eda_logs)
    _categorical_eda(df, target, eda_logs)
    _num_feature_by_class(df, target, eda_logs)
    _feature_importance(df, target, eda_logs)
    _outlier_overview(df, target, eda_logs)
    _multicollinearity_vif(df, eda_logs)
    _correlation(df, eda_logs)
    _dim_reduction(df, target, eda_logs)
    _full_profile(df)
    return eda_logs 