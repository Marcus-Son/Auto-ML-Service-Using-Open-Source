import streamlit as st
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

class Preprocessor:
    def __init__(self):
        self.imputer = None
        self.scaler = None
        self.encoder = None
        self.num_cols = []
        self.cat_cols = []
        self.selected_columns = []
        self.vif_threshold = 10
        self.log = []

    def fit(self, df, target):
        self.log.clear()
        # 1. 컬럼 분리
        self.num_cols = [col for col in df.select_dtypes(include='number').columns if col != target]
        self.cat_cols = list(df.select_dtypes(include=['object', 'category']).columns)

        # 2. 결측치 처리 (수치형: 평균/중앙값, 범주형: 최빈값)
        for col in self.num_cols:
            null_ratio = df[col].isna().mean()
            if null_ratio == 0:
                continue
            if null_ratio < 0.3:
                strategy = "mean"
                why = f"{col} 결측치 비율 {null_ratio:.1%}: 30% 미만 → 평균(mean)으로 대치."
            else:
                strategy = "median"
                why = f"{col} 결측치 비율 {null_ratio:.1%}: 30% 이상 → 중앙값(median)으로 대치."
            self.log.append(why)
        self.imputer = SimpleImputer(strategy="mean")
        self.imputer.fit(df[self.num_cols])
        self.log.append("수치형 결측치는 scikit-learn SimpleImputer로 일괄 대치 (strategy=mean).")

        # 3. 스케일링: Skewness에 따라 결정
        skewness = np.abs(df[self.num_cols].skew()).mean() if self.num_cols else 0
        if skewness < 1:
            self.scaler = StandardScaler()
            self.log.append(f"평균 Skewness {skewness:.2f}: 1 미만 → StandardScaler 적용 (정규분포 가정).")
        else:
            self.scaler = RobustScaler()
            self.log.append(f"평균 Skewness {skewness:.2f}: 1 이상 → RobustScaler 적용 (이상치 영향 최소화).")
        X_num = self.imputer.transform(df[self.num_cols])
        self.scaler.fit(X_num)

        # 4. 인코딩
        if self.cat_cols:
            n_unique = df[self.cat_cols[0]].nunique() if len(self.cat_cols) == 1 else -1
            if len(self.cat_cols) == 1 and n_unique < 10:
                self.encoder = LabelEncoder()
                self.encoder.fit(df[self.cat_cols[0]])
                self.log.append(f"{self.cat_cols[0]}: 유니크값 {n_unique} < 10 → LabelEncoding 적용.")
            else:
                self.encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
                self.encoder.fit(df[self.cat_cols])
                self.log.append(f"{', '.join(self.cat_cols)}: OneHotEncoding 적용 (범주형 >1개 또는 유니크값 ≥10).")
        else:
            self.encoder = None
            self.log.append("범주형 변수 없음 → 인코딩 생략.")

        # 5. 다중공선성(VIF) 제거 (train 기준)
        X_num_scaled = self.scaler.transform(X_num)
        vif_data = pd.DataFrame(X_num_scaled, columns=self.num_cols)
        vifs = [variance_inflation_factor(vif_data.values, i) for i in range(len(self.num_cols))]
        self.selected_columns = [col for col, v in zip(self.num_cols, vifs) if v < self.vif_threshold]
        dropped = [col for col, v in zip(self.num_cols, vifs) if v >= self.vif_threshold]
        for col, v in zip(self.num_cols, vifs):
            if v >= self.vif_threshold:
                self.log.append(f"{col}: VIF={v:.1f} > 10 → 다중공선성 높아 자동 제거.")
            else:
                self.log.append(f"{col}: VIF={v:.1f} < 10 → 유지.")
        if dropped:
            self.log.append(f"VIF 10↑ 컬럼 자동 제거: {', '.join(dropped)}")
        else:
            self.log.append("VIF 10↑ 컬럼 없음 → 전부 사용.")

    def transform(self, df, target):
        X_num = pd.DataFrame(self.imputer.transform(df[self.num_cols]), columns=self.num_cols, index=df.index)
        X_num_scaled = pd.DataFrame(self.scaler.transform(X_num), columns=self.num_cols, index=df.index)
        if self.encoder and self.cat_cols:
            # LabelEncoder와 OneHotEncoder 분기
            if isinstance(self.encoder, LabelEncoder):
                X_cat = pd.DataFrame({self.cat_cols[0]: self.encoder.transform(df[self.cat_cols[0]])}, index=df.index)
            else:
                X_cat = pd.DataFrame(self.encoder.transform(df[self.cat_cols]), 
                                     columns=self.encoder.get_feature_names_out(self.cat_cols), index=df.index)
            X = pd.concat([X_num_scaled, X_cat], axis=1)
        else:
            X = X_num_scaled
        # VIF 기반 feature 선택
        X = X[self.selected_columns + [c for c in X.columns if c not in self.selected_columns]]
        return X

    def fit_transform(self, df, target):
        self.fit(df, target)
        return self.transform(df, target)

    def get_log(self):
        return self.log

def run(df, target, mode="fit_transform", preprocessor=None):
    if mode == "fit_transform":
        pre = Preprocessor()
        X = pre.fit_transform(df, target)
        st.header("⚡ 전처리 단계별 설명 (Train 기준)")
        for step in pre.get_log():
            st.markdown(f"- {step}")
        st.subheader("📋 전처리 데이터 미리보기 (Train)")
        st.write(X.head())
        st.write("최종 shape:", X.shape)
        return X, pre
    elif mode == "transform" and preprocessor is not None:
        X = preprocessor.transform(df, target)
        st.header("⚡ 전처리 결과 (Valid/Test 기준)")
        st.write("Train에서 fit된 기준으로 변환(transform)만 적용")
        st.write(X.head())
        st.write("최종 shape:", X.shape)
        return X
    else:
        raise ValueError("mode는 fit_transform/transform 중 하나여야 하고, transform시 preprocessor 필요")