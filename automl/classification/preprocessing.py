import streamlit as st
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.utils import class_weight
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE

class ClassificationPreprocessor:
    def __init__(self):
        self.num_cols = []
        self.cat_cols = []
        self.imputer_num = None
        self.imputer_cat = None
        self.scaler = None
        self.encoder = None
        self.class_weights = None
        self.selected_columns = []
        self.log = []
        self.smote_applied = False
        self.isolation_mask = None   # 이상치 제거용 마스크

    def fit(self, df, target):
        self.log.clear()
        # 1. 컬럼 구분
        self.num_cols = [col for col in df.select_dtypes(include='number').columns if col != target]
        self.cat_cols = list(df.select_dtypes(include=['object', 'category']).columns)

        # 2. 결측치 처리 (수치형: 평균/중앙값, 범주형: 최빈값)
        num_null_ratio = df[self.num_cols].isna().mean() if self.num_cols else pd.Series()
        cat_null_ratio = df[self.cat_cols].isna().mean() if self.cat_cols else pd.Series()
        # 수치형
        for col in self.num_cols:
            null_ratio = num_null_ratio[col]
            if null_ratio == 0:
                continue
            if null_ratio < 0.3:
                strategy = "mean"
                reason = f"{col} 결측치 {null_ratio:.1%}: 30% 미만 → 평균 대치"
            else:
                strategy = "median"
                reason = f"{col} 결측치 {null_ratio:.1%}: 30% 이상 → 중앙값 대치"
            self.log.append(reason)
        self.imputer_num = SimpleImputer(strategy="mean")
        self.imputer_num.fit(df[self.num_cols])
        self.log.append("수치형 결측치: 평균(mean)으로 일괄 대치 (SimpleImputer).")
        # 범주형
        if self.cat_cols:
            self.imputer_cat = SimpleImputer(strategy="most_frequent")
            self.imputer_cat.fit(df[self.cat_cols])
            self.log.append("범주형 결측치: 최빈값(most_frequent)으로 대치 (SimpleImputer).")
        else:
            self.imputer_cat = None
            self.log.append("범주형 변수 없음 → 결측 대치 생략.")

        # 3. 스케일링: 수치형 변수에 대해 분포 확인 후 적용
        skewness = np.abs(df[self.num_cols].skew()).mean() if self.num_cols else 0
        if skewness < 1:
            self.scaler = StandardScaler()
            self.log.append(f"평균 Skewness {skewness:.2f}: 1 미만 → StandardScaler 적용 (정규분포 가정).")
        else:
            self.scaler = RobustScaler()
            self.log.append(f"평균 Skewness {skewness:.2f}: 1 이상 → RobustScaler 적용 (이상치 영향 최소화).")
        num_filled = pd.DataFrame(self.imputer_num.transform(df[self.num_cols]), columns=self.num_cols, index=df.index)
        self.scaler.fit(num_filled)

        # 4. 인코딩: 범주형 변수 전체 OneHotEncoder
        if self.cat_cols:
            n_unique = df[self.cat_cols[0]].nunique() if len(self.cat_cols) == 1 else -1
            if len(self.cat_cols) == 1 and n_unique < 10:
                self.encoder = LabelEncoder()
                self.encoder.fit(df[self.cat_cols[0]].astype(str))
                self.log.append(f"{self.cat_cols[0]}: 유니크값 {n_unique} < 10 → LabelEncoding 적용.")
            else:
                ohe_kwargs = dict(handle_unknown="ignore")
                if int(sklearn.__version__.split('.')[1]) >= 2:  # scikit-learn 1.2 이상
                    ohe_kwargs['sparse_output'] = False
                else:
                    ohe_kwargs['sparse'] = False
                self.encoder = OneHotEncoder(**ohe_kwargs)
                self.encoder.fit(df[self.cat_cols].astype(str))
                self.log.append(f"{', '.join(self.cat_cols)}: OneHotEncoding 적용 (범주형 >1개 또는 유니크값 ≥10).")

        # 5. 이상치 처리: IsolationForest 적용, 이상치 row 실제 제거 (훈련셋에서만)
        self.isolation_mask = np.ones(len(df), dtype=bool)  # 기본적으로 모두 사용
        if len(df) > 200 and self.num_cols:
            iso = IsolationForest(n_estimators=40, contamination=0.01, random_state=42)
            out_pred = iso.fit_predict(num_filled)
            outlier_idx = np.where(out_pred == -1)[0]
            if len(outlier_idx) > 0:
                self.log.append(f"IsolationForest로 {len(outlier_idx)}개 row 이상치 자동 제거")
                self.isolation_mask[outlier_idx] = False  # 이상치 마스크 저장
            else:
                self.log.append("IsolationForest로 심각한 이상치 없음.")
        else:
            self.log.append("행 개수 부족/수치형 없음: 이상치 탐지 생략")

        # 6. 클래스 imbalance 자동 보정: SMOTE(소수 클래스 10% 미만/imbalance>4 배)
        class_counts = df[target].value_counts()
        ratio = class_counts.max() / class_counts.min() if len(class_counts) > 1 else 1.0
        if ratio > 4 and class_counts.min() / len(df) < 0.1:
            self.smote_applied = True
            self.log.append(f"클래스 불균형 심각 (ratio={ratio:.1f}) → SMOTE(합성샘플) 적용 예정")
        else:
            self.smote_applied = False
            self.log.append("클래스 분포 균형 양호 or 심각하지 않아 SMOTE 미적용")

        # 7. 클래스 가중치 계산
        classes = np.unique(df[target])
        weights = class_weight.compute_class_weight("balanced", classes=classes, y=df[target])
        self.class_weights = dict(zip(classes, weights))
        self.log.append(f"클래스별 자동 가중치 부여: {self.class_weights}")

    def transform(self, df, target):
        # 이상치 마스크 적용 (train set에서만 적용!)
        if self.isolation_mask is not None and len(self.isolation_mask) == len(df):
            df = df[self.isolation_mask]
        # 1. 결측치
        X_num = pd.DataFrame(self.imputer_num.transform(df[self.num_cols]), columns=self.num_cols, index=df.index)
        if self.cat_cols:
            X_cat = pd.DataFrame(self.imputer_cat.transform(df[self.cat_cols]), columns=self.cat_cols, index=df.index)
        else:
            X_cat = pd.DataFrame(index=df.index)
        # 2. 스케일링
        X_num_scaled = pd.DataFrame(self.scaler.transform(X_num), columns=self.num_cols, index=df.index)
        # 3. 인코딩
        if self.encoder and self.cat_cols:
            if isinstance(self.encoder, LabelEncoder):
                X_cat_enc = pd.DataFrame({self.cat_cols[0]: self.encoder.transform(X_cat[self.cat_cols[0]].astype(str))}, index=df.index)
            else:
                X_cat_enc = pd.DataFrame(self.encoder.transform(X_cat), columns=self.encoder.get_feature_names_out(self.cat_cols), index=df.index)
            X = pd.concat([X_num_scaled, X_cat_enc], axis=1)
        else:
            X = X_num_scaled
        return X

    def fit_transform(self, df, target):
        self.fit(df, target)
        # isolation mask로 이상치 row 제거 (train set)
        df_clean = df[self.isolation_mask] if self.isolation_mask is not None and len(self.isolation_mask) == len(df) else df
        X = self.transform(df_clean, target)
        y = df_clean[target]
        # SMOTE 적용 (오직 fit_transform(train)에서만)
        if self.smote_applied:
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)
            self.log.append(f"SMOTE 적용 후 샘플 수: {X.shape[0]}")
        return X, y

    def get_log(self):
        return self.log

def run(df, target, mode="fit_transform", preprocessor=None):
    if mode == "fit_transform":
        pre = ClassificationPreprocessor()
        X, y = pre.fit_transform(df, target)  # 이제 X, y 반환
        st.header("⚡ 전처리 단계별 설명 (Train 기준)")
        for step in pre.get_log():
            st.markdown(f"- {step}")
        st.subheader("📋 전처리 데이터 미리보기 (Train)")
        st.write(X.head())
        st.write("최종 shape:", X.shape)
        return X, y, pre
    elif mode == "transform" and preprocessor is not None:
        X = preprocessor.transform(df, target)
        st.header("⚡ 전처리 결과 (Valid/Test 기준)")
        st.write("Train에서 fit된 기준으로 변환(transform)만 적용")
        st.write(X.head())
        st.write("최종 shape:", X.shape)
        return X
    else:
        raise ValueError("mode는 fit_transform/transform 중 하나여야 하고, transform시 preprocessor 필요")