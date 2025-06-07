# automl/timeseries/preprocessing.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest

def auto_datetime(df, col):
    # datetime 변환
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return df[col]
    converted = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
    n_missing = converted.isna().sum()
    if n_missing > 0:
        sample_vals = df[col][converted.isna()].astype(str).unique()[:5]
        st.warning(
            f"'{col}' 컬럼을 datetime으로 변환 실패: {n_missing}개 NaT. 예시: {sample_vals}"
        )
    return converted

def auto_numeric(series, colname=""):
    result = pd.to_numeric(series, errors='coerce')
    n_missing = result.isna().sum()
    if n_missing > 0:
        sample_vals = series[result.isna()].astype(str).unique()[:5]
        st.warning(
            f"[{colname}] 컬럼을 수치형으로 변환 실패: {n_missing}개 NaN. 예시: {sample_vals}"
        )
    return result

class TimeSeriesPreprocessor:
    def __init__(self):
        self.time_col = None
        self.num_cols = []
        self.cat_cols = []
        self.scaler = None
        self.selected_columns = []
        self.log = []

    def fit(self, df, time_col, target):
        self.log.clear()
        self.time_col = time_col

        # 1. 시간 컬럼 robust하게 변환 및 정렬
        df[time_col] = auto_datetime(df, time_col)
        if df[time_col].isna().sum() > 0:
            self.log.append(f"'{time_col}' 컬럼 datetime 변환에서 {df[time_col].isna().sum()}개 NaT 발생")
        df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
        self.log.append("시간순으로 데이터 정렬")

        # 2. 타깃 포함 모든 수치형 robust 변환 (object→float)
        if df[target].dtype == 'object':
            df[target] = auto_numeric(df[target], target)
        # 모든 object → numeric 시도
        for col in df.columns:
            if col in [time_col, target]:
                continue
            if df[col].dtype == "object":
                try:
                    df[col] = auto_numeric(df[col], col)
                except Exception as e:
                    st.warning(f"[{col}] 컬럼 변환 에러: {e}")

        self.num_cols = [col for col in df.select_dtypes(include='number').columns if col != target]
        self.cat_cols = list(df.select_dtypes(include=['object', 'category']).columns)

        # 3. 결측치 처리 (타깃: 보간/채움, 수치: 평균, 범주: 최빈)
        if df[target].isna().sum() > 0:
            df[target] = df[target].interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")
            self.log.append(f"타깃({target}) 결측치: 선형보간 및 양방향 채움")
        if self.num_cols:
            for col in self.num_cols:
                if df[col].isna().sum() > 0:
                    df[col] = df[col].interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")
                    self.log.append(f"{col} 결측치: 선형보간 및 양방향 채움")
        else:
            self.log.append("수치형 결측 없음")
        if self.cat_cols:
            for col in self.cat_cols:
                if df[col].isna().sum() > 0:
                    mode_val = df[col].mode(dropna=True)[0] if not df[col].mode(dropna=True).empty else ""
                    df[col].fillna(mode_val, inplace=True)
                    self.log.append(f"{col}: 최빈값 '{mode_val}'로 결측 대치")
        else:
            self.log.append("범주형 결측 없음")

        # 4. 중복 시간 제거
        dup_cnt = df.duplicated(subset=[time_col]).sum()
        if dup_cnt > 0:
            df = df.drop_duplicates(subset=[time_col], keep='last')
            self.log.append(f"중복 timestamp {dup_cnt}건: 마지막값으로 자동 제거")
        else:
            self.log.append("중복 timestamp 없음")

        # 5. 시간 파생 피처
        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df["month"] = df[time_col].dt.month.fillna(0).astype(int)
            df["weekday"] = df[time_col].dt.dayofweek.fillna(0).astype(int)
            df["hour"] = df[time_col].dt.hour.fillna(0).astype(int)
            df["is_weekend"] = (df["weekday"] >= 5).astype(int)
            self.log.append("month/weekday/hour/is_weekend 파생")
        else:
            for feat in ["month","weekday","hour","is_weekend"]:
                df[feat] = 0
            self.log.append("datetime 변환 실패로 시간파생변수 0 처리")

        # 6. lag/rolling 자동 추가
        df["lag1"] = df[target].shift(1)
        df["lag7"] = df[target].shift(7)
        df["rolling_mean_7"] = df[target].rolling(7, min_periods=1).mean()
        df["rolling_std_7"] = df[target].rolling(7, min_periods=1).std().fillna(0)
        self.log.append("lag1, lag7, rolling_mean_7, rolling_std_7 생성")

        # 7. 이상치 탐지: IsolationForest
        X_num = df[self.num_cols + ["lag1", "lag7", "rolling_mean_7", "rolling_std_7"]].fillna(0)
        if len(df) > 200 and X_num.shape[1] > 0:
            iso = IsolationForest(n_estimators=40, contamination=0.01, random_state=42)
            out_pred = iso.fit_predict(X_num)
            outlier_idx = np.where(out_pred == -1)[0]
            if len(outlier_idx) > 0:
                df = df.drop(df.index[outlier_idx])
                self.log.append(f"IsolationForest로 {len(outlier_idx)}개 이상치 자동 제거")
            else:
                self.log.append("심각한 이상치 없음.")
        else:
            self.log.append("행 개수 부족/수치형 부족: 이상치 탐지 생략")

        # 8. 스케일링
        skewness = np.abs(df[self.num_cols]).skew().mean() if self.num_cols else 0
        if skewness < 1:
            self.scaler = StandardScaler()
            self.log.append(f"평균 Skewness {skewness:.2f}: 1 미만 → StandardScaler")
        else:
            self.scaler = RobustScaler()
            self.log.append(f"평균 Skewness {skewness:.2f}: 1 이상 → RobustScaler")

        scaled_num = self.scaler.fit_transform(df[self.num_cols + ["lag1", "lag7", "rolling_mean_7", "rolling_std_7"]].fillna(0))
        self.selected_columns = list(self.num_cols) + ["lag1", "lag7", "rolling_mean_7", "rolling_std_7", "month", "weekday", "hour", "is_weekend"]
        self.log.append(f"스케일링 + 최종 입력 피처: {', '.join(self.selected_columns)}")
        self.df_ref = df.copy()

    def transform(self, df, time_col, target):
        df[time_col] = auto_datetime(df, time_col)
        df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
        if df[target].dtype == 'object':
            df[target] = auto_numeric(df[target], target)
        for col in df.columns:
            if col in [time_col, target]:
                continue
            if df[col].dtype == "object":
                try:
                    df[col] = auto_numeric(df[col], col)
                except Exception:
                    continue
        # 시간 파생
        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df["month"] = df[time_col].dt.month.fillna(0).astype(int)
            df["weekday"] = df[time_col].dt.dayofweek.fillna(0).astype(int)
            df["hour"] = df[time_col].dt.hour.fillna(0).astype(int)
            df["is_weekend"] = (df["weekday"] >= 5).astype(int)
        else:
            for feat in ["month","weekday","hour","is_weekend"]:
                df[feat] = 0
        # lag/rolling
        df["lag1"] = df[target].shift(1)
        df["lag7"] = df[target].shift(7)
        df["rolling_mean_7"] = df[target].rolling(7, min_periods=1).mean()
        df["rolling_std_7"] = df[target].rolling(7, min_periods=1).std().fillna(0)
        # 결측 보간
        for col in self.selected_columns:
            if df[col].isna().sum() > 0:
                df[col] = df[col].interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")
        # 스케일링
        X_num = df[self.num_cols + ["lag1", "lag7", "rolling_mean_7", "rolling_std_7"]].fillna(0)
        X_num_scaled = self.scaler.transform(X_num)
        X = pd.DataFrame(X_num_scaled, columns=self.num_cols + ["lag1", "lag7", "rolling_mean_7", "rolling_std_7"], index=df.index)
        X = pd.concat([X, df[["month", "weekday", "hour", "is_weekend"]].reset_index(drop=True)], axis=1)
        X = X[self.selected_columns]
        return X

    def fit_transform(self, df, time_col, target):
        self.fit(df, time_col, target)
        return self.transform(df, time_col, target), self.df_ref, self

    def get_log(self):
        return self.log

def run(df, target, time_col, mode="fit_transform", preprocessor=None):
    if mode == "fit_transform":
        pre = TimeSeriesPreprocessor()
        X, df_ref, pre = pre.fit_transform(df.copy(), time_col, target)
        st.header("⚡ 전처리 단계별 설명 (Train 기준)")
        for step in pre.get_log():
            st.markdown(f"- {step}")
        st.subheader("📋 전처리 데이터 미리보기 (Train)")
        st.write(X.head())
        st.write("최종 shape:", X.shape)
        y = df[target].reset_index(drop=True)
        return X, y, pre
    elif mode == "transform" and preprocessor is not None:
        X = preprocessor.transform(df.copy(), time_col, target)
        st.header("⚡ 전처리 결과 (Valid/Test 기준)")
        st.write("Train에서 fit된 기준으로 변환(transform)만 적용")
        st.write(X.head())
        st.write("최종 shape:", X.shape)
        return X
    else:
        raise ValueError("mode는 fit_transform/transform 중 하나여야 하고, transform시 preprocessor 필요")