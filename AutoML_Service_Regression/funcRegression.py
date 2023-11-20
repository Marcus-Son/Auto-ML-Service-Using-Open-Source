import pandas as pd
import numpy as np
import pycaret.regression
from pycaret.regression import *
import io
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.inspection import permutation_importance
from SALib.sample import morris as morris_sampler
from SALib.analyze import morris as morris_analyzer
import get_model_name as gmn



# 회귀 평가 기준
from scipy.stats import iqr

def recommend_metric(data):
    # IQR을 사용하여 이상치를 찾습니다.
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = iqr(data)
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = [point for point in data if point < lower_bound or point > upper_bound]
    
    ratio_outliers = len(outliers) / len(data)
    
    # 이상치의 비율에 따른 메트릭 추천
    if ratio_outliers >= 0.10:
        return "mae"
    elif 0.05 <= ratio_outliers < 0.10:
        return "mape"
    elif 0.01 <= ratio_outliers < 0.05:
        return "mse"
    else:
        return "rmse"


#pycaret 함수
def setup(data, target, train, use_gpu, outliar, fold):
    return pycaret.regression.setup(data, target=target, session_id=123, train_size=train, use_gpu=use_gpu, remove_outliers=outliar,fold=5)


def save_df():
    results = pycaret.regression.pull()
    return results


def compare(target_model_list):
    return pycaret.regression.compare_models(include=target_model_list)

def tune_all_models(model_ids, opt, model_names):
    tuned_models = {}
    results_list = []

    
    for idx, model_id in enumerate(model_ids):
        model = pycaret.regression.create_model(model_id)
        tuned_model = pycaret.regression.tune_model(model, optimize=opt, choose_better=True)
        tuned_models[model_id] = tuned_model
        
        # 모델의 결과를 가져옵니다.
        model_results = pycaret.regression.pull()  # 마지막 실행 결과 가져오기
        model_result_row = model_results.iloc[-1].copy()  # 마지막 행이 현재 모델의 결과
        model_result_row = model_result_row.drop(model_result_row.index[0])


        # 모델의 이름을 추가합니다.
        model_result_row['Model Name'] = model_names[idx]  # 현재 모델의 이름을 추가
        
        results_list.append(model_result_row)


    # 결과를 데이터프레임으로 변환
    results_df = pd.concat(results_list, axis=1).T  # Series를 합쳐서 DataFrame으로 변환

    return tuned_models, results_df.sort_values(by=opt,ascending=False)



def tune(model, opt):
    return pycaret.regression.tune_model(model, optimize=opt, choose_better=True)


def Blend(arr):
    arr[0] = pycaret.regression.create_model(arr[0])
    arr[1] = pycaret.regression.create_model(arr[1])
    arr[2] = pycaret.regression.create_model(arr[2])
    return pycaret.regression.blend_models([arr[0], arr[1], arr[2]])


def single(name):
    return pycaret.regression.create_model(name)


def single_visual(df):
    visual = df.iloc[0:9]
    return visual.plot()


def evaluate(model):
    return pycaret.regression.evaluate_model(model)


def prediction(model):
    return pycaret.regression.predict_model(model)


def save_model(model, name):
    return pycaret.regression.save_model(model, name)


def load(name):
    return pycaret.regression.load_model(name)



# 시각화 부분 
def feature_importances_plot( tuned_result):

    plot_model(tuned_result, plot='feature', save=True)
    plot_model(tuned_result, plot='learning', save=True)
    

def xai_plot(tuned_result):

    interpret_model(tuned_result, plot = 'pdp', save=True)
    interpret_model(tuned_result, plot = 'pfi', save=True)
    interpret_model(tuned_result, plot = 'msa', save=True)

# MAPE 함수
def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def metric_analysis(selected_metric, value):
    if selected_metric == "MSE":
        if value < 10:
            st.info("MSE 값이 상대적으로 낮습니다. 모델의 예측이 대부분의 경우 실제 값에 매우 근접하다는 것을 나타냅니다.")
        else:
            st.warning("MSE 값이 상대적으로 높습니다. 예측값과 실제 값 사이에 큰 오차가 있을 수 있음을 나타냅니다.")
    elif selected_metric == "MAE":
        if value < 5:
            st.info("MAE 값이 낮습니다. 이는 모델이 일반적으로 작은 오차를 가지고 예측을 수행하고 있다는 것을 의미합니다.")
        else:
            st.warning("MAE 값이 상대적으로 높습니다. 일부 예측에서 큰 오차가 있을 수 있음을 나타냅니다.")
    elif selected_metric == "RMSE":
        if value < 10:
            st.info("RMSE 값이 상대적으로 낮습니다. 모델의 예측이 대부분의 경우 실제 값에 매우 근접하다는 것을 나타냅니다.")
        else:
            st.warning("RMSE 값이 상대적으로 높습니다. 예측값과 실제 값 사이에 큰 오차가 있을 수 있음을 나타냅니다.")
    elif selected_metric == "MAPE":
        if value < 10:
            st.info("MAPE 값이 10% 미만입니다. 이는 모델의 예측이 대체로 정확하다는 것을 나타냅니다.")
        else:
            st.warning("MAPE 값이 10% 이상입니다. 이는 예측이 실제 값과 큰 차이를 보일 수 있다는 것을 의미합니다. 모델 튜닝 또는 데이터 전처리를 고려해 보는 것이 좋습니다.")

def compute_pfi_scores(model, X_test, y_test):
    perm = permutation_importance(model, X=X_test, y=y_test, random_state=1)  # 여기를 수정
    pfi_scores = perm.importances_mean
    return pfi_scores


def compute_msa_scores(model, X_test):
    # 변수 범위 및 값 정의
    num_vars = len(X_test.columns)  # 변수의 수
    bounds = np.array([[X_test[col].min(), X_test[col].max()] for col in X_test.columns])  # 각 변수의 범위 설정
    
    problem = {
        'num_vars': num_vars,
        'names': X_test.columns.tolist(),
        'bounds': bounds
    }
    
    # Morris 실험 설계 및 기본 효과 계산
    param_values = morris_sampler.sample(problem, N=100, num_levels=4)
    
    def model_func(input_array):
        input_df = pd.DataFrame(input_array.reshape(1, -1), columns=X_test.columns)
        return model.predict(input_df)
    
    # 실행 결과 가져오기
    model_outputs = np.apply_along_axis(model_func, 1, param_values)
    
    # 평균 절대 효과 및 평균 표준 편차 계산
    morris_eval = morris_analyzer.analyze(problem, param_values, model_outputs, print_to_console=False)
    
    # 평균 절대 효과를 반환 (또는 평균 표준 편차를 반환할 수도 있음)
    return morris_eval['mu']

import shap
def compute_shap_values(model, X_test):
    
    explainer = shap.Explainer(model, X_test)
    
    shap_values = explainer.shap_values(X_test)
    return shap_values
