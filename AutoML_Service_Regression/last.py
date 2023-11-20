import streamlit as st
from pycaret.regression import *
import funcRegression as fr
from sklearn.metrics import mean_squared_error, mean_absolute_error


def last(tune_result,selected_metric,selected_feature,mse,rmse,mae,mape):
    st.header("8. 최종 분석 및 결과")
    
    st.subheader("8-1. 모델 성능")
     # 예측에 사용된 모델 표시
    st.write("#### 사용된 모델:")
    st.write(str(tune_result))
    
    # 모델 성능 메트릭 표시
    st.write("#### 모델 성능:")
    st.write(selected_metric)
    if selected_metric == "MSE":
        fr.metric_analysis(selected_metric, mse)
        st.write(f"MSE: {mse:.2f}")  # 예측된 성능 지표 출력
    elif selected_metric == "MAE":
        fr.metric_analysis(selected_metric, mae)
        st.write(f"MAE: {mae:.2f}")  # 예측된 성능 지표 출력
    elif selected_metric == "RMSE":
        fr.metric_analysis(selected_metric, rmse)
        st.write(f"RMSE: {rmse:.2f}")  # 예측된 성능 지표 출력
    elif selected_metric == "MAPE":
        fr.metric_analysis(selected_metric, mape)
        st.write(f"MAPE: {mape:.2f}%")
    
    st.subheader("8-2. 가장 중요한 변수")
    st.write("""
        선택된 모델에 따르면, **{}** 변수가 가장 큰 영향을 미치는 것으로 확인되었습니다. 이 변수는 모델의 예측
        성능에 결정적인 역할을 하는 것으로 보여지며, 다른 변수들보다 상당히 높은 중요도를 가지고 있습니다.
        따라서 **{}** 변수는 모델을 이해하고 최적화하기 위해 중점적으로 분석되어야 할 필요가 있습니다.
    """.format(selected_feature, selected_feature))
    
    

    
   
    
