import streamlit as st
import funcRegression as fr
import best_model as bm

def automl(sampling_data_copy,target_col,remove_outliers,only_id,metrics,only_name):
    
    st.header('4. AutoML 이용하여 최적의 모델 선정')

    use_gpu = st.checkbox("Use GPU")  # GPU 사용 여부 선택
    st.subheader('train size 선정')
    train_size = st.slider('Select train size ratio', min_value=0.1, max_value=0.9, value=0.7, step=0.1)  # 훈련 데이터 크기 선택
    tuned_result = None


    # 모든 모델을 튜닝하고 결과를 가져옵니다.

    # 이상치의 비율에 따른 메트릭 추천
    recommended_metric = fr.recommend_metric(sampling_data_copy[target_col].values)
    st.write(f"데이터의 이상치 비율에 따라 추천하는 메트릭은 '{recommended_metric}' 입니다.")
    st.warning("해당 추천은 이상치만을 고려한 추천이므로, 다른 고려할 중요 사항이 있다면 직접 선택해주세요.")

    selected_metric = st.selectbox("원하는 메트릭을 선택하세요", metrics, index=metrics.index(recommended_metric))

    if selected_metric == "mse":
        selected_metric = "MSE"
        st.session_state.selected_metric = selected_metric
    elif selected_metric == "mae":  
        selected_metric = "MAE"
        st.session_state.selected_metric = selected_metric
    elif selected_metric == "rmse":
        selected_metric = "RMSE"
        st.session_state.selected_metric = selected_metric
    elif selected_metric == "mape":
        selected_metric = "MAPE"
        st.session_state.selected_metric = selected_metric
    
    # pycaret으로 모델 비교
    if st.button("AutoML 실행"):
        fr.setup(sampling_data_copy, target_col, train_size, use_gpu, remove_outliers, fold=5)

        tuned_models, results_sorted = fr.tune_all_models(only_id, selected_metric, only_name)
    
        # 튜닝 결과를 지정된 평가 기준에 따라 오름차순으로 정렬합니다.
        results_sorted = results_sorted.sort_values(by=selected_metric, ascending=True)
    
        st.write(f"Models sorted by {selected_metric}:")
        results_sorted = results_sorted.reset_index(drop=True)
        st.write(results_sorted[['Model Name', selected_metric]])
    
        best_model = results_sorted.iloc[0]['Model Name'] 
        tuned_result = tuned_models[bm.best_model(best_model)]
        st.write("AutoML 라이브러리 내 validation한 결과입니다.")
        st.markdown(f"**{selected_metric}에 따른 최적의 모델은 {best_model} 입니다.**")

        st.session_state.tuned_result = tuned_result


    st.session_state.train_size = train_size


    st.text('')
    if st.button("예측 후 시각화 및 분석(변수 중요도)"):
        st.session_state.page = "예측 후 시각화 및 분석(변수 중요도)"
        