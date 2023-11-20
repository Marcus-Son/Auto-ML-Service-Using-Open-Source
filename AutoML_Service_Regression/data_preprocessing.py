import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from scipy.stats import skew, kurtosis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import sys

def data_preprocessing(data):
    st.header('1. 데이터 전처리')
    
    st.text('')


    st.subheader('1-1.데이터 형태')  # 원본 데이터의 상위 5행 표시
    with st.expander("사용자가 업로드한 파일의 상위 5행의 데이터를 표시합니다."):
        st.write(data.head())

    st.text('')
   

    st.subheader('1-2.목표변수 선택')
    target_col = st.selectbox('Select the target column', data.columns)  # 목표변수 선택

    st.text('')

    # 결측값 처리
    st.subheader('1-3.결측치')
    missing_values = data.isnull().sum()
    with st.expander("각 feature의 결측치의 개수를 보여줍니다."):
        st.write(missing_values)
        
    # 결측치 처리 방법 선택
    options = ['0으로 채우기', '평균값으로 채우기', '중앙값으로 채우기', '최빈값으로 채우기', '결측값을 포함한 행 제거', '아무 것도 하지 않기']
    choice = st.selectbox("결측치 처리 방법을 선택하세요.", options)

    if choice == '0으로 채우기':
        analysis_data = data.fillna(0)
    elif choice == '평균값으로 채우기':
        analysis_data = data.fillna(data.mean())
    elif choice == '중앙값으로 채우기':
        analysis_data = data.fillna(data.median())
    elif choice == '최빈값으로 채우기':
        mode_val = data.mode().iloc[0]  # DataFrame에서 최빈값을 얻기 위해 iloc 사용
        analysis_data = data.fillna(mode_val)
    elif choice == '결측값을 포함한 행 제거':
        analysis_data = data.dropna()
    else:
        analysis_data = data.copy()  # 아무 것도 하지 않기 선택 시, 원본 데이터 그대로 유지

    st.write("선택된 방법으로 결측치 처리가 완료되었습니다.")

    st.text('')

    #이상치 처리
    st.subheader('1-4. 이상치 처리')
    select_column = st.selectbox('변수를 선택하세요 (박스 플롯)', analysis_data.columns)
    fig, ax = plt.subplots()
    sns.boxplot(y=analysis_data[select_column], ax=ax)
    st.pyplot(fig)

    st.text("* 체크 표시 선택 시 : 이상치 제거 / 미선택 시 : 이상치 유지")

        # 이상치 판단 및 추천 로직
    Q1 = analysis_data.quantile(0.25)
    Q3 = analysis_data.quantile(0.75)
    IQR = Q3 - Q1
    outlier_condition = ((analysis_data < (Q1 - 1.5 * IQR)) | (analysis_data > (Q3 + 1.5 * IQR)))
    outlier_count = outlier_condition.sum().sum() # 전체 이상치 개수
    
        # 임의로 설정한 임계값, 예를 들어 전체 데이터 중 이상치가 1%를 넘으면 제거를 추천
    threshold = 0.01 * analysis_data.size
    if outlier_count > threshold:
        st.warning(f"이상치 수가 전체 데이터의 1% 이상 차지합니다. 각 컬럼에서 이상치가 1%이상 차지하는 피처를 추천드립니다.")
        # 모든 컬럼의 이상치 비율 계산
    outlier_ratio = outlier_condition.mean()
        # 10% 이상의 이상치 비율을 가진 컬럼들 찾기
    recommended_columns_to_remove = outlier_ratio[outlier_ratio > 0.1].index.tolist()

        # 이런 컬럼들이 있는 경우에만 메시지 출력
    if recommended_columns_to_remove:
        st.write(f"추천하는 이상치를 제거할 컬럼들: {', '.join(recommended_columns_to_remove)}")
    else:
        st.write("10% 이상의 이상치를 가진 컬럼이 없습니다.")

        # 사용자에게 다른 옵션도 보여주기
    columns_sorted_by_outliers = outlier_ratio[outlier_ratio > 0].sort_values(ascending=False)
    with st.expander("다른 컬럼들의 이상치 비율"):
        st.write(columns_sorted_by_outliers)

    remover_outlier_data=analysis_data.copy()
    remove_outliers=None
    st.warning(f"너무 많은 이상치 제거는 데이터 수의 감소와 분포 변화를 일으키니 유의해주세요.")   
        # 전체 데이터에 대한 이상치를 한번에 제거하는 옵션
    remove_all_outliers = st.checkbox("전체 데이터의 이상치 제거")
    if remove_all_outliers:
        # 전체 이상치 조건에 맞지 않는 행만 필터링하여 저장
        remover_outlier_data = analysis_data[~outlier_condition.any(axis=1)]
        st.write("전체 데이터의 이상치가 제거되었습니다.")

        # 여러 컬럼을 선택하고 해당 컬럼들의 이상치를 제거하는 옵션
    selected_columns_for_outliers = st.multiselect('이상치를 제거할 컬럼들을 선택하세요', analysis_data.columns)
    if selected_columns_for_outliers:
        remove_outliers = st.checkbox(f"{', '.join(selected_columns_for_outliers)} 컬럼들의 이상치 제거")
        if remove_outliers:
            # 각 선택된 컬럼의 이상치 조건에 맞지 않는 행만 필터링하여 저장
            for col in selected_columns_for_outliers:
                col_outlier_condition = ((analysis_data[col] < (Q1[col] - 1.5 * IQR[col])) | (analysis_data[col] > (Q3[col] + 1.5 * IQR[col])))
                remover_outlier_data = analysis_data[~col_outlier_condition]
            st.write(f"{', '.join(selected_columns_for_outliers)} 컬럼들의 이상치가 제거되었습니다.")
    
    st.text('')

     # 데이터의 개수와 결측치 확인
    st.write("### Data Information:")
    buf = StringIO()
    sys.stdout = buf
    remover_outlier_data.info()
    sys.stdout = sys.__stdout__
    st.text(buf.getvalue())

    st.text('')

    #데이터 스케일링   
    st.subheader('1-5. 데이터 스케일링')

    st.text("-데이터에 이상치가 많을 경우: Robust Scaling 추천")
    st.text("-데이터가 정규 분포를 따르는 경우: Standard Scaling 추천")
    st.text("-그렇지 않은 경우: Min-Max Scaling 추천")
        # 데이터의 skewness와 kurtosis 계산
    skewness = remover_outlier_data.drop(columns=[target_col]).apply(skew).mean()
    kurt = remover_outlier_data.drop(columns=[target_col]).apply(kurtosis).mean()
        # 추천 로직
    recommended_scaling = 'None'
    if outlier_count > threshold:  # 이전에 계산한 이상치 개수를 사용
        recommended_scaling = 'Robust Scaling'
    elif abs(skewness) < 0.5 and abs(kurt - 3) < 0.5:
        recommended_scaling = 'Standard Scaling'
    else:
        recommended_scaling = 'Min-Max Scaling'

    st.write(f"추천하는 스케일링 방식: {recommended_scaling}")


        # 스케일링 방법 선택
    scaling_methods = ['None', 'Min-Max Scaling', 'Standard Scaling', 'Robust Scaling']
    selected_method = st.selectbox('스케일링 방식을 선택하세요', scaling_methods)
        # 스케일링 수행
    if selected_method == 'Min-Max Scaling':
        scaler = MinMaxScaler()
    elif selected_method == 'Standard Scaling':
        scaler = StandardScaler()
    elif selected_method == 'Robust Scaling':
        scaler = RobustScaler()
    else:
        scaler = None  # 기본값으로 None을 할당

    scaling_data=remover_outlier_data.copy()
    # 스케일러가 정의되지 않은 경우 예외 처리
    if scaler is not None:
        data_to_scale = remover_outlier_data.drop(columns=[target_col])

        # 타겟 열을 별도로 저장
        target_column_data = remover_outlier_data[target_col].reset_index(drop=True)

        # 스케일러를 사용하여 데이터 스케일링
        data_scaled = scaler.fit_transform(data_to_scale)

        # 스케일링된 데이터 프레임 생성
        scaled_columns = data_to_scale.columns
        scaling_data = pd.DataFrame(data_scaled, columns=scaled_columns)

        # 스케일링된 데이터에 타겟 열을 추가
        scaling_data[target_col] = target_column_data

        # 스케일링된 데이터의 상위 5행 표시
        with st.expander("스케일링된 데이터의 상위 5행을 표시합니다."):
            st.write(scaling_data.head())


    st.session_state.scaling_data = scaling_data
    st.session_state.missing_values = missing_values
    st.session_state.outlier_count = outlier_count
    st.session_state.recommended_scaling = recommended_scaling
    st.session_state.target_col = target_col
    st.session_state.remove_outliers = remove_outliers
    st.session_state.choice = choice
    st.session_state.selected_method = selected_method

    st.text('')

    if st.button("EDA 진행하기"):
        st.session_state.page = "EDA"