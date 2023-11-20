import streamlit as st
import numpy as np
import pandas as pd
from io import StringIO
import sys

def discussion(missing_values,high_corr_var,corr_var_string,scaling_data,choice,selected_method):
    st.header('3. 결론 도출 및 보고서 작성')

    st.text('')


    st.subheader('3-1. EDA 주요 통찰 정리')

    # 결측치 처리
    st.markdown("**결측치 처리**")
    if missing_values.any():
        st.write(f"- {choice}를 이용하여 결측치를 처리했습니다.")
    else:
        st.write("- 데이터에 결측치가 없습니다.")

    # 이상치
    st.markdown("**이상치**")
    st.write("- 박스 플롯을 사용하여 데이터 내 이상치를 확인하였습니다.")
    st.write(f"- 각 컬럼에서 이상치 비율이 1%가 넘는 컬럼의 이상치 제거를 추천했습니다.")

    # 데이터 스케일링
    st.markdown("**데이터 스케일링**")
    st.write(f"- {selected_method}을 이용하여 데이터 스케일링을 하였습니다.")

    # 변수 분포 확인
    st.markdown("**변수 분포 확인**")
    st.write("- 각 변수별로 히스토그램을 통해 분포를 시각화하였습니다.")

    # 변수 간 상관관계
    st.markdown("**변수 간 상관관계**")
    if high_corr_var:
        st.write(f"- 상관계수가 0.8 이상인 변수 쌍: [{corr_var_string}]")
        st.write("해당 feature들은 서로 강한 상관관계를 보입니다.")
    else:
        st.write("상관계수가 0.8 이상인 변수 쌍이 없습니다.")

    st.text('')
    st.text('')


    st.subheader('3-2. 추후 조치')

    feature_delete_data = scaling_data.copy()
    st.write("### 상관분석 기반으로한 피쳐 선택")
    if high_corr_var:
        st.write("다중공선성 문제를 피하기 위해, 상관계수가 높은 변수 중 하나만 선택하여 모델링에 사용하는 것을 고려해볼 수 있습니다.")
        remove=st.checkbox("피처 제거", key="remove_features_checkbox")
        if remove:
            drop_vars = set()
            for var1, var2 in high_corr_var:  # high_corr_var_pairs는 높은 상관관계를 가진 변수쌍을 저장하는 리스트입니다.
                drop_vars.add(var2)  # var2를 제거하는 것으로 가정. 필요에 따라 var1로 변경 가능.

            # 변수 제거
            feature_delete_data = scaling_data.drop(columns=drop_vars)
            st.write(f"{', '.join(drop_vars)} 변수들을 제외한 데이터셋이 생성되었습니다.")
    else:
        st.write("상관계수가 높은 변수 쌍이 없으므로, 모든 변수를 모델링에 사용하는 것을 고려해볼 수 있습니다.")

    st.write("### 2. 데이터 확장 방법 시도")

    data_augmentation = st.checkbox('데이터 확장')

    sampling_data = feature_delete_data.copy()
    if data_augmentation:
        st.write("데이터의 양이 부족할 경우, 데이터 확장(Data Augmentation) 방법을 통해 훈련 데이터의 양을 증가시켜볼 수 있습니다.")
        # 데이터 확장 방법 선택
        augmentation_methods = [
            '아무것도 하지 않음', 
            'Random Sampling', 
            'Noise 추가 (Gaussian)',
            'Jittering'
        ]
        
        st.write("- Random Sampling은 데이터를 random으로 sampling하여 증강시킵니다.")
        st.write("- Noise 추가 (Gaussian): 각 데이터 포인트에 가우시안 노이즈를 추가합니다.")
        st.write("- Jittering: 각 데이터 포인트에 균등한 노이즈를 추가하여 데이터를 약간 이동시킵니다.")
        
        chosen_method = st.selectbox('어떤 데이터 확장 방법을 시도하시겠습니까?', augmentation_methods)

        if chosen_method == 'Random Sampling':
            # 데이터 무작위 샘플링을 통한 확장
            sampling_data = feature_delete_data.sample(n=len(feature_delete_data)*2, replace= True)
            st.write("Random Sampling을 통해 데이터가 확장되었습니다.")
        
        elif chosen_method == 'Noise 추가 (Gaussian)':
            noise_factor = st.slider("노이즈 크기 설정 (0: 노이즈 없음, 1: 큰 노이즈)", 0.0, 1.0, step=0.01)
            noisy_data = feature_delete_data + noise_factor * np.random.randn(*feature_delete_data.shape)
            sampling_data = pd.concat([feature_delete_data, noisy_data], axis=0)
            st.write("Gaussian Noise를 추가하여 데이터가 확장되었습니다.")

        elif chosen_method == 'Jittering':
            jitter_factor = st.slider("Jittering 값 설정 (0: jitter 없음, 1: 큰 jitter)", 0.0, 1.0, step=0.01)
            jittered_data = feature_delete_data + jitter_factor * (np.random.rand(*feature_delete_data.shape) - 0.5)
            sampling_data = pd.concat([feature_delete_data, jittered_data], axis=0)
            st.write("Jittering을 통해 데이터가 확장되었습니다.")
  
        else:
            st.write("데이터 확장을 하지 않습니다.")


        # 데이터의 개수와 결측치 확인
    st.write("### Data Information:")
    buf = StringIO()
    sys.stdout = buf
    sampling_data.info()
    sys.stdout = sys.__stdout__
    st.text(buf.getvalue())

    # 데이터의 기술통계량 출력
    st.write("### 데이터의 기술통계량:")
    desc_stats = sampling_data.describe().transpose()
    st.write(desc_stats)

    sampling_data_copy = sampling_data.copy()

    st.session_state.my_data = sampling_data_copy.reset_index(drop=True)

    st.text('')
    
    
    if st.button("AutoML 이용해서 최적의 모델 선정"):
        st.session_state.page = "AutoML 이용해서 최적의 모델 선정"

            