import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def eda(scaling_data):
    st.header('2. EDA')

    st.text('')


    # 2-1. 기술통계량 계산
    st.subheader('2-1. 기술통계량')

        # 데이터의 기술통계량 출력
    desc_stats = scaling_data.describe().transpose()
    st.write(desc_stats)

    st.text('')

    #2-2 변수별 히스토 그램
    st.subheader('2-2. 변수별 분포 확인 (히스토그램)')
    select_column = st.selectbox('변수를 선택하세요', scaling_data.columns)
    fig, ax = plt.subplots()
    sns.histplot(scaling_data[select_column], kde=True, ax=ax)
    st.pyplot(fig)

    st.text('')


    # 2-3. 변수 간의 상관관계 확인
    st.subheader('2-3. 변수 간의 상관관계')

        # 상관계수 행렬 출력
    corr_matrix = scaling_data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

        # 상관계수가 높은 변수 쌍 찾기 (옵션)
    threshold = 0.8
    high_corr_var = set()

    for col1 in corr_matrix.columns:
        for col2 in corr_matrix.columns:
            if (abs(corr_matrix[col1][col2]) > threshold) & (col1 != col2):
                # 쌍을 정렬해서 동일한 쌍이라고 간주
                pair = tuple(sorted([col1, col2]))
                high_corr_var.add(pair)

    # set를 list로 변환
    high_corr_var = list(high_corr_var)
    corr_var_string = ""
    if high_corr_var:
        corr_var_string = ", ".join([f"({col1}, {col2})" for col1, col2 in high_corr_var])
        st.write(f"상관계수가 0.8 이상인 변수 쌍: [{corr_var_string}]")
        st.write("해당 feature들은 서로 강한 상관관계를 보입니다.")
    else:
        st.write("상관계수가 0.8 이상인 변수 쌍이 없습니다.")
    
    st.text('')

    # 2-4.산점도 
    st.subheader('2-4. 변수 간의 관계 확인 (산점도)')

    col1, col2 = st.selectbox('첫 번째 변수를 선택하세요', scaling_data.columns), st.selectbox('두 번째 변수를 선택하세요', scaling_data.columns)

        # 산점도와 함께 선형 회귀선을 그립니다.
    fig, ax = plt.subplots()
    sns.regplot(x=scaling_data[col1], y=scaling_data[col2], ax=ax)
    st.pyplot(fig)

        # 상관계수를 계산하여 출력합니다.
    correlation = scaling_data[col1].corr(scaling_data[col2])
    
    if correlation > 0:
        relation_type = "양의 상관관계"
    elif correlation < 0:
        relation_type = "음의 상관관계"
    else:
        relation_type = "상관관계가 없음"
    
    st.write(f"'{col1}'와 '{col2}'의 상관계수는 {correlation:.2f}로, {relation_type}를 가집니다.")

    st.session_state.high_corr_var = high_corr_var
    st.session_state.corr_var_string = corr_var_string

    st.text('')
    if st.button("중간 정리"):
        st.session_state.page = "중간 정리"