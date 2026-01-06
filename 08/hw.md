# Homework - 인터랙티브 대시보드

## 이 homework의 의미
사용자가 직접 조작하고 탐색할 수 있는 동적 대시보드를 구축하는 과제야. 정적인 차트를 넘어서 실시간으로 데이터를 분석하고 의사결정을 지원하는 도구를 만드는 거지.

## 관련 정보 사이트
1. **Streamlit Documentation**: https://docs.streamlit.io/
2. **Plotly Dash**: https://dash.plotly.com/

## 프로세스 진행 논리적 흐름
1. **요구사항 정의** → 어떤 데이터를 어떻게 보여줄지 설계
2. **프레임워크 선택** → Streamlit, Dash, Gradio 등
3. **UI 구성** → 위젯, 차트, 테이블 배치
4. **인터랙션 구현** → 필터, 슬라이더, 드롭다운 추가
5. **배포** → 클라우드에 대시보드 호스팅

## 권장 코드
```python
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# 페이지 설정
st.set_page_config(page_title="판매 대시보드", layout="wide")

# 타이틀
st.title("📊 실시간 판매 대시보드")

# 사이드바 필터
st.sidebar.header("필터 옵션")
date_range = st.sidebar.date_input("날짜 범위", [])
category = st.sidebar.multiselect("제품 카테고리", 
                                   ['전자제품', '의류', '식품'])

# 샘플 데이터
df = pd.DataFrame({
    '날짜': pd.date_range('2024-01-01', periods=100),
    '카테고리': np.random.choice(['전자제품', '의류', '식품'], 100),
    '매출': np.random.randint(100000, 1000000, 100)
})

# 메트릭 표시
col1, col2, col3 = st.columns(3)
col1.metric("총 매출", f"{df['매출'].sum():,}원")
col2.metric("평균 매출", f"{df['매출'].mean():,.0f}원")
col3.metric("거래 건수", len(df))

# 인터랙티브 차트
fig = px.line(df, x='날짜', y='매출', color='카테고리',
              title='일별 매출 추이')
st.plotly_chart(fig, use_container_width=True)

# 데이터 테이블
st.subheader("상세 데이터")
st.dataframe(df, use_container_width=True)
```

## 관련 업계 회사
1. **Streamlit (Snowflake)** - Python 대시보드 프레임워크
2. **Plotly** - 인터랙티브 시각화 및 Dash 프레임워크
3. **Looker (Google)** - 비즈니스 인텔리전스 플랫폼
