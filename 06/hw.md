# Homework - 데이터 클리닝

## 이 homework의 의미
실제 데이터의 결측치, 이상치, 중복 등을 처리하여 분석 가능한 깨끗한 데이터로 만드는 과제야. 데이터 분석의 80%는 전처리라는 말이 있을 정도로 중요한 스킬이지.

## 관련 정보 사이트
1. **Data Cleaning with Python**: https://realpython.com/python-data-cleaning-numpy-pandas/
2. **Kaggle Data Cleaning Course**: https://www.kaggle.com/learn/data-cleaning

## 프로세스 진행 논리적 흐름
1. **데이터 탐색** → 결측치, 이상치, 중복 확인
2. **결측치 처리** → 삭제, 평균/중앙값 대체, 보간
3. **이상치 탐지** → IQR, Z-score 방법 활용
4. **데이터 정규화** → 스케일링 및 표준화
5. **검증** → 클리닝 결과 확인

## 권장 코드
```python
import pandas as pd
import numpy as np

# 더러운 데이터 생성
df = pd.DataFrame({
    '이름': ['Alice', 'Bob', None, 'David', 'Eve'],
    '나이': [25, 30, 35, 200, 28],  # 200은 이상치
    '급여': [50000, None, 60000, 55000, 58000]
})

# 결측치 확인
print("결측치 개수:")
print(df.isnull().sum())

# 결측치 처리
df['이름'].fillna('Unknown', inplace=True)
df['급여'].fillna(df['급여'].median(), inplace=True)

# 이상치 처리 (IQR 방법)
Q1 = df['나이'].quantile(0.25)
Q3 = df['나이'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df['나이'] = df['나이'].apply(lambda x: df['나이'].median() if x < lower or x > upper else x)

print("\n클리닝 후:")
print(df)
```

## 관련 업계 회사
1. **Palantir** - 대규모 데이터 통합 및 클리닝 솔루션
2. **Databricks** - 데이터 엔지니어링 플랫폼
3. **Snowflake** - 클라우드 데이터 웨어하우스 및 전처리
