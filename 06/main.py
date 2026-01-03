import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# 결측치 포함 샘플 데이터 생성
data = {
    '이름': ['김철수', '이영희', '박민준', '최지아', '정서연', '강동훈', '윤미래'],
    '나이': [25, np.nan, 35, 28, 32, 45, np.nan],
    '성별': ['M', 'F', 'M', 'F', 'F', 'M', np.nan],
    '급여': [4500, 6200, np.nan, 5500, 4800, np.nan, 5200],
    '부서': ['영업', '개발', '마케팅', '개발', np.nan, '영업', '개발']
}
df = pd.DataFrame(data)

print("=== 데이터 클리닝과 전처리 예제 ===")
print("\n원본 데이터:")
print(df)

# 결측치 분석
print("\n=== 결측치 분석 ===")
print("결측치 개수:")
print(df.isnull().sum())
print("\n결측치 비율 (%):")
print((df.isnull().sum() / len(df) * 100).round(2))

# 결측치 처리
df_cleaned = df.copy()

# 수치형: 평균/중앙값 대체
df_cleaned['나이'].fillna(df_cleaned['나이'].mean(), inplace=True)
df_cleaned['급여'].fillna(df_cleaned['급여'].median(), inplace=True)

# 범주형: 최빈값 대체
df_cleaned['성별'].fillna(df_cleaned['성별'].mode()[0], inplace=True)
df_cleaned['부서'].fillna(df_cleaned['부서'].mode()[0], inplace=True)

print("\n=== 결측치 처리 후 ===")
print(df_cleaned)

# 이상치 탐지 (급여 기준)
print("\n=== 이상치 탐지 ===")
Q1 = df_cleaned['급여'].quantile(0.25)
Q3 = df_cleaned['급여'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"급여 IQR 경계: {lower_bound:.2f} ~ {upper_bound:.2f}")
outliers = df_cleaned[(df_cleaned['급여'] < lower_bound) | (df_cleaned['급여'] > upper_bound)]
print(f"이상치 개수: {len(outliers)}")

# 데이터 스케일링
numeric_data = df_cleaned[['나이', '급여']]

# StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)
scaled_df = pd.DataFrame(scaled_data, columns=['나이_표준화', '급여_표준화'])

print("\n=== 데이터 표준화 ===")
print("원본 데이터:")
print(numeric_data.describe())
print("\n표준화된 데이터:")
print(scaled_df.describe().round(2))

# 범주형 데이터 인코딩
print("\n=== 범주형 데이터 인코딩 ===")

# 성별 레이블 인코딩
le_gender = LabelEncoder()
df_cleaned['성별_인코딩'] = le_gender.fit_transform(df_cleaned['성별'])

# 부서 원-핫 인코딩
dept_dummies = pd.get_dummies(df_cleaned['부서'], prefix='부서')
df_encoded = pd.concat([df_cleaned, dept_dummies], axis=1)

print("인코딩 결과:")
print(df_encoded[['이름', '성별', '성별_인코딩'] + list(dept_dummies.columns)].head())

# 최종 정제된 데이터
final_data = df_encoded.drop(['성별', '부서'], axis=1)

print("\n=== 최종 정제된 데이터 ===")
print(final_data.head())
print(f"\n데이터 크기: {final_data.shape}")
print(f"메모리 사용량: {final_data.memory_usage(deep=True).sum() / 1024:.2f} KB")

# 데이터 품질 평가
completeness = (1 - final_data.isnull().sum().sum() / (final_data.shape[0] * final_data.shape[1])) * 100
print(f"\n데이터 완전성: {completeness:.2f}%")

print("\n데이터 전처리 완료!")
