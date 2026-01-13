# 3학년: 회귀 분석 및 다변량 자료 분석 예제

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Matplotlib에서 한글 깨짐 방지를 위한 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


# --- 회귀 분석 ---
print("--- 1. 회귀 분석 ---")

# 'students_data.csv' 파일에서 키(height)와 성적(score) 데이터 사용
# 키가 크면 성적이 좋을까? 라는 가설을 확인하기 위한 선형 회귀 모델
data = pd.read_csv('students_data.csv')
X = data[['height']] # 독립 변수 (키)
y = data['score']   # 종속 변수 (성적)

# 선형 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X, y)

# 학습된 모델의 계수(coefficient)와 절편(intercept)
coef = model.coef_[0]
intercept = model.intercept_

print(f"[단순 선형 회귀 분석: 키(X)와 성적(y)의 관계]")
print(f"회귀식: y = {coef:.2f} * X + {intercept:.2f}")
print(f"-> 키가 1cm 커질 때마다 성적이 약 {coef:.2f}점 오르거나 내리는 경향을 보입니다.")

# 시각화
plt.figure(figsize=(8, 5))
plt.scatter(X, y, label='실제 데이터')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='회귀선')
plt.title('키와 성적의 관계 (선형 회귀 분석)')
plt.xlabel('키 (cm)')
plt.ylabel('성적 (점)')
plt.legend()
plt.grid(True)
plt.show()


# --- 다변량 자료 분석 ---
print("\n\n--- 2. 다변량 자료 분석 ---")

# 'multivariate_data.csv' 파일 사용
m_data = pd.read_csv('multivariate_data.csv')
print("[다변량 데이터 샘플]")
print(m_data.head())

# 1. 상관관계 분석 (Correlation Matrix)
print("\n[상관관계 분석]")
# 변수 간의 상관계수 행렬 계산
corr_matrix = m_data.corr()

print("상관계수 행렬:")
print(corr_matrix.round(2))

# 히트맵으로 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('변수 간 상관관계 히트맵')
plt.show()
print("\n-> 히트맵을 통해 변수들 간의 양(+) 또는 음(-)의 상관관계를 한눈에 파악할 수 있습니다.")


# 2. 주성분 분석 (PCA, Principal Component Analysis)
print("\n[주성분 분석 (PCA)]")
# PCA는 여러 변수에 흩어져 있는 정보를 몇 개의 주성분(종합 변수)으로 요약하여
# 데이터의 차원을 축소하고 해석을 용이하게 하는 기법입니다.

# 데이터 표준화 (각 변수의 척도를 맞춤)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(m_data)

# PCA 모델 생성 및 학습 (주성분을 2개로 축소)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

# 주성분 분석 결과 데이터프레임으로 변환
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
print("주성분으로 변환된 데이터 (상위 2개):")
print(pca_df.head())

# 각 주성분이 원본 변수를 얼마나 설명하는지 (설명 분산 비율)
explained_variance = pca.explained_variance_ratio_
print(f"\n설명 분산 비율: PC1={explained_variance[0]:.2f}, PC2={explained_variance[1]:.2f}")
print(f"-> PC1과 PC2 두 주성분만으로 전체 데이터 분산의 약 {(sum(explained_variance)*100):.1f}%를 설명할 수 있습니다.")

# PCA 결과 시각화
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'])
plt.title('주성분 분석(PCA) 결과')
plt.xlabel('첫 번째 주성분 (PC1)')
plt.ylabel('두 번째 주성분 (PC2)')
plt.grid(True)
plt.show()
