import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # Mac
plt.rcParams['axes.unicode_minus'] = False

# 샘플 데이터 생성
np.random.seed(42)
n_samples = 200

data = pd.DataFrame({
    'x': np.random.randn(n_samples),
    'y': np.random.randn(n_samples) * 2 + np.random.randn(n_samples),
    'category': np.random.choice(['A', 'B', 'C'], n_samples),
    'group': np.random.choice(['Group1', 'Group2'], n_samples),
    'value': np.random.randint(1, 100, n_samples)
})

print("=== 데이터 시각화 마스터 예제 ===")
print(f"데이터 크기: {data.shape}")
print("\n데이터 샘플:")
print(data.head())

# 기본 통계
print("\n=== 기본 통계 ===")
print(data.describe())

# 1. 다양한 차트 종류
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('데이터 시각화 기법 모음', fontsize=16, fontweight='bold')

# 산점도
axes[0, 0].scatter(data['x'], data['y'], alpha=0.6, c=data['value'], cmap='viridis')
axes[0, 0].set_title('산점도')
axes[0, 0].grid(True, alpha=0.3)

# 히스토그램
axes[0, 1].hist(data['x'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 1].set_title('히스토그램')
axes[0, 1].grid(True, alpha=0.3)

# 박스플롯
axes[0, 2].boxplot([data[data['category'] == cat]['value'] for cat in ['A', 'B', 'C']], 
                   labels=['A', 'B', 'C'])
axes[0, 2].set_title('박스플롯')
axes[0, 2].grid(True, alpha=0.3)

# 막대그래프
category_means = data.groupby('category')['value'].mean()
axes[1, 0].bar(category_means.index, category_means.values, color=['red', 'green', 'blue'])
axes[1, 0].set_title('카테고리별 평균')
axes[1, 0].grid(True, alpha=0.3)

# 파이차트
category_counts = data['category'].value_counts()
axes[1, 1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
axes[1, 1].set_title('카테고리 비중')

# 라인 플롯
x_line = np.linspace(0, 10, 50)
y_line1 = np.sin(x_line)
y_line2 = np.cos(x_line)
axes[1, 2].plot(x_line, y_line1, label='sin(x)', linewidth=2)
axes[1, 2].plot(x_line, y_line2, label='cos(x)', linewidth=2)
axes[1, 2].set_title('라인 플롯')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 2. Seaborn 고급 시각화
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
sns.scatterplot(data=data, x='x', y='y', hue='category', style='group', size='value')
plt.title('Seaborn 산점도')

plt.subplot(2, 3, 2)
sns.histplot(data=data, x='x', hue='category', multiple='stack')
plt.title('히스토그램 (Seaborn)')

plt.subplot(2, 3, 3)
sns.boxplot(data=data, x='category', y='value', hue='group')
plt.title('박스플롯 (Seaborn)')

plt.subplot(2, 3, 4)
sns.violinplot(data=data, x='category', y='value')
plt.title('바이올린 플롯')

plt.subplot(2, 3, 5)
sns.barplot(data=data, x='category', y='value', estimator=np.mean)
plt.title('막대그래프 (Seaborn)')

plt.subplot(2, 3, 6)
sns.countplot(data=data, x='category', hue='group')
plt.title('카운트 플롯')

plt.tight_layout()
plt.show()

# 3. 상관관계 히트맵
correlation_data = data[['x', 'y', 'value']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, fmt='.2f')
plt.title('상관관계 히트맵')
plt.show()

# 4. 페어플롯
print("\n=== 페어플롯 (별도 창에서 표시) ===")
sns.pairplot(data, hue='category', vars=['x', 'y', 'value'])
plt.suptitle('변수 간 관계 페어플롯', y=1.02)
plt.show()

# 5. 시계열 데이터 시각화
dates = pd.date_range('2023-01-01', periods=100, freq='D')
time_series = pd.DataFrame({
    '날짜': dates,
    '매출': np.cumsum(np.random.randn(100) * 1000 + 50000) + 1000000,
    '방문자': np.cumsum(np.random.randn(100) * 50 + 200) + 5000
}).set_index('날짜')

fig, ax1 = plt.subplots(figsize=(12, 6))

# 첫 번째 y축 (매출)
ax1.plot(time_series.index, time_series['매출'], color='blue', linewidth=2, label='매출')
ax1.set_xlabel('날짜')
ax1.set_ylabel('매출 (원)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True, alpha=0.3)

# 두 번째 y축 (방문자)
ax2 = ax1.twinx()
ax2.plot(time_series.index, time_series['방문자'], color='red', linewidth=2, label='방문자')
ax2.set_ylabel('방문자 수', color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('시계열 데이터 (매출과 방문자)')
plt.show()

# 6. 통계 분석 시각화
plt.figure(figsize=(12, 8))

# 정규성 검정 Q-Q 플롯
plt.subplot(2, 2, 1)
from scipy import stats
stats.probplot(data['x'], dist="norm", plot=plt)
plt.title('Q-Q 플롯 (정규성 검정)')

# 커널 밀도 추정
plt.subplot(2, 2, 2)
sns.kdeplot(data=data, x='x', hue='category', fill=True)
plt.title('커널 밀도 추정')

# 잔차 플롯
plt.subplot(2, 2, 3)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(data[['x']], data['y'])
y_pred = model.predict(data[['x']])
residuals = data['y'] - y_pred.flatten()

plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('예측값')
plt.ylabel('잔차')
plt.title('잔차 플롯')
plt.grid(True, alpha=0.3)

# 누적 분포 함수
plt.subplot(2, 2, 4)
for category in ['A', 'B', 'C']:
    subset = data[data['category'] == category]['value']
    plt.hist(subset, bins=20, alpha=0.5, label=category, cumulative=True, density=True)
plt.xlabel('값')
plt.ylabel('누적 확률')
plt.title('누적 분포 함수')
plt.legend()

plt.tight_layout()
plt.show()

print("\n시각화 예제 완료!")
print("다양한 차트 유형과 Seaborn 고급 기법을 시연했습니다.")
