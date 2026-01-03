# 데이터 시각화 마스터 (Matplotlib/Seaborn)

## Matplotlib 기초
파이썬의 가장 기본적인 시각화 라이브러리인 Matplotlib의 핵심 기능을 마스터합니다.

### 1. 기본 플롯 생성
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 한글 폰트 설정 (Windows/Mac/Linux)
plt.rcParams['font.family'] = 'AppleGothic'  # Mac
# plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
# plt.rcParams['font.family'] = 'NanumGothic'  # Linux (나눔고딕 설치 필요)

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 기본 데이터
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.exp(-x/5) * np.sin(x)

# 기본 라인 플롯
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x)', linewidth=2)
plt.plot(x, y2, label='cos(x)', linewidth=2)
plt.plot(x, y3, label='damped sin(x)', linewidth=2, linestyle='--')

# 플롯 꾸미기
plt.title('삼각함수와 감쇠 진동', fontsize=16, fontweight='bold')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(0, 10)
plt.ylim(-1.5, 1.5)

plt.show()
```

### 2. 서브플롯 (Subplots)
```python
# 다양한 종류의 플롯 한 번에 그리기
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('다양한 시각화 기법', fontsize=16, fontweight='bold')

# 1. 라인 플롯
axes[0, 0].plot(x, y1, 'b-', linewidth=2)
axes[0, 0].set_title('라인 플롯')
axes[0, 0].grid(True, alpha=0.3)

# 2. 산점도
np.random.seed(42)
x_scatter = np.random.randn(100)
y_scatter = x_scatter * 2 + np.random.randn(100) * 0.5
axes[0, 1].scatter(x_scatter, y_scatter, alpha=0.6, c='red')
axes[0, 1].set_title('산점도')
axes[0, 1].grid(True, alpha=0.3)

# 3. 히스토그램
data_hist = np.random.normal(100, 15, 1000)
axes[0, 2].hist(data_hist, bins=30, alpha=0.7, color='green', edgecolor='black')
axes[0, 2].set_title('히스토그램')
axes[0, 2].grid(True, alpha=0.3)

# 4. 박스플롯
data_box = [np.random.normal(0, std, 100) for std in range(1, 4)]
axes[1, 0].boxplot(data_box, labels=['그룹1', '그룹2', '그룹3'])
axes[1, 0].set_title('박스플롯')
axes[1, 0].grid(True, alpha=0.3)

# 5. 막대그래프
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]
axes[1, 1].bar(categories, values, color=['red', 'green', 'blue', 'orange', 'purple'])
axes[1, 1].set_title('막대그래프')
axes[1, 1].grid(True, alpha=0.3)

# 6. 파이차트
sizes = [30, 25, 20, 15, 10]
labels = ['A', 'B', 'C', 'D', 'E']
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
axes[1, 2].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
axes[1, 2].set_title('파이차트')

plt.tight_layout()
plt.show()
```

## Seaborn 고급 시각화
통계 시각화에 특화된 Seaborn을 활용하여 더 전문적인 차트를 만듭니다.

### 1. Seaborn 기본 스타일
```python
import seaborn as sns

# Seaborn 스타일 설정
sns.set_style("whitegrid")
sns.set_palette("husl")

# 샘플 데이터 생성
np.random.seed(42)
data = pd.DataFrame({
    'x': np.random.randn(200),
    'y': np.random.randn(200),
    'category': np.random.choice(['A', 'B', 'C'], 200),
    'group': np.random.choice(['Group1', 'Group2'], 200),
    'value': np.random.randint(1, 100, 200)
})

# Seaborn 산점도
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
sns.scatterplot(data=data, x='x', y='y', hue='category', style='group', size='value')
plt.title('Seaborn 산점도')

plt.subplot(2, 3, 2)
sns.histplot(data=data, x='x', hue='category', multiple='stack')
plt.title('히스토그램')

plt.subplot(2, 3, 3)
sns.boxplot(data=data, x='category', y='x', hue='group')
plt.title('박스플롯')

plt.subplot(2, 3, 4)
sns.violinplot(data=data, x='category', y='y')
plt.title('바이올린 플롯')

plt.subplot(2, 3, 5)
sns.barplot(data=data, x='category', y='value', estimator=np.mean)
plt.title('막대그래프')

plt.subplot(2, 3, 6)
sns.countplot(data=data, x='category', hue='group')
plt.title('카운트 플롯')

plt.tight_layout()
plt.show()
```

### 2. 상관관계 시각화
```python
# 상관관계 데이터 생성
corr_data = pd.DataFrame({
    '수학': np.random.normal(80, 10, 100),
    '영어': np.random.normal(75, 12, 100),
    '과학': np.random.normal(78, 15, 100),
    '국어': np.random.normal(82, 8, 100),
    '사회': np.random.normal(77, 11, 100)
})

# 일부러 상관관계 추가
corr_data['영어'] = corr_data['영어'] + corr_data['수학'] * 0.3 + np.random.normal(0, 5, 100)
corr_data['과학'] = corr_data['과학'] + corr_data['수학'] * 0.4 + np.random.normal(0, 5, 100)

# 상관관계 행렬
correlation_matrix = corr_data.corr()

# 히트맵
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, fmt='.2f')
plt.title('상관관계 히트맵')

# 페어플롯
plt.subplot(1, 2, 2)
# 페어플롯은 별도의 figure에서 그려야 함
plt.text(0.5, 0.5, 'Pair Plot\n(별도 창에서 표시)', 
         ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
plt.axis('off')

plt.tight_layout()
plt.show()

# 페어플롯 (별도 창)
sns.pairplot(corr_data)
plt.suptitle('과목별 점수 페어플롯', y=1.02)
plt.show()
```

## 전문적인 차트 디자인
실제 보고서와 발표자료에 사용할 수 있는 수준의 차트를 만듭니다.

### 1. 커스텀 스타일 적용
```python
# 커스텀 스타일 정의
def custom_style():
    """커스텀 스타일 설정"""
    plt.style.use('seaborn-v0_8')
    
    # 색상 팔레트
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']
    return colors

# 시계열 데이터 시각화
dates = pd.date_range('2023-01-01', periods=365, freq='D')
sales = np.cumsum(np.random.randn(365) * 1000 + 50000) + 1000000
visitors = np.cumsum(np.random.randn(365) * 50 + 200) + 5000

colors = custom_style()

fig, ax1 = plt.subplots(figsize=(14, 8))

# 첫 번째 y축 (매출)
ax1.plot(dates, sales, color=colors[0], linewidth=2.5, label='매출')
ax1.set_xlabel('날짜', fontsize=12, fontweight='bold')
ax1.set_ylabel('매출 (원)', fontsize=12, fontweight='bold', color=colors[0])
ax1.tick_params(axis='y', labelcolor=colors[0])
ax1.grid(True, alpha=0.3)

# 두 번째 y축 (방문자)
ax2 = ax1.twinx()
ax2.plot(dates, visitors, color=colors[1], linewidth=2.5, label='방문자')
ax2.set_ylabel('방문자 수', fontsize=12, fontweight='bold', color=colors[1])
ax2.tick_params(axis='y', labelcolor=colors[1])

# 제목과 범례
plt.title('2023년 매출과 방문자 추이', fontsize=16, fontweight='bold', pad=20)

# 범례 합치기
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

# x축 날짜 포맷
import matplotlib.dates as mdates
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator())

plt.tight_layout()
plt.show()
```

### 2. 인터랙티브 요소 추가
```python
# 주석과 강조 표시
plt.figure(figsize=(12, 8))

# 데이터 준비
months = ['1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월']
revenue = [120, 135, 125, 145, 160, 155, 170, 165, 180, 175, 190, 200]
target = [150] * 12

# 막대그래프
bars = plt.bar(months, revenue, color=colors[0], alpha=0.8, width=0.6)

# 목표선
plt.plot(months, target, color=colors[2], linestyle='--', linewidth=2, label='목표')

# 특정 월 강조
highlight_months = ['5월', '8월', '12월']
for i, month in enumerate(months):
    if month in highlight_months:
        bars[i].set_color(colors[1])
        bars[i].set_alpha(1.0)
        
        # 주석 추가
        plt.annotate(f'최고: {revenue[i]}만원', 
                    xy=(i, revenue[i]), 
                    xytext=(i, revenue[i] + 10),
                    ha='center',
                    arrowprops=dict(arrowstyle='->', color='red'))

# 스타일링
plt.title('2023년 월별 매출 현황', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('월', fontsize=12, fontweight='bold')
plt.ylabel('매출 (만원)', fontsize=12, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, axis='y')

# y축 범위 설정
plt.ylim(0, max(revenue) * 1.2)

plt.tight_layout()
plt.show()
```

## 통계 차트 마스터
다양한 통계 분석 결과를 시각화하는 방법을 익힙니다.

### 1. 분포 시각화
```python
# 다양한 분포 비교
np.random.seed(42)

# 데이터 생성
normal_data = np.random.normal(100, 15, 1000)
uniform_data = np.random.uniform(50, 150, 1000)
exponential_data = np.random.exponential(30, 1000)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('다양한 분포 시각화', fontsize=16, fontweight='bold')

# 히스토그램
axes[0, 0].hist(normal_data, bins=30, alpha=0.7, color='blue', density=True)
axes[0, 0].set_title('정규분포 히스토그램')

# 커널 밀도 추정
sns.kdeplot(data=normal_data, ax=axes[0, 1], color='blue', label='정규분포')
sns.kdeplot(data=uniform_data, ax=axes[0, 1], color='red', label='균등분포')
sns.kdeplot(data=exponential_data, ax=axes[0, 1], color='green', label='지수분포')
axes[0, 1].set_title('커널 밀도 추정')
axes[0, 1].legend()

# Q-Q 플롯
from scipy import stats
stats.probplot(normal_data, dist="norm", plot=axes[0, 2])
axes[0, 2].set_title('Q-Q 플롯 (정규성 검정)')

# 박스플롯과 바이올린 플롯 비교
combined_data = pd.DataFrame({
    '값': np.concatenate([normal_data, uniform_data, exponential_data]),
    '분포': ['정규분포'] * 1000 + ['균등분포'] * 1000 + ['지수분포'] * 1000
})

sns.boxplot(data=combined_data, x='분포', y='값', ax=axes[1, 0])
axes[1, 0].set_title('박스플롯')

sns.violinplot(data=combined_data, x='분포', y='값', ax=axes[1, 1])
axes[1, 1].set_title('바이올린 플롯')

# 누적 분포 함수
axes[1, 2].hist(normal_data, bins=30, alpha=0.7, color='blue', cumulative=True, density=True)
axes[1, 2].set_title('누적 분포 함수')

plt.tight_layout()
plt.show()
```

### 2. 회귀 분석 시각화
```python
# 회귀 분석 시각화
np.random.seed(42)

# 데이터 생성
x_reg = np.linspace(0, 10, 100)
y_reg = 2 * x_reg + 1 + np.random.normal(0, 2, 100)

# 다항 회귀 데이터
x_poly = np.linspace(-3, 3, 100)
y_poly = x_poly**2 + np.random.normal(0, 1, 100)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 선형 회귀
sns.regplot(x=x_reg, y=y_reg, ax=axes[0, 0], 
            scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
axes[0, 0].set_title('선형 회귀')

# 잔차 플롯
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_reg.reshape(-1, 1), y_reg)
y_pred = model.predict(x_reg.reshape(-1, 1))
residuals = y_reg - y_pred

axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
axes[0, 1].axhline(y=0, color='red', linestyle='--')
axes[0, 1].set_xlabel('예측값')
axes[0, 1].set_ylabel('잔차')
axes[0, 1].set_title('잔차 플롯')

# 다항 회귀
sns.regplot(x=x_poly, y=y_poly, ax=axes[1, 0], order=2,
            scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
axes[1, 0].set_title('다항 회귀 (2차)')

# 신뢰 구간
sns.regplot(x=x_reg, y=y_reg, ax=axes[1, 1], 
            scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'},
            ci=95)  # 95% 신뢰구간
axes[1, 1].set_title('신뢰 구간 포함 회귀')

plt.tight_layout()
plt.show()
```

## 실전 프로젝트: 비즈니스 대시보드
실제 비즈니스 데이터를 시각화하는 종합 예제입니다.

### 1. 판매 데이터 대시보드
```python
# 판매 데이터 생성
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=365, freq='D')

categories = ['전자제품', '의류', '식품', '가구', '도서']
regions = ['서울', '부산', '대구', '대전', '광주']

# 일별 판매 데이터
sales_data = []
for date in dates:
    for _ in range(np.random.randint(50, 200)):  # 일별 거래 수
        sales_data.append({
            '날짜': date,
            '카테고리': np.random.choice(categories),
            '지역': np.random.choice(regions),
            '수량': np.random.randint(1, 10),
            '단가': np.random.randint(1000, 50000),
            '고객등급': np.random.choice(['VIP', '일반', '신규'])
        })

df_sales = pd.DataFrame(sales_data)
df_sales['총액'] = df_sales['수량'] * df_sales['단가']

# 대시보드 생성
fig = plt.figure(figsize=(20, 16))
fig.suptitle('2023년 판매 분석 대시보드', fontsize=20, fontweight='bold')

# 1. 월별 매출 추이
monthly_sales = df_sales.groupby(df_sales['날짜'].dt.month)['총액'].sum()

ax1 = plt.subplot(3, 3, 1)
monthly_sales.plot(kind='line', ax=ax1, color=colors[0], linewidth=3, marker='o')
ax1.set_title('월별 매출 추이', fontweight='bold')
ax1.set_xlabel('월')
ax1.set_ylabel('매출 (원)')
ax1.grid(True, alpha=0.3)
ax1.ticklabel_format(style='plain', axis='y')

# 2. 카테고리별 매출
category_sales = df_sales.groupby('카테고리')['총액'].sum().sort_values(ascending=False)

ax2 = plt.subplot(3, 3, 2)
category_sales.plot(kind='bar', ax=ax2, color=colors)
ax2.set_title('카테고리별 매출', fontweight='bold')
ax2.set_xlabel('카테고리')
ax2.set_ylabel('매출 (원)')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3)

# 3. 지역별 매출 파이차트
region_sales = df_sales.groupby('지역')['총액'].sum()

ax3 = plt.subplot(3, 3, 3)
wedges, texts, autotexts = ax3.pie(region_sales, labels=region_sales.index, autopct='%1.1f%%', 
                                  startangle=90, colors=colors)
ax3.set_title('지역별 매출 비중', fontweight='bold')

# 4. 고객등급별 분포
customer_dist = df_sales['고객등급'].value_counts()

ax4 = plt.subplot(3, 3, 4)
customer_dist.plot(kind='bar', ax=ax4, color=['gold', 'silver', 'lightcoral'])
ax4.set_title('고객등급별 거래 수', fontweight='bold')
ax4.set_xlabel('고객등급')
ax4.set_ylabel('거래 수')
ax4.grid(True, alpha=0.3)

# 5. 요일별 판매량
df_sales['요일'] = df_sales['날짜'].dt.day_name()
weekday_sales = df_sales.groupby('요일')['수량'].sum()
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_sales = weekday_sales.reindex(weekday_order)

ax5 = plt.subplot(3, 3, 5)
weekday_sales.plot(kind='bar', ax=ax5, color='lightgreen')
ax5.set_title('요일별 판매량', fontweight='bold')
ax5.set_xlabel('요일')
ax5.set_ylabel('판매량')
ax5.tick_params(axis='x', rotation=45)
ax5.grid(True, alpha=0.3)

# 6. 카테고리-지역 히트맵
category_region = df_sales.pivot_table(values='총액', index='카테고리', columns='지역', aggfunc='sum')

ax6 = plt.subplot(3, 3, 6)
sns.heatmap(category_region, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax6)
ax6.set_title('카테고리-지역 매출 히트맵', fontweight='bold')

# 7. 단가 분포
ax7 = plt.subplot(3, 3, 7)
ax7.hist(df_sales['단가'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
ax7.set_title('단가 분포', fontweight='bold')
ax7.set_xlabel('단가 (원)')
ax7.set_ylabel('빈도')
ax7.grid(True, alpha=0.3)

# 8. 수량-총액 산점도
sample_data = df_sales.sample(1000)  # 샘플링하여 표시

ax8 = plt.subplot(3, 3, 8)
scatter = ax8.scatter(sample_data['수량'], sample_data['총액'], 
                     c=sample_data['단가'], alpha=0.6, cmap='viridis')
ax8.set_title('수량-총액 관계', fontweight='bold')
ax8.set_xlabel('수량')
ax8.set_ylabel('총액 (원)')
ax8.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax8, label='단가')

# 9. 월별 카테고리 누적 막대그래프
monthly_category = df_sales.groupby([df_sales['날짜'].dt.month, '카테고리'])['총액'].sum().unstack()

ax9 = plt.subplot(3, 3, 9)
monthly_category.plot(kind='bar', stacked=True, ax=ax9, colormap='tab10')
ax9.set_title('월별 카테고리 매출 (누적)', fontweight='bold')
ax9.set_xlabel('월')
ax9.set_ylabel('매출 (원)')
ax9.legend(title='카테고리', bbox_to_anchor=(1.05, 1), loc='upper left')
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 2. KPI 지표 시각화
```python
# KPI 지표 계산
total_revenue = df_sales['총액'].sum()
total_transactions = len(df_sales)
avg_transaction_value = total_revenue / total_transactions
top_category = category_sales.index[0]
top_region = region_sales.index[0]

# KPI 대시보드
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('핵심 성과 지표 (KPI)', fontsize=18, fontweight='bold')

# 총매출
axes[0, 0].text(0.5, 0.7, f'총매출', ha='center', va='center', fontsize=16, 
               transform=axes[0, 0].transAxes)
axes[0, 0].text(0.5, 0.4, f'{total_revenue:,.0f}원', ha='center', va='center', 
               fontsize=24, fontweight='bold', color='green',
               transform=axes[0, 0].transAxes)
axes[0, 0].set_xlim(0, 1)
axes[0, 0].set_ylim(0, 1)
axes[0, 0].axis('off')

# 총거래수
axes[0, 1].text(0.5, 0.7, f'총거래수', ha='center', va='center', fontsize=16,
               transform=axes[0, 1].transAxes)
axes[0, 1].text(0.5, 0.4, f'{total_transactions:,}건', ha='center', va='center',
               fontsize=24, fontweight='bold', color='blue',
               transform=axes[0, 1].transAxes)
axes[0, 1].set_xlim(0, 1)
axes[0, 1].set_ylim(0, 1)
axes[0, 1].axis('off')

# 평균 거래액
axes[1, 0].text(0.5, 0.7, f'평균 거래액', ha='center', va='center', fontsize=16,
               transform=axes[1, 0].transAxes)
axes[1, 0].text(0.5, 0.4, f'{avg_transaction_value:,.0f}원', ha='center', va='center',
               fontsize=24, fontweight='bold', color='orange',
               transform=axes[1, 0].transAxes)
axes[1, 0].set_xlim(0, 1)
axes[1, 0].set_ylim(0, 1)
axes[1, 0].axis('off')

# 최상위 카테고리 및 지역
axes[1, 1].text(0.5, 0.7, f'최상위 카테고리', ha='center', va='center', fontsize=16,
               transform=axes[1, 1].transAxes)
axes[1, 1].text(0.5, 0.5, f'{top_category}', ha='center', va='center',
               fontsize=20, fontweight='bold', color='purple',
               transform=axes[1, 1].transAxes)
axes[1, 1].text(0.5, 0.3, f'최상위 지역: {top_region}', ha='center', va='center',
               fontsize=14,
               transform=axes[1, 1].transAxes)
axes[1, 1].set_xlim(0, 1)
axes[1, 1].set_ylim(0, 1)
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()
```

## 고급 시각화 기법
전문적인 데이터 시각화를 위한 고급 기법들을 익힙니다.

### 1. 애니메이션 차트
```python
# 시간에 따른 변화 애니메이션 (개념 예시)
from matplotlib.animation import FuncAnimation

# 월별 데이터 준비
monthly_data = []
for month in range(1, 13):
    month_data = df_sales[df_sales['날짜'].dt.month == month]
    monthly_data.append(month_data.groupby('카테고리')['총액'].sum())

# 애니메이션 함수 (실제 실행은 주피터 노트북에서)
def animate(month):
    """애니메이션 프레임 함수"""
    plt.clf()
    monthly_data[month].plot(kind='bar', color=colors)
    plt.title(f'{month}월 카테고리별 매출')
    plt.xlabel('카테고리')
    plt.ylabel('매출 (원)')
    plt.xticks(rotation=45)

# 애니메이션 생성 (주피터에서 실행)
# anim = FuncAnimation(plt.figure(), animate, frames=12, interval=1000, repeat=True)
```

### 2. 3D 시각화
```python
from mpl_toolkits.mplot3d import Axes3D

# 3D 산점도
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 3D 데이터 생성
np.random.seed(42)
x_3d = np.random.randn(100)
y_3d = np.random.randn(100)
z_3d = x_3d + y_3d + np.random.randn(100) * 0.5

# 카테고리별 색상
colors_3d = np.random.choice(['red', 'blue', 'green'], 100)

scatter = ax.scatter(x_3d, y_3d, z_3d, c=colors_3d, alpha=0.6, s=50)

ax.set_xlabel('X축')
ax.set_ylabel('Y축')
ax.set_zlabel('Z축')
ax.set_title('3D 산점도', fontweight='bold')

plt.show()
```

## 이미지 저장 및 공유
시각화 결과를 저장하고 공유하는 방법을 익힙니다.

```python
# 다양한 형식으로 이미지 저장
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot([1, 2, 3, 4], [1, 4, 2, 3], 'o-', linewidth=2)
ax.set_title('샘플 차트')
ax.grid(True, alpha=0.3)

# 다양한 형식으로 저장
plt.savefig('/Users/kenu/git/pyllm/07/sample_chart.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/kenu/git/pyllm/07/sample_chart.pdf', bbox_inches='tight')
plt.savefig('/Users/kenu/git/pyllm/07/sample_chart.svg', bbox_inches='tight')

print("차트가 다양한 형식으로 저장되었습니다:")
print("- PNG (고해상도)")
print("- PDF (벡터)")
print("- SVG (벡터)")

plt.show()
```

이 시각화 기법들을 통해 데이터의 의미를 효과적으로 전달하고, 전문적인 보고서와 발표 자료를 만들 수 있습니다.
