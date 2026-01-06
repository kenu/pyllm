# Homework - 데이터 시각화

## 이 homework의 의미
복잡한 데이터를 직관적인 차트와 그래프로 표현하여 인사이트를 전달하는 능력을 키우는 과제야. 데이터를 보는 사람이 쉽게 이해할 수 있도록 스토리텔링하는 거지.

## 관련 정보 사이트
1. **Matplotlib Gallery**: https://matplotlib.org/stable/gallery/
2. **Seaborn Tutorial**: https://seaborn.pydata.org/tutorial.html

## 프로세스 진행 논리적 흐름
1. **데이터 이해** → 어떤 인사이트를 전달할지 결정
2. **차트 선택** → 막대, 선, 산점도, 히트맵 등
3. **시각화 구현** → Matplotlib, Seaborn 활용
4. **스타일링** → 색상, 레이블, 제목 최적화
5. **해석 및 공유** → 결과 설명 및 리포트 작성

## 권장 코드
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 샘플 데이터
df = pd.DataFrame({
    '월': ['1월', '2월', '3월', '4월', '5월', '6월'],
    '매출': [120, 150, 180, 160, 200, 220],
    '비용': [80, 90, 100, 95, 110, 120]
})

# 서브플롯 생성
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 막대 그래프
axes[0, 0].bar(df['월'], df['매출'], color='skyblue')
axes[0, 0].set_title('월별 매출')
axes[0, 0].set_ylabel('매출 (만원)')

# 선 그래프
axes[0, 1].plot(df['월'], df['매출'], marker='o', label='매출')
axes[0, 1].plot(df['월'], df['비용'], marker='s', label='비용')
axes[0, 1].set_title('매출 vs 비용')
axes[0, 1].legend()

# 파이 차트
axes[1, 0].pie([df['매출'].sum(), df['비용'].sum()], 
               labels=['매출', '비용'], autopct='%1.1f%%')
axes[1, 0].set_title('매출/비용 비율')

# 히트맵
data = np.random.rand(5, 5)
sns.heatmap(data, annot=True, cmap='YlOrRd', ax=axes[1, 1])
axes[1, 1].set_title('상관관계 히트맵')

plt.tight_layout()
plt.show()
```

## 관련 업계 회사
1. **Tableau** - 비즈니스 인텔리전스 및 시각화 도구
2. **Plotly** - 인터랙티브 시각화 라이브러리 개발
3. **D3.js (Observable)** - 웹 기반 데이터 시각화
