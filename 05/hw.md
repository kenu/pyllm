# Homework - 시계열 데이터 분석

## 이 homework의 의미
시간에 따라 변화하는 데이터를 분석하고 예측하는 능력을 키우는 과제야. 주식, 날씨, 판매량 등 실생활의 다양한 시계열 데이터를 다루는 거지.

## 관련 정보 사이트
1. **Time Series Analysis in Python**: https://www.machinelearningplus.com/time-series/
2. **Prophet Documentation**: https://facebook.github.io/prophet/

## 프로세스 진행 논리적 흐름
1. **시계열 데이터 로드** → 날짜/시간 인덱스 설정
2. **트렌드 분석** → 장기적인 패턴 파악
3. **계절성 분석** → 주기적 패턴 탐지
4. **이동 평균** → 노이즈 제거 및 평활화
5. **예측 모델링** → ARIMA, Prophet 등 활용

## 권장 코드
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 시계열 데이터 생성
dates = pd.date_range('2023-01-01', periods=365, freq='D')
values = np.cumsum(np.random.randn(365)) + 100

df = pd.DataFrame({'날짜': dates, '값': values})
df.set_index('날짜', inplace=True)

# 이동 평균
df['MA_7'] = df['값'].rolling(window=7).mean()
df['MA_30'] = df['값'].rolling(window=30).mean()

# 시각화
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['값'], label='원본', alpha=0.5)
plt.plot(df.index, df['MA_7'], label='7일 이동평균')
plt.plot(df.index, df['MA_30'], label='30일 이동평균')
plt.legend()
plt.title('시계열 데이터 분석')
plt.show()
```

## 관련 업계 회사
1. **Tesla** - 에너지 사용량 및 차량 데이터 시계열 분석
2. **Amazon** - 수요 예측 및 재고 관리
3. **JP Morgan** - 금융 시계열 데이터 분석 및 트레이딩
