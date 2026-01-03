import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 시계열 데이터 생성
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=252, freq='D')

# 주가 시뮬레이션
trend = np.linspace(100, 150, 252)
seasonal = 10 * np.sin(2 * np.pi * np.arange(252) / 252 * 4)
noise = np.random.normal(0, 5, 252)
price = trend + seasonal + noise

# DataFrame 생성
stock_df = pd.DataFrame({
    '종가': price,
    '거래량': np.random.randint(100000, 1000000, 252)
}, index=dates)

print("=== 시계열 데이터 분석 예제 ===")
print(f"데이터 기간: {stock_df.index.min()} ~ {stock_df.index.max()}")
print(f"데이터 개수: {len(stock_df)}")

# 이동평균 계산
stock_df['MA20'] = stock_df['종가'].rolling(window=20).mean()
stock_df['MA60'] = stock_df['종가'].rolling(window=60).mean()

# 수익률 계산
stock_df['일일수익률'] = stock_df['종가'].pct_change()
stock_df['로그수익률'] = np.log(stock_df['종가'] / stock_df['종가'].shift(1))

# 기본 통계
print("\n=== 기본 통계 ===")
print(f"평균 종가: {stock_df['종가'].mean():.2f}")
print(f"최고 종가: {stock_df['종가'].max():.2f}")
print(f"최저 종가: {stock_df['종가'].min():.2f}")
print(f"평균 일일수익률: {stock_df['일일수익률'].mean():.4f}")
print(f"연환산 변동성: {stock_df['일일수익률'].std() * np.sqrt(252):.4f}")

# 월별 리샘플링
monthly_data = stock_df.resample('M').agg({
    '종가': ['first', 'max', 'min', 'last'],
    '거래량': 'sum'
})

monthly_data.columns = ['시가', '고가', '저가', '종가', '총거래량']

print("\n=== 월별 데이터 ===")
print(monthly_data.head())

# 시각화
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(stock_df.index, stock_df['종가'], label='종가', alpha=0.7)
plt.plot(stock_df.index, stock_df['MA20'], label='20일 이동평균', linewidth=2)
plt.title('주가와 이동평균')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(stock_df.index, stock_df['로그수익률'])
plt.title('로그 수익률')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.hist(stock_df['로그수익률'].dropna(), bins=30, alpha=0.7)
plt.title('수익률 분포')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(monthly_data.index, monthly_data['종가'], 'o-', linewidth=2)
plt.title('월별 종가')
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n분석 완료!")
