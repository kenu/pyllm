# 시계열 데이터 다루기

## 시계열 데이터 기초
시간에 따른 데이터 변화를 분석하고 예측하는 방법을 학습합니다.

### 1. Datetime 인덱스 생성
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 다양한 방법으로 DatetimeIndex 생성
# 1. date_range 사용
dates1 = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
dates2 = pd.date_range(start='2023-01-01', periods=365, freq='D')
dates3 = pd.date_range(start='2023-01-01', periods=12, freq='M')  # 월말 기준

# 2. 특정 빈도
daily = pd.date_range('2023-01-01', periods=10, freq='D')  # 일별
weekly = pd.date_range('2023-01-01', periods=10, freq='W')  # 주별
monthly = pd.date_range('2023-01-01', periods=10, freq='M')  # 월별
quarterly = pd.date_range('2023-01-01', periods=10, freq='Q')  # 분기별
yearly = pd.date_range('2023-01-01', periods=10, freq='Y')  # 연별

print("일별 날짜:")
print(daily[:5])
print("\n주별 날짜:")
print(weekly[:5])
```

### 2. 시계열 데이터 생성
```python
# 샘플 주가 데이터 생성
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=252, freq='D')  # 1년 영업일

# 추세 + 계절성 + 노이즈
trend = np.linspace(100, 150, 252)
seasonal = 10 * np.sin(2 * np.pi * np.arange(252) / 252 * 4)  # 연 4회 주기
noise = np.random.normal(0, 5, 252)
price = trend + seasonal + noise

# 시계열 DataFrame 생성
stock_df = pd.DataFrame({
    '날짜': dates,
    '종가': price,
    '거래량': np.random.randint(100000, 1000000, 252)
}).set_index('날짜')

print("주가 데이터:")
print(stock_df.head())
print(f"\n데이터 기간: {stock_df.index.min()} ~ {stock_df.index.max()}")
```

## 시계열 데이터 조작
시간 기반 데이터를 효과적으로 필터링, 리샘플링, 변환합니다.

### 1. 시간 기반 필터링
```python
# 특정 기간 필터링
q1_2023 = stock_df['2023-01-01':'2023-03-31']
q2_2023 = stock_df['2023-Q2']
first_half = stock_df['2023-H1']

# 특정 월/요일 필터링
january = stock_df[stock_df.index.month == 1]
mondays = stock_df[stock_df.index.dayofweek == 0]  # Monday=0

# 특정 시간대 (분데이터일 경우)
# morning_trades = stock_df.between_time('09:00', '12:00')

print("1분기 데이터:")
print(q1_2023.head())
print(f"\n1분기 평균 종가: {q1_2023['종가'].mean():.2f}")
```

### 2. 리샘플링 (Resampling)
```python
# 일별 -> 주별/월별로 변환
weekly_data = stock_df.resample('W').agg({
    '종가': ['first', 'max', 'min', 'last'],  # 시가, 고가, 저가, 종가
    '거래량': 'sum'
})

monthly_data = stock_df.resample('M').agg({
    '종가': ['mean', 'std'],
    '거래량': 'sum'
})

# 컬럼 이름 정리
weekly_data.columns = ['시가', '고가', '저가', '종가', '총거래량']
monthly_data.columns = ['평균종가', '종가표준편차', '월별거래량']

print("\n주별 데이터:")
print(weekly_data.head())
print("\n월별 데이터:")
print(monthly_data.head())
```

### 3. 이동 평균 및 롤링
```python
# 다양한 이동평균
stock_df['MA5'] = stock_df['종가'].rolling(window=5).mean()
stock_df['MA20'] = stock_df['종가'].rolling(window=20).mean()
stock_df['MA60'] = stock_df['종가'].rolling(window=60).mean()

# 이동 표준편차 (변동성)
stock_df['Volatility20'] = stock_df['종가'].rolling(window=20).std()

# 지수 가중 이동평균 (EMA)
stock_df['EMA12'] = stock_df['종가'].ewm(span=12).mean()
stock_df['EMA26'] = stock_df['종가'].ewm(span=26).mean()

# MACD
stock_df['MACD'] = stock_df['EMA12'] - stock_df['EMA26']
stock_df['Signal'] = stock_df['MACD'].ewm(span=9).mean()

print("\n이동평균 결과:")
print(stock_df[['종가', 'MA5', 'MA20', 'MA60']].tail(10))
```

## 시계열 분석 기법
시계열 데이터의 패턴을 분석하고 예측합니다.

### 1. 추세 분석
```python
# 선형 추세 추정
from scipy import stats

# 시간 인덱스를 숫자로 변환 (0, 1, 2, ...)
time_index = np.arange(len(stock_df))
slope, intercept, r_value, p_value, std_err = stats.linregress(time_index, stock_df['종가'])

# 추선선
trend_line = slope * time_index + intercept

stock_df['추세'] = trend_line

print(f"추세 분석 결과:")
print(f"기울기: {slope:.4f} (일일 평균 변화)")
print(f"R-squared: {r_value**2:.4f}")
print(f"P-value: {p_value:.4f}")

# 추세 제거 데이터
stock_df['추세제거'] = stock_df['종가'] - stock_df['추세']
```

### 2. 계절성 분석
```python
# 월별 평균으로 계절성 확인
monthly_avg = stock_df.groupby(stock_df.index.month)['종가'].mean()

# 계절성 지수
overall_mean = stock_df['종가'].mean()
seasonal_index = monthly_avg / overall_mean

print("\n월별 계절성 지수:")
for month, index in seasonal_index.items():
    print(f"{month}월: {index:.3f}")

# 계절성 제거
stock_df['계절성'] = stock_df.index.map(lambda x: seasonal_index[x.month])
stock_df['계절성제거'] = stock_df['종가'] / stock_df['계절성']
```

### 3. 차분 (Differencing)
```python
# 1차 차분
stock_df['1차차분'] = stock_df['종가'].diff()

# 계절 차분 (월별)
stock_df['계절차분'] = stock_df['종가'].diff(periods=30)  # 30일 차분

# 로그 수익률
stock_df['로그수익률'] = np.log(stock_df['종가'] / stock_df['종가'].shift(1))

print("\n차분 결과:")
print(stock_df[['종가', '1차차분', '로그수익률']].head(10))
```

## 시계열 시각화
시간에 따른 데이터 변화를 시각적으로 분석합니다.

```python
# 기본 시계열 플롯
plt.figure(figsize=(15, 10))

# 1. 종가와 이동평균
plt.subplot(2, 2, 1)
plt.plot(stock_df.index, stock_df['종가'], label='종가', alpha=0.7)
plt.plot(stock_df.index, stock_df['MA20'], label='20일 이동평균', linewidth=2)
plt.plot(stock_df.index, stock_df['MA60'], label='60일 이동평균', linewidth=2)
plt.title('주가와 이동평균')
plt.legend()
plt.grid(True)

# 2. 거래량
plt.subplot(2, 2, 2)
plt.bar(stock_df.index, stock_df['거래량'], alpha=0.6)
plt.title('일별 거래량')
plt.grid(True)

# 3. 변동성
plt.subplot(2, 2, 3)
plt.plot(stock_df.index, stock_df['Volatility20'], color='red')
plt.title('20일 변동성')
plt.grid(True)

# 4. 로그 수익률 분포
plt.subplot(2, 2, 4)
plt.hist(stock_df['로그수익률'].dropna(), bins=50, alpha=0.7)
plt.title('로그 수익률 분포')
plt.grid(True)

plt.tight_layout()
plt.show()
```

## 실전 예제: 주식 데이터 분석
```python
# 실제 분석 시나리오
def 주식_분석_보고서(df):
    """주식 데이터 분석 보고서 생성"""
    
    보고서 = {}
    
    # 기본 통계
    보고서['기본정보'] = {
        '분석기간': f"{df.index.min().date()} ~ {df.index.max().date()}",
        '총거래일수': len(df),
        '평균종가': df['종가'].mean(),
        '최고종가': df['종가'].max(),
        '최저종가': df['종가'].min(),
        '변동성': df['종가'].std()
    }
    
    # 수익률 분석
    수익률 = df['로그수익률'].dropna()
    보고서['수익률분석'] = {
        '연환산수익률': 수익률.mean() * 252,  # 연 252거래일 기준
        '연환산변동성': 수익률.std() * np.sqrt(252),
        '샤프지수': (수익률.mean() * 252) / (수익률.std() * np.sqrt(252))
    }
    
    # 추세 분석
    slope, _, r_value, _, _ = stats.linregress(np.arange(len(df)), df['종가'])
    보고서['추세분석'] = {
        '일평균변화': slope,
        '추세강도': r_value**2,
        '추세방향': '상승' if slope > 0 else '하락'
    }
    
    return 보고서

# 보고서 생성
분석보고서 = 주식_분석_보고서(stock_df)

print("=== 주식 분석 보고서 ===")
for 섹션, 정보 in 분석보고서.items():
    print(f"\n{섹션}:")
    for 항목, 값 in 정보.items():
        if isinstance(값, float):
            print(f"  {항목}: {값:.4f}")
        else:
            print(f"  {항목}: {값}")
```

## 시계열 예측 기초
간단한 예측 모델을 구현합니다.

### 1. 단순 예측 모델
```python
# 이동평균 기반 예측
def moving_average_forecast(data, window=20, forecast_periods=5):
    """이동평균 기반 예측"""
    last_ma = data.rolling(window=window).mean().iloc[-1]
    
    # 마지막 추세 기반 예측
    recent_trend = data.diff(window).iloc[-1] / window
    
    forecasts = []
    for i in range(forecast_periods):
        forecast = last_ma + recent_trend * (i + 1)
        forecasts.append(forecast)
    
    return forecasts

# 예측 실행
forecasts = moving_average_forecast(stock_df['종가'], window=20, forecast_periods=5)

print("\n이동평균 기반 5일 예측:")
for i, forecast in enumerate(forecasts, 1):
    print(f"Day +{i}: {forecast:.2f}")
```

### 2. 계절성을 고려한 예측
```python
# 계절성 기반 예측
def seasonal_forecast(data, periods=5):
    """계절성 패턴 기반 예측"""
    # 최근 30일 패턴 저장
    recent_pattern = data.tail(30).values
    
    # 마지막 5일 패턴으로 미래 예측
    last_5_days = recent_pattern[-5:]
    
    # 패턴 반복으로 예측 (단순화)
    forecasts = []
    for i in range(periods):
        # 최근 패턴의 평균 변화 계산
        if i < len(last_5_days) - 1:
            change = last_5_days[i+1] - last_5_days[i]
        else:
            change = np.mean(np.diff(last_5_days))
        
        last_value = forecasts[-1] if forecasts else data.iloc[-1]
        forecast = last_value + change
        forecasts.append(forecast)
    
    return forecasts

seasonal_forecasts = seasonal_forecast(stock_df['종가'], periods=5)

print("\n계절성 기반 5일 예측:")
for i, forecast in enumerate(seasonal_forecasts, 1):
    print(f"Day +{i}: {forecast:.2f}")
```

## 고급 시계열 기법
```python
# 자기상관 분석
from pandas.plotting import autocorrelation_plot

# 자기상관 함수 (ACF)
def autocorrelation_analysis(series, lags=20):
    """자기상관 분석"""
    autocorr_values = [series.autocorr(lag=i) for i in range(lags + 1)]
    
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(autocorr_values)), autocorr_values)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='유의수준')
    plt.axhline(y=-0.2, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('자기상관 함수 (ACF)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return autocorr_values

# 자기상관 분석 실행
autocorr_values = autocorrelation_analysis(stock_df['종가'], lags=20)

# 유의한 자기상관 찾기
significant_lags = [i for i, val in enumerate(autocorr_values) if abs(val) > 0.2]
print(f"\n유의한 자기상관 Lag: {significant_lags}")
```

## 성능 최적화
```python
# 대용량 시계열 데이터 처리 최적화
def optimize_timeseries_memory(df):
    """시계열 데이터 메모리 최적화"""
    
    # datetime 인덱스 최적화
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.astype('datetime64[s]')  # 초 단위로 정밀도 조정
    
    # 수치형 데이터 다운캐스팅
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    return df

# 메모리 사용량 확인
def memory_usage(df):
    """DataFrame 메모리 사용량 확인"""
    return df.memory_usage(deep=True).sum() / 1024**2  # MB

print(f"\n최적화 전 메모리: {memory_usage(stock_df):.2f} MB")
optimized_df = optimize_timeseries_memory(stock_df.copy())
print(f"최적화 후 메모리: {memory_usage(optimized_df):.2f} MB")
```

## 실전 프로젝트: 주식 포트폴리오 분석
```python
# 여러 주식의 포트폴리오 분석
def create_portfolio_data():
    """포트폴리오 데이터 생성"""
    stocks = ['삼성전자', 'SK하이닉스', 'NAVER', '카카오', 'LG화학']
    portfolio_data = {}
    
    for stock in stocks:
        # 각 주식별로 다른 패턴 생성
        np.random.seed(hash(stock) % 1000)
        
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        # 기본 추세 (주식별로 다름)
        base_trend = np.random.uniform(80, 120)
        trend = np.linspace(base_trend, base_trend * np.random.uniform(1.1, 1.3), 252)
        
        # 계절성
        seasonal = np.random.uniform(5, 15) * np.sin(2 * np.pi * np.arange(252) / 252 * np.random.uniform(2, 6))
        
        # 노이즈
        noise = np.random.normal(0, np.random.uniform(3, 8), 252)
        
        price = trend + seasonal + noise
        
        portfolio_data[stock] = pd.Series(price, index=dates, name=stock)
    
    return pd.DataFrame(portfolio_data)

# 포트폴리오 생성 및 분석
portfolio = create_portfolio_data()

print("=== 포트폴리오 데이터 ===")
print(portfolio.head())

# 수익률 계산
returns = portfolio.pct_change().dropna()

# 포트폴리오 통계
portfolio_stats = pd.DataFrame({
    '평균수익률': returns.mean() * 252,
    '변동성': returns.std() * np.sqrt(252),
    '샤프지수': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
    '최대낙폍': (portfolio / portfolio.cummax() - 1).min()
})

print("\n=== 포트폴리오 통계 ===")
print(portfolio_stats.round(4))

# 상관관계 분석
correlation_matrix = returns.corr()

print("\n=== 수익률 상관관계 ===")
print(correlation_matrix.round(3))
```

이 코드들을 실행하면서 시계열 데이터 분석의 다양한 기법들을 익힐 수 있습니다. 각 예제는 실제 비즈니스 상황에서 활용할 수 있는 실용적인 내용들로 구성되어 있습니다.
