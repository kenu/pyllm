# 데이터 클리닝과 전처리

## 결측치 처리
실제 데이터에는 항상 결측치가 존재합니다. 효과적인 결측치 처리 방법을 학습합니다.

### 1. 결측치 탐지
```python
import pandas as pd
import numpy as np

# 결측치 포함 샘플 데이터 생성
data = {
    '이름': ['김철수', '이영희', '박민준', '최지아', '정서연', '강동훈', '윤미래'],
    '나이': [25, np.nan, 35, 28, 32, 45, np.nan],
    '성별': ['M', 'F', 'M', 'F', 'F', 'M', np.nan],
    '급여': [4500, 6200, np.nan, 5500, 4800, np.nan, 5200],
    '부서': ['영업', '개발', '마케팅', '개발', np.nan, '영업', '개발'],
    '입사일': ['2020-01-15', '2019-03-22', np.nan, '2021-02-01', '2020-06-15', '2018-09-10', '2022-01-05']
}
df = pd.DataFrame(data)

print("원본 데이터:")
print(df)

# 결측치 확인
print("\n결측치 개수:")
print(df.isnull().sum())

print("\n결측치 비율:")
print((df.isnull().sum() / len(df) * 100).round(2))

# 결측치 위치 시각화
print("\n결측치 위치 (True = 결측치):")
print(df.isnull())
```

### 2. 결측치 제거
```python
# 완전 제거 (행 기준)
df_drop_rows = df.dropna()
print("결측치 행 제거:")
print(df_drop_rows)

# 완전 제거 (열 기준)
df_drop_cols = df.dropna(axis=1)
print("\n결측치 열 제거:")
print(df_drop_cols)

# 특정 조건으로 제거
df_subset = df.dropna(subset=['이름', '급여'])  # 이름과 급여가 모두 있는 행만
print("\n이름과 급여가 모두 있는 데이터:")
print(df_subset)

# 임계치 기반 제거 (50% 이상 결측치인 열 제거)
df_thresh = df.dropna(thresh=len(df) * 0.5, axis=1)
print("\n50% 이상 데이터가 있는 열만:")
print(df_thresh.columns.tolist())
```

### 3. 결측치 대체
```python
# 단일 값 대체
df_filled = df.copy()
df_filled['나이'].fillna(df_filled['나이'].mean(), inplace=True)  # 평균으로 대체
df_filled['급여'].fillna(df_filled['급여'].median(), inplace=True)  # 중앙값으로 대체
df_filled['성별'].fillna('Unknown', inplace=True)  # 특정 값으로 대체

print("단일 값 대체:")
print(df_filled)

# 그룹별 대체
df_group_filled = df.copy()
나이_평균 = df_group_filled.groupby('성별')['나이'].transform('mean')
df_group_filled['나이'].fillna(나이_평균, inplace=True)

print("\n성별 그룹별 나이 평균으로 대체:")
print(df_group_filled[['이름', '성별', '나이']])

# 보간법 (시계열 데이터)
ts_data = pd.DataFrame({
    '날짜': pd.date_range('2023-01-01', periods=10),
    '가격': [100, 102, np.nan, 105, np.nan, 108, 107, np.nan, 110, 112]
}).set_index('날짜')

# 선형 보간
ts_data['선형보간'] = ts_data['가격'].interpolate(method='linear')

# 시간 보간
ts_data['시간보간'] = ts_data['가격'].interpolate(method='time')

# 스플라인 보간
ts_data['스플라인보간'] = ts_data['가격'].interpolate(method='spline', order=2)

print("\n시계열 보간법:")
print(ts_data)
```

## 이상치 탐지 및 처리
정상 범위를 벗어나는 데이터를 식별하고 처리합니다.

### 1. 통계적 이상치 탐지
```python
import matplotlib.pyplot as plt

# 이상치 포함 데이터 생성
np.random.seed(42)
normal_data = np.random.normal(100, 15, 100)
outliers = np.array([200, 250, 30, 20])  # 명백한 이상치
data_with_outliers = np.concatenate([normal_data, outliers])

outlier_df = pd.DataFrame({'값': data_with_outliers})

# 기술 통계
print("기술 통계:")
print(outlier_df.describe())

# Z-score 기반 이상치 탐지
mean = outlier_df['값'].mean()
std = outlier_df['값'].std()
outlier_df['Z-score'] = (outlier_df['값'] - mean) / std
outliers_z = outlier_df[np.abs(outlier_df['Z-score']) > 3]

print("\nZ-score 기반 이상치:")
print(outliers_z)

# IQR 기반 이상치 탐지
Q1 = outlier_df['값'].quantile(0.25)
Q3 = outlier_df['값'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_iqr = outlier_df[(outlier_df['값'] < lower_bound) | (outlier_df['값'] > upper_bound)]

print(f"\nIQR 기반 이상치 (경계: {lower_bound:.2f} ~ {upper_bound:.2f}):")
print(outliers_iqr)
```

### 2. 시각적 이상치 탐지
```python
# 박스플롯
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.boxplot(outlier_df['값'])
plt.title('박스플롯')
plt.ylabel('값')

# 히스토그램
plt.subplot(1, 3, 2)
plt.hist(outlier_df['값'], bins=30, alpha=0.7)
plt.title('히스토그램')
plt.xlabel('값')

# 산점도 (정규성 확인)
plt.subplot(1, 3, 3)
import scipy.stats as stats
stats.probplot(outlier_df['값'], dist="norm", plot=plt)
plt.title('Q-Q 플롯')

plt.tight_layout()
plt.show()
```

### 3. 이상치 처리
```python
# 이상치 제거
df_clean = outlier_df[~((outlier_df['값'] < lower_bound) | (outlier_df['값'] > upper_bound))].copy()

print("이상치 제거 후:")
print(f"원본 데이터: {len(outlier_df)}개")
print(f"정제된 데이터: {len(df_clean)}개")
print(f"제거된 이상치: {len(outlier_df) - len(df_clean)}개")

# 이상치 대체 (Winsorization)
df_winsorized = outlier_df.copy()
df_winsorized['Winsorized'] = df_winsorized['값'].clip(lower_bound, upper_bound)

print("\nWinsorization 결과:")
print(df_winsorized[['값', 'Winsorized']].describe())

# 로그 변환 (오른쪽으로 치우친 데이터)
skewed_data = np.random.lognormal(0, 1, 1000)
skewed_df = pd.DataFrame({'원본': skewed_data})
skewed_df['로그변환'] = np.log1p(skewed_df['원본'])

print("\n왜도 데이터 로그 변환:")
print(f"원본 왜도: {skewed_df['원본'].skew():.2f}")
print(f"로그변환 왜도: {skewed_df['로그변환'].skew():.2f}")
```

## 데이터 타입 변환
올바른 데이터 타입으로 변환하여 메모리 효율성과 분석 정확성을 높입니다.

### 1. 기본 타입 변환
```python
# 다양한 타입의 데이터
mixed_data = pd.DataFrame({
    '숫자문자': ['1', '2', '3', '4', '5'],
    '실수문자': ['1.1', '2.2', '3.3', '4.4', '5.5'],
    '날짜문자': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
    '불리언문자': ['True', 'False', 'true', 'false', 'TRUE'],
    '카테고리': ['A', 'B', 'A', 'C', 'B']
})

print("원본 타입:")
print(mixed_data.dtypes)

# 타입 변환
converted_data = mixed_data.copy()
converted_data['숫자문자'] = pd.to_numeric(converted_data['숫자문자'])
converted_data['실수문자'] = pd.to_numeric(converted_data['실수문자'])
converted_data['날짜문자'] = pd.to_datetime(converted_data['날짜문자'])
converted_data['불리언문자'] = converted_data['불리언문자'].str.lower().map({'true': True, 'false': False})
converted_data['카테고리'] = converted_data['카테고리'].astype('category')

print("\n변환 후 타입:")
print(converted_data.dtypes)
```

### 2. 메모리 최적화
```python
# 대용량 데이터 생성
large_data = pd.DataFrame({
    'ID': range(1000000),
    '가격': np.random.uniform(1000, 10000, 1000000),
    '수량': np.random.randint(1, 100, 1000000),
    '카테고리': np.random.choice(['A', 'B', 'C', 'D', 'E'], 1000000),
    '상태': np.random.choice([True, False], 1000000)
})

print("원본 메모리 사용량:")
print(large_data.memory_usage(deep=True).sum() / 1024**2, "MB")

# 메모리 최적화 함수
def optimize_memory(df):
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and col_type.name != 'category':
            if col_type != 'datetime64[ns]':
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
        
        elif col_type == object:
            if df[col].nunique() / len(df) < 0.5:  # 카디널리티가 낮으면
                df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f'메모리 사용량: {start_mem:.2f} MB -> {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% 감소)')
    
    return df

optimized_data = optimize_memory(large_data.copy())
print("\n최적화 후 타입:")
print(optimized_data.dtypes)
```

## 문자열 데이터 전처리
텍스트 데이터를 정제하고 표준화합니다.

### 1. 기본 문자열 처리
```python
# 문자열 데이터 생성
text_data = pd.DataFrame({
    '이름': ['김 철수', '이영희 ', ' 박민준', '최지아', '정서연'],
    '이메일': ['kim@email.com', 'invalid-email', 'park@domain.com', 'choi@site.com', 'jung@email.com'],
    '전화번호': ['010-1234-5678', '01012345678', '(010) 9876-5432', '010.1111.2222', '011-123-4567'],
    '주소': ['서울시 강남구', '부산광역시 해운대구', '대구시 수성구', '서울특별시 서초구', '경기도 성남시']
})

print("원본 문자열 데이터:")
print(text_data)

# 공백 처리
text_data['이름_정제'] = text_data['이름'].str.strip()  # 앞뒤 공백 제거
text_data['이름_정제'] = text_data['이름_정제'].str.replace(r'\s+', ' ', regex=True)  # 다중 공백 단일화

# 대소문자 표준화
text_data['이메일_소문자'] = text_data['이메일'].str.lower()

# 특수문자 제거
text_data['전화번호_정제'] = text_data['전화번호'].str.replace(r'[^\d]', '', regex=True)

print("\n문자열 정제 결과:")
print(text_data[['이름', '이름_정제', '이메일', '이메일_소문자', '전화번호', '전화번호_정제']])
```

### 2. 고급 문자열 처리
```python
# 이메일 유효성 검사
import re

def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

text_data['이메일_유효성'] = text_data['이메일'].apply(is_valid_email)

# 전화번호 형식 표준화
def standardize_phone(phone):
    digits = re.sub(r'[^\d]', '', phone)
    if len(digits) == 11:
        return f'{digits[:3]}-{digits[3:7]}-{digits[7:]}'
    elif len(digits) == 10:
        return f'{digits[:3]}-{digits[3:6]}-{digits[6:]}'
    else:
        return phone

text_data['전화번호_표준'] = text_data['전화번호'].apply(standardize_phone)

# 주소 표준화
def standardize_address(address):
    # 시/도 표준화
    address = re.sub(r'서울특별시|서울시', '서울', address)
    address = re.sub(r'부산광역시', '부산', address)
    address = re.sub(r'대구시', '대구', address)
    address = re.sub(r'경기도', '경기', address)
    return address

text_data['주소_표준'] = text_data['주소'].apply(standardize_address)

print("\n고급 문자열 처리 결과:")
print(text_data[['이메일', '이메일_유효성', '전화번호_표준', '주소_표준']])
```

## 데이터 정규화와 표준화
머신러닝을 위해 데이터를 적절한 스케일로 변환합니다.

### 1. 수치형 데이터 스케일링
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import seaborn as sns

# 다양한 분포의 데이터
np.random.seed(42)
scaling_data = pd.DataFrame({
    '정규분포': np.random.normal(100, 15, 1000),
    '균등분포': np.random.uniform(0, 200, 1000),
    '지수분포': np.random.exponential(50, 1000),
    '카이제곱': np.random.chisquare(10, 1000)
})

print("원본 데이터 통계:")
print(scaling_data.describe())

# StandardScaler (Z-score 표준화)
scaler_standard = StandardScaler()
scaled_standard = pd.DataFrame(
    scaler_standard.fit_transform(scaling_data),
    columns=[col + '_standard' for col in scaling_data.columns]
)

# MinMaxScaler (0-1 정규화)
scaler_minmax = MinMaxScaler()
scaled_minmax = pd.DataFrame(
    scaler_minmax.fit_transform(scaling_data),
    columns=[col + '_minmax' for col in scaling_data.columns]
)

# RobustScaler (이상치에 강건)
scaler_robust = RobustScaler()
scaled_robust = pd.DataFrame(
    scaler_robust.fit_transform(scaling_data),
    columns=[col + '_robust' for col in scaling_data.columns]
)

print("\nStandardScaler 결과:")
print(scaled_standard.describe().round(2))

print("\nMinMaxScaler 결과:")
print(scaled_minmax.describe().round(2))
```

### 2. 시각적 비교
```python
# 스케일링 전후 분포 비교
plt.figure(figsize=(15, 10))

for i, col in enumerate(scaling_data.columns):
    # 원본
    plt.subplot(4, 3, i*3 + 1)
    plt.hist(scaling_data[col], bins=30, alpha=0.7)
    plt.title(f'{col} (원본)')
    
    # StandardScaler
    plt.subplot(4, 3, i*3 + 2)
    plt.hist(scaled_standard[col + '_standard'], bins=30, alpha=0.7)
    plt.title(f'{col} (Standard)')
    
    # MinMaxScaler
    plt.subplot(4, 3, i*3 + 3)
    plt.hist(scaled_minmax[col + '_minmax'], bins=30, alpha=0.7)
    plt.title(f'{col} (MinMax)')

plt.tight_layout()
plt.show()
```

## 데이터 인코딩
범주형 데이터를 수치형으로 변환합니다.

### 1. 레이블 인코딩
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 범주형 데이터
categorical_data = pd.DataFrame({
    '성별': ['남', '여', '남', '여', '남', '여', '남'],
    '혈액형': ['A', 'B', 'O', 'A', 'AB', 'B', 'O'],
    '지역': ['서울', '부산', '대구', '서울', '부산', '대구', '서울'],
    '학력': ['고졸', '대졸', '대원', '고졸', '석박사', '대졸', '대원']
})

print("원본 범주형 데이터:")
print(categorical_data)

# Label Encoding
label_encoded = categorical_data.copy()
le_dict = {}

for col in categorical_data.columns:
    le = LabelEncoder()
    label_encoded[col + '_label'] = le.fit_transform(categorical_data[col])
    le_dict[col] = le

print("\nLabel Encoding 결과:")
print(label_encoded[[col + '_label' for col in categorical_data.columns]])

# 인코딩 매핑 확인
print("\n인코딩 매핑:")
for col, le in le_dict.items():
    print(f"{col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
```

### 2. 원-핫 인코딩
```python
# One-Hot Encoding
one_hot_encoded = pd.get_dummies(categorical_data, prefix=categorical_data.columns)

print("\nOne-Hot Encoding 결과:")
print(one_hot_encoded.head())

# 다중 공선성 문제 해결 (drop_first=True)
one_hot_encoded_drop = pd.get_dummies(categorical_data, prefix=categorical_data.columns, drop_first=True)

print("\nOne-Hot Encoding (첫번째 카테고리 제거):")
print(one_hot_encoded_drop.head())
```

## 데이터 전처리 파이프라인
전체 전처리 과정을 자동화합니다.

### 1. 전처리 함수
```python
def comprehensive_preprocessing(df):
    """종합 데이터 전처리 파이프라인"""
    
    processed_df = df.copy()
    
    print("=== 데이터 전처리 시작 ===")
    print(f"원본 데이터 크기: {processed_df.shape}")
    
    # 1. 결측치 처리
    print("\n1. 결측치 처리...")
    missing_info = processed_df.isnull().sum()
    print(f"결측치: {missing_info[missing_info > 0].to_dict()}")
    
    # 수치형 열: 중앙값으로 대체
    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if processed_df[col].isnull().sum() > 0:
            processed_df[col].fillna(processed_df[col].median(), inplace=True)
    
    # 범주형 열: 최빈값으로 대체
    categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if processed_df[col].isnull().sum() > 0:
            mode_value = processed_df[col].mode()[0] if not processed_df[col].mode().empty else 'Unknown'
            processed_df[col].fillna(mode_value, inplace=True)
    
    # 2. 이상치 처리
    print("\n2. 이상치 처리...")
    for col in numeric_cols:
        Q1 = processed_df[col].quantile(0.25)
        Q3 = processed_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((processed_df[col] < lower_bound) | (processed_df[col] > upper_bound)).sum()
        if outliers > 0:
            processed_df[col] = processed_df[col].clip(lower_bound, upper_bound)
            print(f"{col}: {outliers}개 이상치 처리")
    
    # 3. 데이터 타입 최적화
    print("\n3. 데이터 타입 최적화...")
    processed_df = optimize_memory(processed_df)
    
    # 4. 범주형 데이터 인코딩
    print("\n4. 범주형 데이터 인코딩...")
    for col in categorical_cols:
        if processed_df[col].dtype == 'object':
            # 카디널리티가 낮으면 원-핫 인코딩, 높으면 레이블 인코딩
            if processed_df[col].nunique() <= 10:
                dummies = pd.get_dummies(processed_df[col], prefix=col)
                processed_df = pd.concat([processed_df.drop(col, axis=1), dummies], axis=1)
            else:
                le = LabelEncoder()
                processed_df[col + '_encoded'] = le.fit_transform(processed_df[col])
                processed_df.drop(col, axis=1, inplace=True)
    
    print(f"\n전처리 완료: {processed_df.shape}")
    return processed_df

# 전처리 파이프라인 테스트
test_data = pd.DataFrame({
    '수치1': [1, 2, 3, 4, 5, 100, np.nan],  # 이상치와 결측치 포함
    '수치2': [10, 20, 30, 40, 50, 60, np.nan],
    '범주1': ['A', 'B', 'A', 'C', 'B', 'A', np.nan],
    '범주2': ['X', 'Y', 'Z', 'X', 'Y', 'Z', np.nan]
})

processed_result = comprehensive_preprocessing(test_data)
print("\n전처리 결과:")
print(processed_result.head())
```

### 2. 실전 예제: 고객 데이터 전처리
```python
# 실제 비즈니스 시나리오
def create_customer_data():
    """고객 데이터 생성"""
    np.random.seed(42)
    n_customers = 1000
    
    data = {
        '고객ID': range(1, n_customers + 1),
        '나이': np.random.randint(18, 80, n_customers),
        '성별': np.random.choice(['남', '여', np.nan], n_customers, p=[0.48, 0.48, 0.04]),
        '월수입': np.random.lognormal(10, 0.5, n_customers),
        '가입일자': pd.date_range('2020-01-01', periods=n_customers, freq='D')[:n_customers],
        '지역': np.random.choice(['서울', '부산', '대구', '대전', '광주'], n_customers),
        '구매횟수': np.random.poisson(5, n_customers),
        '총구매액': np.random.exponential(100000, n_customers),
        '회원등급': np.random.choice(['브론즈', '실버', '골드', '플래티넘'], n_customers),
        '이메일수신동의': np.random.choice([True, False, np.nan], n_customers)
    }
    
    df = pd.DataFrame(data)
    
    # 일부 데이터에 결측치와 이상치 추가
    df.loc[np.random.choice(df.index, 50), '나이'] = np.nan
    df.loc[np.random.choice(df.index, 30), '월수입'] = np.random.uniform(10000000, 50000000, 30)  # 이상치
    df.loc[np.random.choice(df.index, 20), '총구매액'] = np.nan
    
    return df

# 고객 데이터 전처리
customer_data = create_customer_data()
print("=== 고객 데이터 전처리 ===")
print(f"원본 데이터: {customer_data.shape}")

# 전처리 적용
processed_customer = comprehensive_preprocessing(customer_data)

# 전처리 후 데이터 분석
print("\n전처리 후 데이터 통계:")
print(processed_customer.describe().round(2))

print("\n전처리 후 데이터 타입:")
print(processed_customer.dtypes)
```

## 데이터 품질 평가
전처리된 데이터의 품질을 평가합니다.

```python
def data_quality_assessment(df):
    """데이터 품질 평가"""
    
    quality_report = {}
    
    # 1. 완전성 (Completeness)
    completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    quality_report['완전성'] = completeness
    
    # 2. 유일성 (Uniqueness)
    uniqueness = {}
    for col in df.columns:
        unique_ratio = df[col].nunique() / len(df) * 100
        uniqueness[col] = unique_ratio
    quality_report['유일성'] = uniqueness
    
    # 3. 일관성 (Consistency)
    consistency = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # 음수가 있어서는 안 되는 열 체크
        if '금액' in col or '수입' in col or '가격' in col:
            negative_ratio = (df[col] < 0).sum() / len(df) * 100
            consistency[col] = 100 - negative_ratio
    quality_report['일관성'] = consistency
    
    # 4. 정확성 (Accuracy) - 도메인 지식 기반
    accuracy = {}
    if '나이' in df.columns:
        valid_age = ((df['나이'] >= 0) & (df['나이'] <= 120)).sum() / len(df) * 100
        accuracy['나이'] = valid_age
    
    quality_report['정확성'] = accuracy
    
    return quality_report

# 데이터 품질 평가 실행
quality_result = data_quality_assessment(processed_customer)

print("\n=== 데이터 품질 평가 보고서 ===")
for 항목, 값 in quality_result.items():
    print(f"\n{항목}:")
    if isinstance(값, dict):
        for key, val in 값.items():
            print(f"  {key}: {val:.2f}%")
    else:
        print(f"  전체: {값:.2f}%")
```

이 전처리 과정들을 통해 실제 데이터 분석 프로젝트에서 마주할 수 있는 다양한 데이터 품질 문제들을 체계적으로 해결할 수 있습니다.
