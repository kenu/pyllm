# 데이터 분석 기초 (Pandas 심화)

## DataFrame 고급 操作
Pandas DataFrame의 고급 기능들을 마스터하여 복잡한 데이터 분석 작업을 수행합니다.

### 1. 데이터 필터링과 선택
```python
import pandas as pd
import numpy as np

# 샘플 데이터 생성
data = {
    '이름': ['김철수', '이영희', '박민준', '최지아', '정서연'],
    '나이': [25, 30, 35, 28, 32],
    '부서': ['영업', '개발', '마케팅', '개발', '영업'],
    '급여': [4500, 6200, 5800, 5500, 4800],
    '입사일': pd.to_datetime(['2020-01-15', '2019-03-22', '2018-07-10', '2021-02-01', '2020-06-15'])
}
df = pd.DataFrame(data)

# 조건부 필터링
개발자 = df[df['부서'] == '개발']
고연봉 = df[df['급여'] >= 5000]

# 복합 조건
개발자_고연봉 = df[(df['부서'] == '개발') & (df['급여'] >= 5000)]

# 특정 열만 선택
필요정보 = df[['이름', '부서', '급여']]

print("개발자 정보:")
print(개발자)
print("\n고연봉자 정보:")
print(고연봉)
```

### 2. 데이터 정렬과 랭킹
```python
# 급여 순으로 정렬
급여순 = df.sort_values('급여', ascending=False)

# 다중 조건 정렬
다중정렬 = df.sort_values(['부서', '급여'], ascending=[True, False])

# 랭킹 매기기
df['급여_순위'] = df['급여'].rank(method='dense', ascending=False)
df['나이_순위'] = df['나이'].rank()

print("\n급여 순위:")
print(df[['이름', '급여', '급여_순위']].sort_values('급여_순위'))
```

## 그룹화 (GroupBy)
데이터를 그룹으로 묶어 통계 분석을 수행합니다.

### 1. 기본 그룹화
```python
# 부서별 그룹화
부서별 = df.groupby('부서')

# 부서별 평균 급여
부서별_평균급여 = 부서별['급여'].mean()

# 부서별 인원 수
부서별_인원 = 부서별.size()

# 여러 통계량 한 번에
부서별_통계 = 부서별['급여'].agg(['mean', 'median', 'std', 'min', 'max'])

print("부서별 평균 급여:")
print(부서별_평균급여)
print("\n부서별 통계:")
print(부서별_통계)
```

### 2. 고급 그룹화
```python
# 여러 열로 그룹화
df['근속연수'] = (pd.Timestamp.now() - df['입사일']).dt.days / 365.25
그룹화_복합 = df.groupby(['부서', df['근속연수'].astype(int)])

# 사용자 정의 함수 적용
def 급여_등급(급여):
    if 급여 >= 6000:
        return '고급'
    elif 급여 >= 5000:
        return '중급'
    else:
        return '초급'

df['급여등급'] = df['급여'].apply(급여_등급)

# 그룹별 변환
부서별_급여_zscore = df.groupby('부서')['급여'].transform(lambda x: (x - x.mean()) / x.std())

print("\n부서별 급여 Z-score:")
print(df[['이름', '부서', '급여']].assign(Z_score=부서별_급여_zscore))
```

## 피벗 테이블 (Pivot Table)
데이터를 재구성하여 요약 정보를 보기 좋게 표현합니다.

### 1. 기본 피벗 테이블
```python
# 더 많은 샘플 데이터
extended_data = {
    '날짜': pd.date_range('2023-01-01', periods=20, freq='D'),
    '제품': ['노트북', '마우스', '키보드', '모니터'] * 5,
    '지역': ['서울', '부산', '대구', '대전'] * 5,
    '판매량': np.random.randint(10, 100, 20),
    '매출': np.random.randint(1000, 10000, 20)
}
sales_df = pd.DataFrame(extended_data)

# 제품별 지역 평균 매출
피벗1 = pd.pivot_table(sales_df, 
                       values='매출', 
                       index='제품', 
                       columns='지역', 
                       aggfunc='mean')

# 다중 인덱스 피벗
피벗2 = pd.pivot_table(sales_df, 
                       values=['판매량', '매출'], 
                       index=['제품', '지역'], 
                       aggfunc={'판매량': 'sum', '매출': 'mean'})

print("제품별 지역 평균 매출:")
print(피벗1)
print("\n다중 인덱스 피벗:")
print(피벗2)
```

### 2. 고급 피벗 테이블
```python
# 마진 계산 포함
피벗_마진 = pd.pivot_table(sales_df, 
                           values='매출', 
                           index='제품', 
                           columns='지역', 
                           aggfunc='sum', 
                           margins=True, 
                           margins_name='총계')

# 퍼센트 계산
피벗_퍼센트 = pd.pivot_table(sales_df, 
                            values='매출', 
                            index='제품', 
                            columns='지역', 
                            aggfunc='sum', 
                            fill_value=0)

# 전체 대비 퍼센트
피벗_퍼센트 = 피벗_퍼센트.div(피벗_퍼센트.sum(axis=1), axis=0) * 100

print("\n마진 포함 피벗:")
print(피벗_마진)
print("\n퍼센트 피벗:")
print(피벗_퍼센트.round(2))
```

## 데이터 머지 (Merge/Join)
여러 DataFrame을 합쳐서 분석합니다.

### 1. 기본 머지
```python
# 직원 정보
직원정보 = pd.DataFrame({
    '직원ID': [1, 2, 3, 4, 5],
    '이름': ['김철수', '이영희', '박민준', '최지아', '정서연'],
    '부서': ['영업', '개발', '마케팅', '개발', '영업']
})

# 성과 정보
성과정보 = pd.DataFrame({
    '직원ID': [1, 2, 3, 4, 6, 7],  # 6,7은 직원정보에 없음
    '평점': [85, 92, 78, 88, 95, 82],
    '보너스': [500, 800, 300, 600, 900, 400]
})

# inner join (공통된 것만)
inner_join = pd.merge(직원정보, 성과정보, on='직원ID', how='inner')

# left join (왼쪽 기준 전부)
left_join = pd.merge(직원정보, 성과정보, on='직원ID', how='left')

# outer join (전부)
outer_join = pd.merge(직원정보, 성과정보, on='직원ID', how='outer')

print("Inner Join:")
print(inner_join)
print("\nLeft Join:")
print(left_join)
```

### 2. 고급 머지
```python
# 여러 키로 머지
부서정보 = pd.DataFrame({
    '부서': ['영업', '개발', '마케팅', '인사'],
    '관리자': ['김부장', '이부장', '박부장', '최부장'],
    '예산': [1000000, 1500000, 800000, 600000]
})

# 여러 단계 머지
단계1 = pd.merge(직원정보, 성과정보, on='직원ID', how='left')
전체정보 = pd.merge(단계1, 부서정보, on='부서', how='left')

# 인덱스로 머지
직원정보_idx = 직원정보.set_index('직원ID')
성과정보_idx = 성과정보.set_index('직원ID')

인덱스_머지 = pd.merge(직원정보_idx, 성과정보_idx, 
                      left_index=True, right_index=True, how='outer')

print("\n전체 정보:")
print(전체정보)
```

## Concat과 Append
DataFrame을 수직/수평으로 연결합니다.

```python
# 수직 연결 (concat)
df1 = sales_df.iloc[:10]
df2 = sales_df.iloc[10:]

수직연결 = pd.concat([df1, df2], ignore_index=True)

# 수평 연결
df3 = pd.DataFrame({
    '직원ID': [1, 2, 3, 4, 5],
    '이름': ['김철수', '이영희', '박민준', '최지아', '정서연']
})

df4 = pd.DataFrame({
    '직원ID': [1, 2, 3, 4, 5],
    '연봉': [4500, 6200, 5800, 5500, 4800]
})

수평연결 = pd.concat([df3.set_index('직원ID'), df4.set_index('직원ID')], 
                    axis=1, join='inner')

print("\n수평 연결 결과:")
print(수평연결)
```

## 실전 예제: 판매 데이터 분석
```python
# 실제 비즈니스 시나리오
판매데이터 = pd.DataFrame({
    '날짜': pd.date_range('2023-01-01', periods=100, freq='D'),
    '제품카테고리': np.random.choice(['전자제품', '의류', '식품', '가구'], 100),
    '고객등급': np.random.choice(['VIP', '일반', '신규'], 100),
    '판매량': np.random.randint(1, 50, 100),
    '단가': np.random.randint(1000, 50000, 100)
})

# 총매출 계산
판매데이터['총매출'] = 판매데이터['판매량'] * 판매데이터['단가']

# 월별 추출
판매데이터['월'] = 판매데이터['날짜'].dt.month

# 분석 보고서 생성
보고서 = pd.pivot_table(판매데이터, 
                        values='총매출', 
                        index=['월', '제품카테고리'], 
                        columns='고객등급', 
                        aggfunc='sum', 
                        fill_value=0)

# 월별 총매출 순위
월별_순위 = 판매데이터.groupby('월')['총매출'].sum().rank(ascending=False)

print("\n월별/제품별/고객등급별 매출 보고서:")
print(보고서)
print("\n월별 매출 순위:")
print(월별_순위)
```

## 성능 최적화 팁
```python
# 메모리 사용량 최적화
def 최적화_데이터타입(df):
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # 카디널리티가 낮으면
            df[col] = df[col].astype('category')
    
    return df

# 대용량 데이터 처리
chunk_size = 10000
results = []

for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    processed_chunk = chunk.groupby('category')['value'].sum()
    results.append(processed_chunk)

final_result = pd.concat(results).groupby(level=0).sum()

print("데이터 처리 완료!")
```
