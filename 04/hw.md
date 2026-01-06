# Homework - Pandas 심화

## 이 homework의 의미
Pandas의 고급 기능을 활용해 복잡한 데이터 분석 작업을 효율적으로 수행하는 능력을 키우는 과제야. 실무에서 마주치는 다양한 데이터 처리 시나리오를 해결하는 거지.

## 관련 정보 사이트
1. **Pandas 공식 문서**: https://pandas.pydata.org/docs/
2. **Kaggle Learn**: https://www.kaggle.com/learn/pandas

## 프로세스 진행 논리적 흐름
1. **데이터 필터링** → 조건에 맞는 데이터 추출
2. **그룹화 분석** → GroupBy로 집계 통계 계산
3. **피벗 테이블** → 데이터 재구성 및 요약
4. **데이터 머지** → 여러 DataFrame 결합
5. **성능 최적화** → 메모리 효율적인 처리

## 권장 코드
```python
import pandas as pd
import numpy as np

# 샘플 데이터
df = pd.DataFrame({
    '부서': ['영업', '개발', '마케팅', '개발', '영업'],
    '이름': ['김철수', '이영희', '박민준', '최지아', '정서연'],
    '급여': [4500, 6200, 5800, 5500, 4800]
})

# 그룹화 분석
부서별_평균 = df.groupby('부서')['급여'].agg(['mean', 'min', 'max'])
print(부서별_평균)

# 피벗 테이블
sales = pd.DataFrame({
    '제품': ['A', 'B', 'A', 'B'] * 3,
    '지역': ['서울', '부산'] * 6,
    '매출': np.random.randint(100, 1000, 12)
})

pivot = pd.pivot_table(sales, values='매출', 
                       index='제품', columns='지역', aggfunc='sum')
print(pivot)
```

## 관련 업계 회사
1. **Bloomberg** - 금융 데이터 분석에 Pandas 활용
2. **Airbnb** - 데이터 과학팀에서 Pandas 사용
3. **Uber** - 대규모 데이터 처리 및 분석
