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

print("=== Pandas 고급 기능 예제 ===")
print("\n원본 데이터:")
print(df)

# 그룹화 분석
print("\n=== 부서별 통계 ===")
부서별_통계 = df.groupby('부서').agg({
    '급여': ['mean', 'median', 'std'],
    '나이': ['mean', 'count']
})
print(부서별_통계)

# 피벗 테이블
print("\n=== 피벗 테이블 ===")
pivot_data = pd.pivot_table(df, values='급여', index='부서', aggfunc=['mean', 'count'])
print(pivot_data)

# 데이터 정렬 및 랭킹
print("\n=== 급여 순위 ===")
df['급여_순위'] = df['급여'].rank(method='dense', ascending=False)
print(df[['이름', '부서', '급여', '급여_순위']].sort_values('급여_순위'))
