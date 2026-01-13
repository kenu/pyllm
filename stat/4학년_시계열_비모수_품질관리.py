# 4학년: 시계열 분석, 비모수 통계학, 통계적 품질 관리 예제

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

# Matplotlib에서 한글 깨짐 방지를 위한 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


# --- 시계열 분석 ---
print("--- 1. 시계열 분석 ---")

# 'monthly_sales.csv' 파일을 시계열 데이터로 읽기
# 'date' 열을 인덱스로 사용하고, 날짜 형식으로 변환
ts_data = pd.read_csv('monthly_sales.csv', index_col='date', parse_dates=True)
ts_data = ts_data.asfreq('MS') # 월별 시작일(Month Start) 빈도로 데이터 설정

print("[월별 매출 시계열 데이터]")
print(ts_data.head())

# 시계열 분해 (Trend, Seasonality, Residual)
# 시계열 데이터를 추세, 계절성, 잔차(불규칙) 성분으로 분해하여 패턴을 파악
decomposition = seasonal_decompose(ts_data['sales'], model='additive') # 덧셈 모델 사용

fig = decomposition.plot()
fig.set_size_inches(10, 8)
fig.suptitle('시계열 분해 결과', y=1.02)
plt.show()

print("\n-> 시계열 분해를 통해 데이터의 장기적인 추세(Trend), 주기적인 계절성(Seasonal), \n   그리고 설명되지 않는 나머지(Residual)를 시각적으로 확인할 수 있습니다.")


# --- 비모수 통계학 ---
print("\n\n--- 2. 비모수 통계학 ---")

# 2학년 예제에서 사용한 두 학과 성적 데이터를 다시 사용
# 데이터가 정규분포를 따른다는 가정이 없을 때 사용하는 비모수 검정 방법
# 예: 만-휘트니 U 검정 (Mann-Whitney U test) - 독립표본 t-검정의 비모수 버전

data = pd.read_csv('students_data.csv')
stat_scores = data[data['major'] == 'Statistics']['score']
cs_scores = data[data['major'] == 'Computer Science']['score']

print("[가설 검정: 만-휘트니 U 검정]")
print("H0(귀무가설): 두 학과 학생들의 성적 분포는 동일하다.")
print("H1(대립가설): 두 학과 학생들의 성적 분포는 동일하지 않다.")

# 만-휘트니 U 검정 수행
u_statistic, p_value = stats.mannwhitneyu(stat_scores, cs_scores, alternative='two-sided')

print(f"\nU-통계량: {u_statistic:.3f}")
print(f"p-value: {p_value:.3f}")

alpha = 0.05 # 유의수준 5%
if p_value < alpha:
    print(f"\n결과: p-value가 유의수준 {alpha}보다 작으므로 귀무가설을 기각합니다.")
    print("-> 두 학과 학생들의 성적 분포에는 통계적으로 유의미한 차이가 있습니다.")
else:
    print(f"\n결과: p-value가 유의수준 {alpha}보다 크므로 귀무가설을 기각할 수 없습니다.")
    print("-> 두 학과 학생들의 성적 분포가 다르다고 말할 수 없습니다.")


# --- 통계적 품질 관리 (SQC) ---
print("\n\n--- 3. 통계적 품질 관리 (SQC) ---")
# 예: X-bar 관리도
# 공정의 평균이 안정된 상태로 관리되고 있는지 모니터링하는 도구

# 샘플 데이터 생성: 20개 그룹, 각 그룹에 5개의 샘플
np.random.seed(42) # 재현성을 위한 시드 설정
n_groups = 20
n_samples_per_group = 5
# 평균 10, 표준편차 0.5인 정규분포에서 데이터 생성
process_data = np.random.normal(loc=10.0, scale=0.5, size=(n_groups, n_samples_per_group))
# 15번째 그룹에 이상치 추가 (공정 이상 가정)
process_data[14] += 1.0

# 각 그룹의 평균 계산
group_means = np.mean(process_data, axis=1)

# 관리도의 중심선(CL), 상한선(UCL), 하한선(LCL) 계산
CL = np.mean(group_means)
# 전체 데이터의 표준편차를 사용하여 관리 한계선 추정
sigma_hat = np.std(np.ravel(process_data))
UCL = CL + 3 * (sigma_hat / np.sqrt(n_samples_per_group))
LCL = CL - 3 * (sigma_hat / np.sqrt(n_samples_per_group))

print("[X-bar 관리도]")
print(f"중심선 (CL): {CL:.3f}")
print(f"관리 상한선 (UCL): {UCL:.3f}")
print(f"관리 하한선 (LCL): {LCL:.3f}")

# X-bar 관리도 시각화
plt.figure(figsize=(12, 6))
plt.plot(range(1, n_groups + 1), group_means, marker='o', linestyle='-', label='그룹별 평균')
plt.axhline(CL, color='green', linestyle='--', label='중심선 (CL)')
plt.axhline(UCL, color='red', linestyle='--', label='관리 상한선 (UCL)')
plt.axhline(LCL, color='red', linestyle='--', label='관리 하한선 (LCL)')
plt.title('X-bar 관리도')
plt.xlabel('그룹 번호')
plt.ylabel('평균')
plt.xticks(range(1, n_groups + 1))
plt.legend()
plt.grid(True)
plt.show()

print("\n-> 15번 그룹의 평균이 관리 상한선을 벗어났습니다. 이는 공정에 이상이 발생했을 가능성을 시사합니다.")
