# 2학년: 수리통계학 I 및 프로그래밍 및 실습 예제

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Matplotlib에서 한글 깨짐 방지를 위한 설정 (Windows: Malgun Gothic, macOS: AppleGothic)
# 본인 컴퓨터에 맞는 폰트로 변경 필요
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지


# --- 수리통계학 I ---
print("--- 1. 수리통계학 I ---")

# 1. 확률 분포 시각화 (정규분포)
# 평균(mu) 0, 표준편차(sigma) 1인 표준 정규분포 생성
mu, sigma = 0, 1
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
pdf = stats.norm.pdf(x, mu, sigma) # 확률 밀도 함수(PDF)

plt.figure(figsize=(8, 5))
plt.plot(x, pdf, label='표준 정규분포 PDF')
plt.title('표준 정규분포의 확률 밀도 함수')
plt.xlabel('값')
plt.ylabel('밀도')
plt.grid(True)
plt.legend()
plt.show()

# 2. 가설 검정 (t-검정)
# 'students_data.csv' 파일에서 통계학과(Statistics)와 컴퓨터공학과(Computer Science) 학생들의 성적 데이터 추출
data = pd.read_csv('students_data.csv')
stat_scores = data[data['major'] == 'Statistics']['score']
cs_scores = data[data['major'] == 'Computer Science']['score']

print("\n[가설 검정: t-검정]")
print("H0(귀무가설): 통계학과와 컴퓨터공학과 학생들의 평균 성적은 차이가 없다.")
print("H1(대립가설): 통계학과와 컴퓨터공학과 학생들의 평균 성적은 차이가 있다.")
print(f"\n통계학과 성적: {list(stat_scores)}")
print(f"컴퓨터공학과 성적: {list(cs_scores)}")

# 독립표본 t-검정 수행
# 두 집단의 분산이 같다고 가정 (equal_var=True)
t_statistic, p_value = stats.ttest_ind(stat_scores, cs_scores, equal_var=True)

print(f"\nt-통계량: {t_statistic:.3f}")
print(f"p-value: {p_value:.3f}")

# p-value를 기준으로 가설 검정 결과 해석
alpha = 0.05 # 유의수준 5%
if p_value < alpha:
    print(f"\n결과: p-value가 유의수준 {alpha}보다 작으므로 귀무가설을 기각합니다.")
    print("-> 두 학과 학생들의 평균 성적에는 통계적으로 유의미한 차이가 있습니다.")
else:
    print(f"\n결과: p-value가 유의수준 {alpha}보다 크므로 귀무가설을 기각할 수 없습니다.")
    print("-> 두 학과 학생들의 평균 성적에 통계적으로 유의미한 차이가 있다고 말할 수 없습니다.")


# --- 프로그래밍 및 실습 (Pandas) ---
print("\n\n--- 2. 프로그래밍 및 실습 (Pandas) ---")
# 'students_data.csv' 파일을 Pandas DataFrame으로 읽기
df = pd.read_csv('students_data.csv')

# 1. 데이터 기본 정보 탐색
print("\n[데이터 기본 정보]")
print("데이터 첫 5행:")
print(df.head())

print("\n데이터 요약 정보:")
df.info()

print("\n수치형 데이터 기술 통계:")
print(df.describe())

# 2. 데이터 조작
# (1) 특정 열 선택하기
print("\n[데이터 조작]")
print("'major'와 'score' 열만 선택:")
print(df[['major', 'score']])

# (2) 특정 조건으로 행 필터링하기
print("\n'score'가 90점 이상인 학생만 필터링:")
high_scores = df[df['score'] >= 90]
print(high_scores)

# (3) 전공별 평균 성적 계산하기
print("\n전공별 평균 성적:")
major_mean_scores = df.groupby('major')['score'].mean()
print(major_mean_scores)
