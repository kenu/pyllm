# 1학년: 기초 통계학 및 확률론 예제

import random
from collections import Counter

# --- 기초 통계학 및 실습 ---

# 예제 데이터: 한 반의 학생 15명의 수학 점수
scores = [85, 92, 78, 88, 95, 81, 76, 89, 90, 85, 83, 79, 91, 87, 80]

print("--- 1. 기초 통계학 ---")
print(f"데이터: {scores}\n")

# 1. 중심 경향성 계산
# (1) 평균 (Mean)
mean = sum(scores) / len(scores)
print(f"평균: {mean:.2f}")

# (2) 중앙값 (Median)
sorted_scores = sorted(scores)
n = len(sorted_scores)
if n % 2 == 0:
    mid1 = sorted_scores[n // 2 - 1]
    mid2 = sorted_scores[n // 2]
    median = (mid1 + mid2) / 2
else:
    median = sorted_scores[n // 2]
print(f"중앙값: {median}")

# (3) 최빈값 (Mode)
# collections.Counter를 사용하여 각 항목의 빈도를 계산
count = Counter(scores)
# 가장 빈도가 높은 항목들을 찾음
max_frequency = max(count.values())
modes = [score for score, freq in count.items() if freq == max_frequency]
print(f"최빈값: {modes}")


# 2. 산포도 계산
# (1) 분산 (Variance)
variance = sum((x - mean) ** 2 for x in scores) / len(scores)
print(f"분산: {variance:.2f}")

# (2) 표준편차 (Standard Deviation)
std_deviation = variance ** 0.5
print(f"표준편차: {std_deviation:.2f}\n")


# --- 확률론 ---

print("--- 2. 확률론 ---")

# 예제: 주사위를 1000번 던지는 시뮬레이션
num_trials = 1000
results = []

for _ in range(num_trials):
    roll = random.randint(1, 6) # 1부터 6까지의 정수 중 하나를 랜덤하게 선택
    results.append(roll)

# 각 눈이 나온 횟수 계산
roll_counts = Counter(results)

print(f"주사위를 {num_trials}번 던지는 시뮬레이션 결과:")

# 각 눈이 나올 확률 계산 및 출력
for i in range(1, 7):
    frequency = roll_counts.get(i, 0)
    probability = frequency / num_trials
    print(f"눈 {i}이(가) 나올 확률: {probability:.3f} (실제 나온 횟수: {frequency}번)")

# 이론적 확률 (1/6)
theoretical_prob = 1/6
print(f"\n이론적 확률 (1/6): {theoretical_prob:.3f}")
print("-> 시뮬레이션 횟수가 늘어날수록 각 눈이 나올 확률은 이론적 확률에 가까워집니다.")
