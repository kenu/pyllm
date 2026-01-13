# 특화 과목: 데이터 마이닝, 생존 분석, 베이지안 통계학 예제

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from lifelines import KaplanMeierFitter

# Matplotlib에서 한글 깨짐 방지를 위한 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


# --- 데이터 마이닝 ---
print("--- 1. 데이터 마이닝 (분류 모델) ---")
# 예: 의사결정나무(Decision Tree)를 이용한 붓꽃(Iris) 품종 분류

# 1. 데이터 준비 (scikit-learn 내장 데이터셋 사용)
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 2. 데이터 분할 (학습용 / 테스트용)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 모델 학습
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 4. 모델 시각화
plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=feature_names, class_names=target_names, filled=True)
plt.title("의사결정나무 모델 시각화")
plt.show()

# 5. 예측 및 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n[모델 평가]")
print(f"정확도: {accuracy:.3f}")
print("\n분류 리포트:")
print(classification_report(y_test, y_pred, target_names=target_names))


# --- 생존 분석 ---
print("\n\n--- 2. 생존 분석 ---")
# 예: 카플란-마이어(Kaplan-Meier) 생존 곡선
# 특정 이벤트(사망, 고장, 재발 등)가 발생할 때까지의 시간을 분석하는 기법

# 1. 샘플 데이터 생성
# T: 이벤트가 발생하기까지 걸린 시간 (또는 관찰이 끝난 시간)
# E: 이벤트 발생 여부 (1: 발생, 0: 중도절단/관찰종료)
T = [10, 15, 20, 25, 30, 30, 40, 45, 50, 50, 60]
E = [1,  0,  1,  1,  0,  1,  1,  0,  1,  1,  0]

# 2. 카플란-마이어 모델 생성 및 학습
kmf = KaplanMeierFitter()
kmf.fit(durations=T, event_observed=E)

# 3. 생존 곡선 시각화
plt.figure(figsize=(8, 6))
kmf.plot_survival_function()
plt.title('카플란-마이어 생존 곡선')
plt.xlabel('시간 (일)')
plt.ylabel('생존 확률')
plt.grid(True)
plt.show()

print("\n-> 시간에 따라 생존 확률이 어떻게 변하는지를 보여줍니다.")
print("   계단식으로 감소하며, x축의 점은 이벤트(사망 등)가 발생한 시점을 의미합니다.")


# --- 베이지안 통계학 ---
print("\n\n--- 3. 베이지안 통계학 ---")
# 예: 동전 던지기 예제를 통한 믿음의 업데이트 과정
# 사전 확률(Prior) -> (데이터 관찰) -> 사후 확률(Posterior)

# 1. 사전 확률(Prior) 설정
# 동전의 앞면이 나올 확률(theta)에 대해 아는 바가 없다고 가정
# -> 0과 1 사이의 모든 가능성이 동일하다고 설정 (Uniform Distribution)
p_theta = np.linspace(0, 1, 101) # 0, 0.01, ..., 1.0 까지의 후보 theta 값
prior = np.repeat(1/101, 101)    # 모든 theta 값에 동일한 확률 할당

# 2. 데이터 관찰 (Likelihood)
# 10번 던져서 8번 앞면이 나왔다고 가정 (n=10, k=8)
n = 10
k = 8
# 각 theta 값에 대해 이 데이터가 관찰될 가능도(Likelihood)를 계산 (이항분포 사용)
likelihood = stats.binom.pmf(k, n, p_theta)

# 3. 사후 확률(Posterior) 계산
# Posterior ∝ Likelihood * Prior
unnormalized_posterior = likelihood * prior
# 정규화 (모든 확률의 합이 1이 되도록)
posterior = unnormalized_posterior / np.sum(unnormalized_posterior)

# 4. 결과 시각화
plt.figure(figsize=(10, 6))
plt.plot(p_theta, prior, label='사전 확률 (Prior)')
plt.plot(p_theta, posterior, label='사후 확률 (Posterior)', color='orange')
plt.axvline(p_theta[np.argmax(posterior)], color='red', linestyle='--', label=f'사후 확률 최대값 (MAP): {p_theta[np.argmax(posterior)]:.2f}')
plt.title('베이즈 업데이트: 동전 앞면 확률 추정')
plt.xlabel('앞면이 나올 확률 (theta)')
plt.ylabel('확률 밀도')
plt.legend()
plt.grid(True)
plt.show()

print("\n-> '10번 중 8번 앞면'이라는 데이터를 관찰한 후,")
print("   우리의 믿음(확률 분포)은 0.5 근처에 있던 사전 확률에서 0.8 근처로 업데이트되었습니다.")
