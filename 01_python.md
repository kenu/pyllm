# 파이썬: 데이터 분석 및 머신러닝을 위한 기본 언어

## 파이썬 기초 문법
변수, 데이터 타입, 제어문, 함수 등 파이썬의 기본 문법과 구조를 이해합니다.

**예제:**
```python
# 변수와 데이터 타입
name = "Alice"  # 문자열(str)
age = 30       # 정수(int)
height = 165.5 # 실수(float)
is_student = True # 불리언(bool)

# 리스트(List)
fruits = ["apple", "banana", "cherry"]

# 딕셔너리(Dictionary)
person = {"name": "Bob", "age": 25}

# 제어문 (if-else)
if age >= 19:
    print("성인입니다.")
else:
    print("미성년자입니다.")

# 제어문 (for)
for fruit in fruits:
    print(fruit)

# 함수
def add(a, b):
    return a + b

result = add(5, 3)
print(f"5 + 3 = {result}")
```

## 데이터 분석 라이브러리
Pandas, NumPy, Matplotlib 등 데이터 분석 및 시각화를 위한 주요 라이브러리의 사용법을 배웁니다.

**예제:**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# NumPy: 숫자 데이터를 다루는 배열 생성
numpy_array = np.array([1, 2, 3, 4, 5])
print("NumPy Array:", numpy_array)

# Pandas: 표 형식의 데이터(DataFrame) 생성
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Paris', 'London']}
df = pd.DataFrame(data)

print("Pandas DataFrame:")
print(df)

# 특정 조건에 맞는 데이터 선택 (나이가 30 이상인 사람)
print("\nAge >= 30:")
print(df[df['Age'] >= 30])

# Matplotlib: 데이터 시각화
plt.figure(figsize=(8, 5))
plt.bar(df['Name'], df['Age'], color=['red', 'green', 'blue'])
plt.title('Age of Users')
plt.xlabel('Name')
plt.ylabel('Age')
plt.show()
```

## 머신러닝 기초
Scikit-learn을 활용한 머신러닝 모델의 구축 및 평가 방법을 학습합니다.

**예제:**
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# 예제 데이터: 공부한 시간(X)과 시험 점수(y)
X = np.array([2, 4, 6, 8, 10]).reshape(-1, 1)
y = np.array([60, 70, 80, 85, 95])

# 데이터를 학습용과 테스트용으로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 선형 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 7시간 공부했을 때의 점수 예측
predicted_score = model.predict([[7]])
print(f"7시간 공부했을 때의 예상 점수: {predicted_score[0]:.2f}")

# 모델 성능 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"모델의 평균 제곱 오차(MSE): {mse:.2f}")

```
