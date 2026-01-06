# Homework - 머신러닝

## 이 homework의 의미
데이터로부터 패턴을 학습하고 예측하는 머신러닝 모델을 구축하는 과제야. 지도학습, 비지도학습의 다양한 알고리즘을 실습하면서 AI의 핵심 원리를 이해하는 거지.

## 관련 정보 사이트
1. **Scikit-learn Documentation**: https://scikit-learn.org/stable/
2. **Machine Learning Mastery**: https://machinelearningmastery.com/

## 프로세스 진행 논리적 흐름
1. **데이터 준비** → 학습/테스트 데이터 분리
2. **특성 엔지니어링** → 중요한 특성 선택 및 변환
3. **모델 선택** → 회귀, 분류, 군집화 등
4. **학습 및 튜닝** → 하이퍼파라미터 최적화
5. **평가** → 정확도, F1-score, RMSE 등 측정

## 권장 코드
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np

# 샘플 데이터 (붓꽃 분류)
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
print("정확도:", model.score(X_test, y_test))
print("\n분류 리포트:")
print(classification_report(y_test, y_pred, 
                           target_names=iris.target_names))

# 특성 중요도
feature_importance = pd.DataFrame({
    '특성': iris.feature_names,
    '중요도': model.feature_importances_
}).sort_values('중요도', ascending=False)

print("\n특성 중요도:")
print(feature_importance)
```

## 관련 업계 회사
1. **Google (TensorFlow)** - 머신러닝 프레임워크 개발
2. **Meta (PyTorch)** - 딥러닝 라이브러리 제공
3. **DataRobot** - 자동화된 머신러닝 플랫폼
