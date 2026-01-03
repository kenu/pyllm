# 머신러닝 입문 (Scikit-learn)

## 머신러닝 기초 개념
머신러닝의 기본 개념과 Scikit-learn 라이브러리의 사용법을 익힙니다.

### 1. 머신러닝의 종류
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, make_blobs
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 지도학습 예제 데이터 생성
X_class, y_class = make_classification(
    n_samples=1000, n_features=2, n_redundant=0,
    n_informative=2, n_clusters_per_class=1, random_state=42
)

# 비지도학습 예제 데이터 생성
X_unsup, _ = make_blobs(n_samples=1000, centers=3, random_state=42)

# 회귀 예제 데이터 생성
X_reg, y_reg = make_regression(n_samples=1000, n_features=1, noise=20, random_state=42)

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 지도학습 (분류)
axes[0].scatter(X_class[:, 0], X_class[:, 1], c=y_class, cmap='viridis', alpha=0.6)
axes[0].set_title('지도학습 (분류)')
axes[0].set_xlabel('특성 1')
axes[0].set_ylabel('특성 2')

# 비지도학습 (클러스터링)
axes[1].scatter(X_unsup[:, 0], X_unsup[:, 1], alpha=0.6)
axes[1].set_title('비지도학습 (클러스터링)')
axes[1].set_xlabel('특성 1')
axes[1].set_ylabel('특성 2')

# 회귀
axes[2].scatter(X_reg, y_reg, alpha=0.6)
axes[2].set_title('지도학습 (회귀)')
axes[2].set_xlabel('특성')
axes[2].set_ylabel('타겟')

plt.tight_layout()
plt.show()

print("머신러닝의 종류:")
print("1. 지도학습: 정답이 있는 데이터로 학습 (분류, 회귀)")
print("2. 비지도학습: 정답이 없는 데이터로 학습 (클러스터링, 차원축소)")
print("3. 강화학습: 보상을 통해 학습 (게임, 로봇 제어)")
```

## 지도학습: 분류
정답 레이블을 사용하여 데이터를 분류하는 모델을 만듭니다.

### 1. 데이터 준비와 전처리
```python
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Iris 데이터셋 로드
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# 데이터프레임으로 변환
iris_df = pd.DataFrame(X_iris, columns=iris.feature_names)
iris_df['target'] = y_iris
iris_df['target_name'] = iris_df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("Iris 데이터셋 정보:")
print(f"특성 개수: {X_iris.shape[1]}")
print(f"샘플 개수: {X_iris.shape[0]}")
print(f"클래스: {iris.target_names}")
print("\n데이터 샘플:")
print(iris_df.head())

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42, stratify=y_iris
)

print(f"\n훈련 데이터: {X_train.shape}")
print(f"테스트 데이터: {X_test.shape}")

# 특성 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n특성 스케일링 완료")
```

### 2. 다양한 분류 알고리즘
```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# 분류 모델 정의
models = {
    '로지스틱 회귀': LogisticRegression(random_state=42),
    '결정 트리': DecisionTreeClassifier(random_state=42),
    '랜덤 포레스트': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'K-최근접 이웃': KNeighborsClassifier(),
    '나이브 베이즈': GaussianNB()
}

# 모델 학습 및 평가
results = {}

for name, model in models.items():
    # 모델 학습
    model.fit(X_train_scaled, y_train)
    
    # 예측
    y_pred = model.predict(X_test_scaled)
    
    # 평가
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"\n{name}:")
    print(f"정확도: {accuracy:.4f}")
    print(f"분류 보고서:\n{classification_report(y_test, y_pred, target_names=iris.target_names)}")

# 성능 비교
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values())
plt.title('분류 모델 성능 비교')
plt.ylabel('정확도')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.show()

print("\n모델 성능 순위:")
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
for i, (name, acc) in enumerate(sorted_results, 1):
    print(f"{i}. {name}: {acc:.4f}")
```

### 3. 교차 검증
```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV

# 교차 검증
cv = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv)
    print(f"{name} 교차 검증 점수: {cv_scores}")
    print(f"평균 정확도: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# 하이퍼파라미터 튜닝 (랜덤 포레스트 예시)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print(f"\n최적 하이퍼파라미터: {grid_search.best_params_}")
print(f"최적 교차 검증 점수: {grid_search.best_score_:.4f}")

# 최적 모델로 테스트
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test_scaled)
print(f"최적 모델 테스트 정확도: {accuracy_score(y_test, y_pred_best):.4f}")
```

## 지도학습: 회귀
연속적인 값을 예측하는 회귀 모델을 만듭니다.

### 1. 회귀 데이터 준비
```python
from sklearn.datasets import load_boston, make_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 보스턴 주택 가격 데이터 (대체 데이터셋 사용)
X_california, y_california = load_boston(return_X_y=True)  # 대체 데이터셋

# 데이터프레임 생성
california_df = pd.DataFrame(X_california, columns=[
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
])
california_df['PRICE'] = y_california

print("주택 가격 데이터셋:")
print(f"특성 개수: {X_california.shape[1]}")
print(f"샘플 개수: {X_california.shape[0]}")
print("\n데이터 샘플:")
print(california_df.head())

# 기본 통계
print("\n기본 통계:")
print(california_df.describe())

# 데이터 분할
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_california, y_california, test_size=0.3, random_state=42
)

# 특성 스케일링
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)
```

### 2. 회귀 모델 학습
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# 회귀 모델 정의
regression_models = {
    '선형 회귀': LinearRegression(),
    '릿지 회귀': Ridge(alpha=1.0),
    '라쏘 회귀': Lasso(alpha=1.0),
    '결정 트리 회귀': DecisionTreeRegressor(random_state=42),
    '랜덤 포레스트 회귀': RandomForestRegressor(random_state=42),
    '그래디언트 부스팅': GradientBoostingRegressor(random_state=42),
    'SVM 회귀': SVR(kernel='rbf')
}

# 모델 학습 및 평가
regression_results = {}

for name, model in regression_models.items():
    # 모델 학습
    model.fit(X_train_reg_scaled, y_train_reg)
    
    # 예측
    y_pred_reg = model.predict(X_test_reg_scaled)
    
    # 평가 지표
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)
    
    regression_results[name] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    print(f"\n{name}:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

# 성능 비교 시각화
metrics = ['MSE', 'RMSE', 'MAE', 'R2']
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for i, metric in enumerate(metrics):
    values = [regression_results[model][metric] for model in regression_results.keys()]
    models = list(regression_results.keys())
    
    axes[i].bar(models, values)
    axes[i].set_title(f'{metric} 비교')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 3. 회귀 결과 시각화
```python
# 최적 모델 (랜덤 포레스트) 결과 시각화
best_reg_model = regression_models['랜덤 포레스트 회귀']
y_pred_best_reg = best_reg_model.predict(X_test_reg_scaled)

# 실제값 vs 예측값 산점도
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test_reg, y_pred_best_reg, alpha=0.6)
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel('실제값')
plt.ylabel('예측값')
plt.title('실제값 vs 예측값')
plt.grid(True, alpha=0.3)

# 잔차 플롯
residuals = y_test_reg - y_pred_best_reg

plt.subplot(1, 2, 2)
plt.scatter(y_pred_best_reg, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('예측값')
plt.ylabel('잔차')
plt.title('잔차 플롯')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 특성 중요도 (랜덤 포레스트)
feature_importance = best_reg_model.feature_importances_
feature_names = california_df.columns[:-1]

plt.figure(figsize=(10, 6))
indices = np.argsort(feature_importance)[::-1]
plt.bar(range(len(feature_importance)), feature_importance[indices])
plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=45)
plt.title('특성 중요도 (랜덤 포레스트)')
plt.xlabel('특성')
plt.ylabel('중요도')
plt.tight_layout()
plt.show()
```

## 비지도학습: 클러스터링
레이블이 없는 데이터를 그룹으로 묶습니다.

### 1. K-평균 클러스터링
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 클러스터링 데이터 생성
X_cluster, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)

# 최적 클러스터 수 찾기 (엘보우 방법)
inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_cluster, kmeans.labels_))

# 엘보우 방법 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(k_range, inertias, 'bo-')
ax1.set_xlabel('클러스터 수 (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('엘보우 방법')
ax1.grid(True, alpha=0.3)

ax2.plot(k_range, silhouette_scores, 'ro-')
ax2.set_xlabel('클러스터 수 (k)')
ax2.set_ylabel('실루엣 점수')
ax2.set_title('실루엣 분석')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 최적 클러스터로 K-평균 수행
optimal_k = 4  # 엘보우와 실루엣 점수를 기반으로 선택
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_cluster)

# 클러스터링 결과 시각화
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_cluster[:, 0], X_cluster[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=300, c='red', marker='x', linewidths=3, label='클러스터 중심')
plt.xlabel('특성 1')
plt.ylabel('특성 2')
plt.title(f'K-평균 클러스터링 (k={optimal_k})')
plt.legend()
plt.colorbar(scatter, label='클러스터')
plt.grid(True, alpha=0.3)
plt.show()

print(f"최적 클러스터 수: {optimal_k}")
print(f"실루엣 점수: {silhouette_score(X_cluster, cluster_labels):.4f}")
```

### 2. 계층적 클러스터링
```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

# 계층적 클러스터링 데이터
X_hierarchical, _ = make_blobs(n_samples=100, centers=3, random_state=42)

# 덴드로그램
linkage_matrix = linkage(X_hierarchical, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix)
plt.title('덴드로그램')
plt.xlabel('샘플 인덱스')
plt.ylabel('거리')
plt.grid(True, alpha=0.3)
plt.show()

# 계층적 클러스터링 수행
hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_hierarchical)

# 결과 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X_hierarchical[:, 0], X_hierarchical[:, 1], c=hierarchical_labels, cmap='viridis', alpha=0.6)
plt.xlabel('특성 1')
plt.ylabel('특성 2')
plt.title('계층적 클러스터링 결과')
plt.grid(True, alpha=0.3)
plt.show()
```

### 3. DBSCAN 클러스터링
```python
from sklearn.cluster import DBSCAN

# DBSCAN 데이터 (달 모양)
from sklearn.datasets import make_moons
X_moons, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

# DBSCAN 클러스터링
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_moons)

# K-평균과 비교
kmeans_moons = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_moons_labels = kmeans_moons.fit_predict(X_moons)

# 비교 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# DBSCAN 결과
ax1.scatter(X_moons[:, 0], X_moons[:, 1], c=dbscan_labels, cmap='viridis', alpha=0.6)
ax1.set_title('DBSCAN 클러스터링')
ax1.set_xlabel('특성 1')
ax1.set_ylabel('특성 2')
ax1.grid(True, alpha=0.3)

# K-평균 결과
ax2.scatter(X_moons[:, 0], X_moons[:, 1], c=kmeans_moons_labels, cmap='viridis', alpha=0.6)
ax2.set_title('K-평균 클러스터링')
ax2.set_xlabel('특성 1')
ax2.set_ylabel('특성 2')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"DBSCAN 클러스터 수: {len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)}")
print(f"노이즈 포인트 수: {list(dbscan_labels).count(-1)}")
```

## 차원 축소
고차원 데이터를 저차원으로 변환합니다.

### 1. PCA (주성분 분석)
```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# 숫자 데이터셋
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

print(f"원본 데이터 차원: {X_digits.shape}")

# PCA 수행
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_digits)

print(f"PCA 후 차원: {X_pca.shape}")
print(f"설명된 분산 비율: {pca.explained_variance_ratio_.sum():.4f}")

# PCA 결과 시각화
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_digits, cmap='tab10', alpha=0.6)
plt.xlabel('주성분 1')
plt.ylabel('주성분 2')
plt.title('PCA를 이용한 숫자 데이터 시각화')
plt.colorbar(scatter, label='숫자')
plt.grid(True, alpha=0.3)
plt.show()

# 설명된 분산 비율
pca_full = PCA()
pca_full.fit(X_digits)

plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), 'bo-')
plt.xlabel('주성분 수')
plt.ylabel('누적 설명된 분산 비율')
plt.title('PCA 설명된 분산 비율')
plt.grid(True, alpha=0.3)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% 설명')
plt.legend()
plt.show()
```

### 2. t-SNE
```python
from sklearn.manifold import TSNE

# t-SNE 수행
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_digits)

# t-SNE 결과 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter1 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_digits, cmap='tab10', alpha=0.6)
plt.xlabel('주성분 1')
plt.ylabel('주성분 2')
plt.title('PCA 시각화')
plt.colorbar(scatter1, label='숫자')

plt.subplot(1, 2, 2)
scatter2 = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_digits, cmap='tab10', alpha=0.6)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE 시각화')
plt.colorbar(scatter2, label='숫자')

plt.tight_layout()
plt.show()
```

## 모델 평가와 선택
다양한 평가 지표를 사용하여 모델 성능을 평가합니다.

### 1. 분류 모델 평가
```python
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

# 다중 클래스 ROC 곡선 (Iris 데이터)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

# 랜덤 포레스트로 확률 예측
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_scaled, y_train)
y_score = rf_classifier.predict_proba(X_test_scaled)

# ROC 곡선
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# ROC 곡선 시각화
plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('다중 클래스 ROC 곡선')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

# 혼동 행렬
y_pred_rf = rf_classifier.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('예측 클래스')
plt.ylabel('실제 클래스')
plt.title('혼동 행렬')
plt.show()
```

### 2. 모델 선택 전략
```python
from sklearn.model_selection import cross_validate, StratifiedKFold

# 교차 검증으로 다양한 평가 지표 확인
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

cv_results = {}
for name, model in models.items():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(model, X_train_scaled, y_train, cv=cv, scoring=scoring)
    cv_results[name] = scores
    
    print(f"\n{name}:")
    for metric in scoring:
        test_score = f'test_{metric}'
        mean_score = scores[test_score].mean()
        std_score = scores[test_score].std()
        print(f"{metric}: {mean_score:.4f} (±{std_score:.4f})")

# 결과를 데이터프레임으로 정리
results_df = pd.DataFrame({
    name: {
        'Accuracy': cv_results[name]['test_accuracy'].mean(),
        'Precision': cv_results[name]['test_precision_macro'].mean(),
        'Recall': cv_results[name]['test_recall_macro'].mean(),
        'F1-Score': cv_results[name]['test_f1_macro'].mean()
    }
    for name in models.keys()
}).T

print("\n=== 모델 성능 요약 ===")
print(results_df.round(4))

# 성능 시각화
results_df.plot(kind='bar', figsize=(12, 6))
plt.title('모델별 성능 비교')
plt.ylabel('점수')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## 실전 프로젝트: 고객 세분화
실제 비즈니스 시나리오에서 머신러닝을 적용합니다.

### 1. 고객 데이터 생성 및 전처리
```python
# 고객 데이터 생성
np.random.seed(42)
n_customers = 1000

customer_data = pd.DataFrame({
    '나이': np.random.randint(18, 80, n_customers),
    '소득': np.random.lognormal(10, 0.5, n_customers),
    '지출점수': np.random.uniform(1, 100, n_customers),
    '방문빈도': np.random.poisson(5, n_customers),
    '가입기간': np.random.randint(1, 60, n_customers),
    '구매상품수': np.random.randint(1, 50, n_customers)
})

# 고객 등급 레이블 생성 (지도학습용)
def assign_customer_segment(row):
    score = (row['소득'] / 100000) * 0.3 + row['지출점수'] * 0.4 + row['방문빈도'] * 0.2 + (row['가입기간'] / 60) * 0.1
    if score >= 70:
        return 'VIP'
    elif score >= 40:
        return '일반'
    else:
        return '저활동'

customer_data['고객등급'] = customer_data.apply(assign_customer_segment, axis=1)

print("고객 데이터:")
print(customer_data.head())
print(f"\n고객 등급 분포:")
print(customer_data['고객등급'].value_counts())

# 특성과 타겟 분리
X_customer = customer_data.drop('고객등급', axis=1)
y_customer = customer_data['고객등급']

# 데이터 분할
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_customer, y_customer, test_size=0.3, random_state=42, stratify=y_customer
)

# 특성 스케일링
scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)
```

### 2. 고객 등급 분류 모델
```python
# 고객 등급 분류
customer_models = {
    '로지스틱 회귀': LogisticRegression(random_state=42, max_iter=1000),
    '랜덤 포레스트': RandomForestClassifier(random_state=42),
    '그래디언트 부스팅': GradientBoostingClassifier(random_state=42)
}

customer_results = {}

for name, model in customer_models.items():
    # 모델 학습
    model.fit(X_train_c_scaled, y_train_c)
    
    # 예측
    y_pred_c = model.predict(X_test_c_scaled)
    
    # 평가
    accuracy = accuracy_score(y_test_c, y_pred_c)
    customer_results[name] = accuracy
    
    print(f"\n{name} 고객 등급 분류:")
    print(f"정확도: {accuracy:.4f}")
    print(f"분류 보고서:\n{classification_report(y_test_c, y_pred_c)}")

# 최적 모델로 특성 중요도 분석
best_customer_model = customer_models['랜덤 포레스트']
feature_importance_c = best_customer_model.feature_importances_
feature_names_c = X_customer.columns

plt.figure(figsize=(10, 6))
indices_c = np.argsort(feature_importance_c)[::-1]
plt.bar(range(len(feature_importance_c)), feature_importance_c[indices_c])
plt.xticks(range(len(feature_importance_c)), [feature_names_c[i] for i in indices_c], rotation=45)
plt.title('고객 등급 분류 특성 중요도')
plt.xlabel('특성')
plt.ylabel('중요도')
plt.tight_layout()
plt.show()
```

### 3. 고객 클러스터링 (비지도학습)
```python
# 고객 클러스터링 (레이블 없이)
X_clustering = scaler_c.fit_transform(X_customer)

# 최적 클러스터 수 찾기
inertias_c = []
silhouette_scores_c = []
k_range_c = range(2, 8)

for k in k_range_c:
    kmeans_c = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_c.fit(X_clustering)
    inertias_c.append(kmeans_c.inertia_)
    silhouette_scores_c.append(silhouette_score(X_clustering, kmeans_c.labels_))

# 최적 클러스터로 클러스터링
optimal_k_c = 3  # 실제로는 엘보우와 실루엣 분석으로 선택
kmeans_c = KMeans(n_clusters=optimal_k_c, random_state=42, n_init=10)
cluster_labels_c = kmeans_c.fit_predict(X_clustering)

# 클러스터별 특성 분석
customer_data['클러스터'] = cluster_labels_c
cluster_analysis = customer_data.groupby('클러스터').agg({
    '나이': 'mean',
    '소득': 'mean',
    '지출점수': 'mean',
    '방문빈도': 'mean',
    '가입기간': 'mean',
    '구매상품수': 'mean',
    '고객등급': lambda x: x.value_counts().index[0]  # 최빈값
}).round(2)

print("\n=== 클러스터별 특성 분석 ===")
print(cluster_analysis)

# 클러스터 시각화 (PCA로 2차원 축소)
pca_c = PCA(n_components=2)
X_pca_c = pca_c.fit_transform(X_clustering)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca_c[:, 0], X_pca_c[:, 1], c=cluster_labels_c, cmap='viridis', alpha=0.6)
plt.xlabel('주성분 1')
plt.ylabel('주성분 2')
plt.title('고객 클러스터링 결과 (PCA)')
plt.colorbar(scatter, label='클러스터')
plt.grid(True, alpha=0.3)
plt.show()
```

이 머신러닝 예제들을 통해 지도학습과 비지도학습의 핵심 개념과 실전 적용 방법을 익힐 수 있습니다.
