import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("=== 머신러닝 입문 (Scikit-learn) 예제 ===")

# 1. 데이터 생성
print("\n=== 1. 데이터 생성 ===")

# 분류 데이터
X_class, y_class = make_classification(
    n_samples=1000, n_features=2, n_redundant=0,
    n_informative=2, n_clusters_per_class=1, random_state=42
)

# 클러스터링 데이터
X_cluster, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)

# 회귀 데이터
X_reg, y_reg = make_regression(n_samples=1000, n_features=1, noise=20, random_state=42)

print(f"분류 데이터: {X_class.shape}, 타겟: {y_class.shape}")
print(f"클러스터링 데이터: {X_cluster.shape}")
print(f"회귀 데이터: {X_reg.shape}, 타겟: {y_reg.shape}")

# 2. 데이터 시각화
print("\n=== 2. 데이터 시각화 ===")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 분류 데이터
axes[0].scatter(X_class[:, 0], X_class[:, 1], c=y_class, cmap='viridis', alpha=0.6)
axes[0].set_title('분류 데이터')
axes[0].set_xlabel('특성 1')
axes[0].set_ylabel('특성 2')

# 클러스터링 데이터
axes[1].scatter(X_cluster[:, 0], X_cluster[:, 1], alpha=0.6)
axes[1].set_title('클러스터링 데이터')
axes[1].set_xlabel('특성 1')
axes[1].set_ylabel('특성 2')

# 회귀 데이터
axes[2].scatter(X_reg, y_reg, alpha=0.6)
axes[2].set_title('회귀 데이터')
axes[2].set_xlabel('특성')
axes[2].set_ylabel('타겟')

plt.tight_layout()
plt.show()

# 3. 지도학습: 분류
print("\n=== 3. 지도학습: 분류 ===")

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_class, y_class, test_size=0.3, random_state=42, stratify=y_class
)

# 특성 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 학습
models = {
    '로지스틱 회귀': LogisticRegression(random_state=42),
    '랜덤 포레스트': RandomForestClassifier(random_state=42)
}

for name, model in models.items():
    # 모델 학습
    model.fit(X_train_scaled, y_train)
    
    # 예측
    y_pred = model.predict(X_test_scaled)
    
    # 평가
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{name}:")
    print(f"정확도: {accuracy:.4f}")
    print(f"분류 보고서:\n{classification_report(y_test, y_pred)}")

# 4. 비지도학습: 클러스터링
print("\n=== 4. 비지도학습: 클러스터링 ===")

# 최적 클러스터 수 찾기
inertias = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster)
    inertias.append(kmeans.inertia_)

# 엘보우 방법 시각화
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('클러스터 수 (k)')
plt.ylabel('Inertia')
plt.title('엘보우 방법')
plt.grid(True, alpha=0.3)
plt.show()

# 최적 클러스터로 K-평균 수행
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_cluster)

# 클러스터링 결과 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X_cluster[:, 0], X_cluster[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=300, c='red', marker='x', linewidths=3, label='클러스터 중심')
plt.xlabel('특성 1')
plt.ylabel('특성 2')
plt.title(f'K-평균 클러스터링 (k={optimal_k})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 5. 차원 축소: PCA
print("\n=== 5. 차원 축소: PCA ===")

# 고차원 데이터 생성
X_high_dim, _ = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)

print(f"원본 데이터 차원: {X_high_dim.shape}")

# PCA 수행
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_high_dim)

print(f"PCA 후 차원: {X_pca.shape}")
print(f"설명된 분산 비율: {pca.explained_variance_ratio_.sum():.4f}")

# PCA 결과 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
plt.xlabel('주성분 1')
plt.ylabel('주성분 2')
plt.title('PCA 결과')
plt.grid(True, alpha=0.3)
plt.show()

# 6. 실전 예제: 고객 세분화
print("\n=== 6. 실전 예제: 고객 세분화 ===")

# 고객 데이터 생성
np.random.seed(42)
n_customers = 500

customer_data = pd.DataFrame({
    '나이': np.random.randint(18, 80, n_customers),
    '소득': np.random.lognormal(10, 0.5, n_customers),
    '지출점수': np.random.uniform(1, 100, n_customers),
    '방문빈도': np.random.poisson(5, n_customers),
    '가입기간': np.random.randint(1, 60, n_customers)
})

print("고객 데이터:")
print(customer_data.head())
print(f"\n고객 데이터 통계:")
print(customer_data.describe())

# 고객 클러스터링
X_customer = customer_data.values
X_customer_scaled = StandardScaler().fit_transform(X_customer)

# 고객 클러스터링
kmeans_customer = KMeans(n_clusters=3, random_state=42, n_init=10)
customer_clusters = kmeans_customer.fit_predict(X_customer_scaled)

# 클러스터별 특성 분석
customer_data['클러스터'] = customer_clusters
cluster_analysis = customer_data.groupby('클러스터').mean()

print("\n=== 클러스터별 특성 분석 ===")
print(cluster_analysis)

# 클러스터 시각화
pca_customer = PCA(n_components=2)
X_customer_pca = pca_customer.fit_transform(X_customer_scaled)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_customer_pca[:, 0], X_customer_pca[:, 1], c=customer_clusters, cmap='viridis', alpha=0.6)
plt.xlabel('주성분 1')
plt.ylabel('주성분 2')
plt.title('고객 클러스터링 결과')
plt.colorbar(scatter, label='클러스터')
plt.grid(True, alpha=0.3)
plt.show()

# 7. 모델 평가
print("\n=== 7. 모델 평가 ===")

# 최적 분류 모델로 혼동 행렬
best_model = models['랜덤 포레스트']
y_pred_best = best_model.predict(X_test_scaled)

# 혼동 행렬
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 5))
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('예측 클래스')
plt.ylabel('실제 클래스')
plt.title('혼동 행렬')
plt.show()

# 특성 중요도 (랜덤 포레스트)
feature_importance = best_model.feature_importances_
feature_names = ['특성 1', '특성 2']

plt.figure(figsize=(8, 5))
plt.bar(feature_names, feature_importance)
plt.title('특성 중요도')
plt.ylabel('중요도')
plt.grid(True, alpha=0.3)
plt.show()

print("\n머신러닝 입문 예제 완료!")
print("지도학습(분류), 비지도학습(클러스터링), 차원축소(PCA)의 기본 개념을 다뤘습니다.")
