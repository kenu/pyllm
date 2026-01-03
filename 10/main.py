import numpy as np
import matplotlib.pyplot as plt

# TensorFlow 임포트 (설치되지 않은 경우 대비)
try:
    import tensorflow as tf
    from tensorflow import keras
    print(f"TensorFlow 버전: {tf.__version__}")
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow가 설치되지 않았습니다. pip install tensorflow로 설치해주세요.")
    TF_AVAILABLE = False

# PyTorch 임포트 (설치되지 않은 경우 대비)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    print(f"PyTorch 버전: {torch.__version__}")
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch가 설치되지 않았습니다. pip install torch로 설치해주세요.")
    TORCH_AVAILABLE = False

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("=== 딥러닝 기초 (TensorFlow/PyTorch) 예제 ===")

# 1. 퍼셉트론 기본 구현
print("\n=== 1. 퍼셉트론 기본 구현 ===")

class SimplePerceptron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.learning_rate = 0.01
    
    def forward(self, x):
        return np.dot(x, self.weights) + self.bias
    
    def activation(self, x):
        return 1 if x > 0 else 0
    
    def predict(self, x):
        return self.activation(self.forward(x))
    
    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error
                
                total_error += abs(error)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Error = {total_error}")

# AND 게이트 예제
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

perceptron = SimplePerceptron(2)
perceptron.train(X_and, y_and, epochs=100)

print("\nAND 게이트 테스트:")
for x in X_and:
    prediction = perceptron.predict(x)
    print(f"Input: {x}, Output: {prediction}")

# 2. TensorFlow 신경망 (가능한 경우)
if TF_AVAILABLE:
    print("\n=== 2. TensorFlow 간단 신경망 ===")
    
    # 간단한 데이터 생성
    X_tf = np.random.randn(1000, 10)
    y_tf = np.random.randint(0, 2, 1000)
    
    # 모델 생성
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("TensorFlow 모델 구조:")
    model.summary()
    
    # 모델 학습 (간단히)
    print("\nTensorFlow 모델 학습...")
    history = model.fit(X_tf, y_tf, epochs=5, batch_size=32, verbose=1)
    
    # 평가
    loss, accuracy = model.evaluate(X_tf, y_tf, verbose=0)
    print(f"TensorFlow 모델 정확도: {accuracy:.4f}")

# 3. PyTorch 신경망 (가능한 경우)
if TORCH_AVAILABLE:
    print("\n=== 3. PyTorch 간단 신경망 ===")
    
    # PyTorch 모델 정의
    class SimpleNN(nn.Module):
        def __init__(self, input_size=10, hidden_size=64, output_size=1):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, 32)
            self.fc3 = nn.Linear(32, output_size)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.sigmoid(self.fc3(x))
            return x
    
    # 데이터 준비
    X_pt = torch.FloatTensor(X_tf)
    y_pt = torch.FloatTensor(y_tf).unsqueeze(1)
    
    # 모델, 손실 함수, 옵티마이저
    model_pt = SimpleNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_pt.parameters(), lr=0.001)
    
    print("PyTorch 모델 구조:")
    print(model_pt)
    
    # 간단한 학습
    print("\nPyTorch 모델 학습...")
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model_pt(X_pt)
        loss = criterion(outputs, y_pt)
        loss.backward()
        optimizer.step()
        
        if epoch % 1 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
    # 평가
    model_pt.eval()
    with torch.no_grad():
        outputs = model_pt(X_pt)
        predictions = (outputs > 0.5).float()
        accuracy = (predictions == y_pt).float().mean()
        print(f"PyTorch 모델 정확도: {accuracy:.4f}")

# 4. 활성화 함수 비교
print("\n=== 4. 활성화 함수 비교 ===")

x = np.linspace(-5, 5, 100)

# 다양한 활성화 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# 시각화
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x, sigmoid(x), 'b-', linewidth=2)
plt.title('시그모이드 (Sigmoid)')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(x, tanh(x), 'r-', linewidth=2)
plt.title('하이퍼볼릭 탄젠트 (Tanh)')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(x, relu(x), 'g-', linewidth=2)
plt.title('ReLU (Rectified Linear Unit)')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(x, leaky_relu(x), 'm-', linewidth=2)
plt.title('Leaky ReLU')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 5. 신경망 기본 개념 시각화
print("\n=== 5. 신경망 기본 개념 ===")

# 간단한 2-레이어 신경망 시각화
def visualize_network():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 레이어 위치
    layer_sizes = [2, 3, 1]  # 입력:2, 은닉:3, 출력:1
    layer_positions = [0, 1, 2]
    
    # 노드 그리기
    for i, (size, pos) in enumerate(zip(layer_sizes, layer_positions)):
        for j in range(size):
            y_pos = j - (size - 1) / 2
            circle = plt.Circle((pos, y_pos), 0.1, color='lightblue', ec='black', linewidth=2)
            ax.add_patch(circle)
    
    # 연결선 그리기
    for i in range(len(layer_sizes) - 1):
        for j in range(layer_sizes[i]):
            for k in range(layer_sizes[i + 1]):
                y1 = j - (layer_sizes[i] - 1) / 2
                y2 = k - (layer_sizes[i + 1] - 1) / 2
                ax.plot([layer_positions[i], layer_positions[i + 1]], [y1, y2], 
                       'gray', alpha=0.5, linewidth=1)
    
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('간단한 신경망 구조', fontsize=16)
    
    # 레이어 레이블
    ax.text(-0.3, 0, '입력층', ha='center', va='center', fontsize=12)
    ax.text(1, -1.5, '은닉층', ha='center', va='center', fontsize=12)
    ax.text(2.3, 0, '출력층', ha='center', va='center', fontsize=12)
    
    plt.show()

visualize_network()

# 6. 경사하강법 시각화
print("\n=== 6. 경사하강법 시각화 ===")

def gradient_descent_visualization():
    # 손실 함수 (2차 함수)
    def loss_function(x):
        return x**2 + 2*x + 1
    
    # 미분 함수
    def gradient(x):
        return 2*x + 2
    
    # 경사하강법
    x_history = []
    x = -3.0  # 시작점
    learning_rate = 0.1
    
    for i in range(20):
        x_history.append(x)
        grad = gradient(x)
        x = x - learning_rate * grad
    
    # 시각화
    x_range = np.linspace(-4, 2, 100)
    y_range = loss_function(x_range)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, y_range, 'b-', linewidth=2, label='손실 함수')
    plt.plot(x_history, [loss_function(x) for x in x_history], 'ro-', 
             markersize=8, label='경사하강법 경로')
    
    plt.xlabel('x')
    plt.ylabel('손실')
    plt.title('경사하강법 최적화')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"최적값: {x_history[-1]:.4f}")

gradient_descent_visualization()

# 7. 최신 딥러닝 트렌드 소개
print("\n=== 7. 최신 딥러닝 트렌드 ===")
print("1. 트랜스포머 아키텍처: 어텐션 메커니즘 기반의 NLP 혁신")
print("2. GPT, BERT와 같은 대규모 언어 모델")
print("3. Stable Diffusion, DALL-E와 같은 생성형 AI")
print("4. Vision Transformer (ViT): 이미지 처리에 트랜스포머 적용")
print("5. Meta-Learning: 소량 데이터로 빠르게 학습하는 모델")
print("6. Federated Learning: 개인정보 보호 분산 학습")
print("7. Neural Architecture Search (NAS): 자동 아키텍처 설계")

print("\n딥러닝 기초 예제 완료!")
print("TensorFlow와 PyTorch의 기본 개념과 간단한 신경망 구현을 다뤘습니다.")
