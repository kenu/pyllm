# 딥러닝 기초 (TensorFlow/PyTorch)

## 신경망 기초 개념
딥러닝의 핵심 개념과 신경망의 기본 구조를 이해합니다.

### 1. 퍼셉트론과 신경망
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 퍼셉트론 기본 구조
class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.learning_rate = 0.01
    
    def forward(self, x):
        """순전파"""
        return np.dot(x, self.weights) + self.bias
    
    def activation(self, x):
        """활성화 함수 (계단 함수)"""
        return 1 if x > 0 else 0
    
    def predict(self, x):
        """예측"""
        return self.activation(self.forward(x))
    
    def train(self, X, y, epochs=100):
        """학습"""
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                
                # 가중치 업데이트
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error
                
                total_error += abs(error)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Error = {total_error}")

# AND 게이트 예제
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# 퍼셉트론 학습
perceptron_and = Perceptron(2)
perceptron_and.train(X_and, y_and, epochs=50)

# 테스트
print("\nAND 게이트 테스트:")
for x in X_and:
    prediction = perceptron_and.predict(x)
    print(f"Input: {x}, Output: {prediction}")

# 시각화
plt.figure(figsize=(12, 4))

# 결정 경계
plt.subplot(1, 2, 1)
x1 = np.linspace(-0.5, 1.5, 100)
x2 = -(perceptron_and.weights[0] * x1 + perceptron_and.bias) / perceptron_and.weights[1]
plt.plot(x1, x2, 'r-', label='결정 경계')

# 데이터 포인트
for i, (x, y) in enumerate(zip(X_and, y_and)):
    color = 'blue' if y == 0 else 'green'
    marker = 'o' if perceptron_and.predict(x) == y else 'x'
    plt.scatter(x[0], x[1], c=color, marker=marker, s=100, label=f'{"0" if y == 0 else "1"}')

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.title('퍼셉트론 AND 게이트')
plt.legend()
plt.grid(True, alpha=0.3)

# 활성화 함수 비교
plt.subplot(1, 2, 2)
x = np.linspace(-5, 5, 100)

# 계단 함수
step = np.where(x > 0, 1, 0)

# 시그모이드
sigmoid = 1 / (1 + np.exp(-x))

# ReLU
relu = np.maximum(0, x)

# Tanh
tanh = np.tanh(x)

plt.plot(x, step, label='계단 함수', linewidth=2)
plt.plot(x, sigmoid, label='시그모이드', linewidth=2)
plt.plot(x, relu, label='ReLU', linewidth=2)
plt.plot(x, tanh, label='Tanh', linewidth=2)
plt.xlabel('입력')
plt.ylabel('출력')
plt.title('활성화 함수 비교')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## TensorFlow 기초
TensorFlow를 사용하여 간단한 신경망을 구현합니다.

### 1. TensorFlow 설치 및 기본 사용
```python
# TensorFlow 2.x 기반 코드
import tensorflow as tf
from tensorflow import keras
import numpy as np

print(f"TensorFlow 버전: {tf.__version__}")

# 기본 텐서 연산
print("\n=== 기본 텐서 연산 ===")

# 텐서 생성
tensor_a = tf.constant([[1, 2], [3, 4]])
tensor_b = tf.constant([[5, 6], [7, 8]])

print("텐서 A:")
print(tensor_a)
print("텐서 B:")
print(tensor_b)

# 기본 연산
print("\n텐서 연산:")
print("덧셈:")
print(tf.add(tensor_a, tensor_b))
print("곱셈:")
print(tf.matmul(tensor_a, tensor_b))

# 자동 미분
print("\n=== 자동 미분 ===")
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x**2

dy_dx = tape.gradient(y, x)
print(f"y = x^2, x = 3일 때 dy/dx = {dy_dx.numpy()}")
```

### 2. 간단한 신경망 구현
```python
# MNIST 데이터셋 로드
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 전처리
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0

print(f"훈련 데이터: {x_train.shape}")
print(f"테스트 데이터: {x_test.shape}")

# 간단한 신경망 모델
def create_simple_model():
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 모델 생성
model = create_simple_model()

print("\n모델 구조:")
model.summary()

# 모델 학습
print("\n모델 학습 시작...")
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

# 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n테스트 정확도: {test_acc:.4f}")

# 학습 과정 시각화
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='훈련 손실')
plt.plot(history.history['val_loss'], label='검증 손실')
plt.xlabel('Epoch')
plt.ylabel('손실')
plt.title('학습 손실')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='훈련 정확도')
plt.plot(history.history['val_accuracy'], label='검증 정확도')
plt.xlabel('Epoch')
plt.ylabel('정확도')
plt.title('학습 정확도')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## PyTorch 기초
PyTorch를 사용하여 동적인 신경망을 구현합니다.

### 1. PyTorch 기본 개념
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms

print(f"PyTorch 버전: {torch.__version__}")

# 기본 텐서 연산
print("\n=== PyTorch 텐서 연산 ===")

# 텐서 생성
tensor_a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
tensor_b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

print("텐서 A:")
print(tensor_a)
print("텐서 B:")
print(tensor_b)

# 기본 연산
print("\n텐서 연산:")
print("덧셈:")
print(torch.add(tensor_a, tensor_b))
print("곱셈:")
print(torch.matmul(tensor_a, tensor_b))

# 자동 미분
print("\n=== 자동 미분 ===")
x = torch.tensor(3.0, requires_grad=True)

y = x**2
y.backward()

print(f"y = x^2, x = 3일 때 dy/dx = {x.grad}")
```

### 2. PyTorch 신경망 구현
```python
# PyTorch 신경망 클래스
class SimpleNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 데이터 준비
# MNIST 데이터를 PyTorch 텐서로 변환
x_train_torch = torch.FloatTensor(x_train)
y_train_torch = torch.LongTensor(y_train)
x_test_torch = torch.FloatTensor(x_test)
y_test_torch = torch.LongTensor(y_test)

# 데이터로더 생성
train_dataset = TensorDataset(x_train_torch, y_train_torch)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 모델, 손실 함수, 옵티마이저
model_pytorch = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_pytorch.parameters(), lr=0.001)

print("PyTorch 모델 구조:")
print(model_pytorch)

# 학습 함수
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            # 순전파
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if epoch % 2 == 0:
            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
    
    return train_losses

# 모델 학습
print("\nPyTorch 모델 학습 시작...")
train_losses = train_model(model_pytorch, train_loader, criterion, optimizer, epochs=10)

# 모델 평가
model_pytorch.eval()
with torch.no_grad():
    outputs = model_pytorch(x_test_torch)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y_test_torch).float().mean()
    print(f"\n테스트 정확도: {accuracy:.4f}")

# 학습 과정 시각화
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='훈련 손실')
plt.xlabel('Epoch')
plt.ylabel('손실')
plt.title('PyTorch 학습 손실')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 합성곱 신경망 (CNN)
이미지 분류를 위한 CNN을 구현합니다.

### 1. TensorFlow CNN
```python
# CNN 모델 (TensorFlow)
def create_cnn_model():
    model = keras.Sequential([
        # 입력 형태: (28, 28, 1)
        keras.layers.Reshape((28, 28, 1), input_shape=(784,)),
        
        # 첫 번째 합성곱 블록
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # 두 번째 합성곱 블록
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # 완전 연결 레이어
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# CNN 모델 생성 및 학습
cnn_model = create_cnn_model()

print("CNN 모델 구조:")
cnn_model.summary()

# 데이터 형태 변경 (CNN을 위해)
x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_test_cnn = x_test.reshape(-1, 28, 28, 1)

print(f"\nCNN 데이터 형태: {x_train_cnn.shape}")

# CNN 학습
print("\nCNN 모델 학습 시작...")
cnn_history = cnn_model.fit(
    x_train_cnn, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

# CNN 평가
cnn_test_loss, cnn_test_acc = cnn_model.evaluate(x_test_cnn, y_test)
print(f"\nCNN 테스트 정확도: {cnn_test_acc:.4f}")
```

### 2. PyTorch CNN
```python
# CNN 모델 (PyTorch)
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 입력 형태: (batch, 1, 28, 28)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = x.view(-1, 64 * 7 * 7)  # flatten
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

# CNN 데이터 준비
x_train_cnn_torch = x_train.reshape(-1, 1, 28, 28)
x_test_cnn_torch = x_test.reshape(-1, 1, 28, 28)

train_cnn_dataset = TensorDataset(torch.FloatTensor(x_train_cnn_torch), 
                                 torch.LongTensor(y_train))
train_cnn_loader = DataLoader(train_cnn_dataset, batch_size=128, shuffle=True)

# CNN 모델 학습
cnn_model_pytorch = CNN()
cnn_criterion = nn.CrossEntropyLoss()
cnn_optimizer = optim.Adam(cnn_model_pytorch.parameters(), lr=0.001)

print("PyTorch CNN 모델 학습 시작...")
cnn_train_losses = train_model(cnn_model_pytorch, train_cnn_loader, 
                              cnn_criterion, cnn_optimizer, epochs=10)

# CNN 평가
cnn_model_pytorch.eval()
with torch.no_grad():
    outputs = cnn_model_pytorch(torch.FloatTensor(x_test_cnn_torch))
    _, predicted = torch.max(outputs.data, 1)
    cnn_accuracy = (predicted == torch.LongTensor(y_test)).float().mean()
    print(f"\nPyTorch CNN 테스트 정확도: {cnn_accuracy:.4f}")
```

## 순환 신경망 (RNN)
시퀀스 데이터를 처리하는 RNN을 구현합니다.

### 1. LSTM을 이용한 텍스트 분류
```python
# IMDB 데이터셋 (TensorFlow)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# IMDB 데이터셋 로드
max_features = 10000  # 사용할 단어 수
maxlen = 500  # 시퀀스 최대 길이

(x_train_imdb, y_train_imdb), (x_test_imdb, y_test_imdb) = imdb.load_data(
    num_words=max_features
)

# 시퀀스 패딩
x_train_imdb = pad_sequences(x_train_imdb, maxlen=maxlen)
x_test_imdb = pad_sequences(x_test_imdb, maxlen=maxlen)

print(f"IMDB 데이터: {x_train_imdb.shape}")
print(f"샘플 시퀀스: {x_train_imdb[0][:10]}...")

# LSTM 모델
def create_lstm_model():
    model = keras.Sequential([
        keras.layers.Embedding(max_features, 128, input_length=maxlen),
        keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# LSTM 모델 학습
lstm_model = create_lstm_model()

print("\nLSTM 모델 구조:")
lstm_model.summary()

print("\nLSTM 모델 학습 시작...")
lstm_history = lstm_model.fit(
    x_train_imdb, y_train_imdb,
    epochs=5,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

# LSTM 평가
lstm_test_loss, lstm_test_acc = lstm_model.evaluate(x_test_imdb, y_test_imdb)
print(f"\nLSTM 테스트 정확도: {lstm_test_acc:.4f}")
```

## 전이학습
사전 훈련된 모델을 사용하는 전이학습을 구현합니다.

### 1. 전이학습을 이용한 이미지 분류
```python
# CIFAR-10 데이터셋
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, GlobalAveragePooling2D

# CIFAR-10 데이터 로드
(x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = cifar10.load_data()

# 데이터 전처리
x_train_cifar = x_train_cifar.astype('float32') / 255.0
x_test_cifar = x_test_cifar.astype('float32') / 255.0

# 레이블 원-핫 인코딩
y_train_cifar = keras.utils.to_categorical(y_train_cifar, 10)
y_test_cifar = keras.utils.to_categorical(y_test_cifar, 10)

print(f"CIFAR-10 데이터: {x_train_cifar.shape}")

# 전이학습 모델 (VGG16)
def create_transfer_model():
    # VGG16 기반 모델 (상위 레이어만 사용)
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(32, 32, 3)
    )
    
    # 기반 모델 동결
    base_model.trainable = False
    
    # 새로운 분류 레이어 추가
    inputs = Input(shape=(32, 32, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(10, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 전이학습 모델 생성 및 학습
transfer_model = create_transfer_model()

print("\n전이학습 모델 구조:")
transfer_model.summary()

print("\n전이학습 모델 학습 시작...")
transfer_history = transfer_model.fit(
    x_train_cifar, y_train_cifar,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

# 전이학습 평가
transfer_test_loss, transfer_test_acc = transfer_model.evaluate(x_test_cifar, y_test_cifar)
print(f"\n전이학습 테스트 정확도: {transfer_test_acc:.4f}")
```

## 모델 저장 및 로드
학습된 모델을 저장하고 다시 로드하는 방법을 익힙니다.

### 1. 모델 저장 및 로드
```python
# TensorFlow 모델 저장
print("=== 모델 저장 및 로드 ===")

# 모델 저장
model.save('/Users/kenu/git/pyllm/10/mnist_model.h5')
print("TensorFlow 모델이 저장되었습니다.")

# 모델 로드
loaded_model = keras.models.load_model('/Users/kenu/git/pyllm/10/mnist_model.h5')
print("TensorFlow 모델이 로드되었습니다.")

# 로드된 모델 테스트
loaded_test_loss, loaded_test_acc = loaded_model.evaluate(x_test, y_test)
print(f"로드된 모델 테스트 정확도: {loaded_test_acc:.4f}")

# PyTorch 모델 저장
torch.save(model_pytorch.state_dict(), '/Users/kenu/git/pyllm/10/pytorch_model.pth')
print("\nPyTorch 모델이 저장되었습니다.")

# PyTorch 모델 로드
loaded_pytorch_model = SimpleNN()
loaded_pytorch_model.load_state_dict(torch.load('/Users/kenu/git/pyllm/10/pytorch_model.pth'))
print("PyTorch 모델이 로드되었습니다.")

# 로드된 PyTorch 모델 테스트
loaded_pytorch_model.eval()
with torch.no_grad():
    outputs = loaded_pytorch_model(x_test_torch)
    _, predicted = torch.max(outputs.data, 1)
    loaded_pytorch_accuracy = (predicted == y_test_torch).float().mean()
    print(f"로드된 PyTorch 모델 테스트 정확도: {loaded_pytorch_accuracy:.4f}")
```

## 실전 프로젝트: 이미지 분류 애플리케이션
실제 이미지 분류 애플리케이션을 만듭니다.

### 1. 커스텀 이미지 분류기
```python
# 커스텀 이미지 분류기 클래스
class ImageClassifier:
    def __init__(self, model_type='tensorflow'):
        self.model_type = model_type
        self.model = None
        self.class_names = None
    
    def create_model(self, input_shape, num_classes):
        """모델 생성"""
        if self.model_type == 'tensorflow':
            self.model = keras.Sequential([
                keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(64, (3, 3), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dense(num_classes, activation='softmax')
            ])
            
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
    
    def train(self, x_train, y_train, epochs=10, batch_size=32):
        """모델 학습"""
        if self.model_type == 'tensorflow':
            history = self.model.fit(
                x_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=1
            )
            return history
    
    def predict(self, x):
        """예측"""
        if self.model_type == 'tensorflow':
            predictions = self.model.predict(x)
            return np.argmax(predictions, axis=1)
    
    def evaluate(self, x_test, y_test):
        """모델 평가"""
        if self.model_type == 'tensorflow':
            loss, accuracy = self.model.evaluate(x_test, y_test)
            return loss, accuracy

# 이미지 분류기 사용 예제
classifier = ImageClassifier('tensorflow')

# 모델 생성
classifier.create_model((28, 28, 1), 10)

# 학습
print("\n커스텀 이미지 분류기 학습...")
history = classifier.train(x_train_cnn, y_train, epochs=5)

# 평가
loss, accuracy = classifier.evaluate(x_test_cnn, y_test)
print(f"\n커스텀 분류기 정확도: {accuracy:.4f}")

# 예측 예제
sample_images = x_test_cnn[:5]
predictions = classifier.predict(sample_images)
print(f"\n샘플 예측: {predictions}")
print(f"실제 레이블: {y_test[:5]}")
```

## 최신 딥러닝 트렌드
최신 딥러닝 기술들을 소개합니다.

### 1. 어텐션 메커니즘
```python
# 간단한 어텐션 메커니즘 구현
class SimpleAttention(nn.Module):
    def __init__(, input_size):
        super(SimpleAttention, self).__init__()
        self.attention = nn.Linear(input_size, 1)
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        attention_weights = torch.softmax(self.attention(x), dim=1)
        # attention_weights: (batch_size, seq_len, 1)
        context_vector = torch.sum(attention_weights * x, dim=1)
        # context_vector: (batch_size, input_size)
        return context_vector, attention_weights

print("\n=== 최신 딥러닝 트렌드 ===")
print("1. 어텐션 메커니즘: 시퀀스 데이터의 중요한 부분에 집중")
print("2. 트랜스포머: 어텐션 기반의 시퀀스 처리 아키텍처")
print("3. GAN: 생성적 적대 신경망을 통한 데이터 생성")
print("4. 강화학습: 환경과의 상호작용을 통한 학습")
print("5. 메타러닝: 학습하는 방법을 학습하는 모델")
```

이 딥러닝 예제들을 통해 TensorFlow와 PyTorch의 기본 개념과 실전 활용 방법을 익힐 수 있습니다.
