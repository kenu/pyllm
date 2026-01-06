# Homework - 딥러닝

## 이 homework의 의미
신경망을 활용한 딥러닝 모델을 구축하고 이미지, 텍스트 등 복잡한 데이터를 처리하는 과제야. CNN, RNN 등 다양한 아키텍처를 실습하면서 최신 AI 기술을 익히는 거지.

## 관련 정보 사이트
1. **TensorFlow Tutorials**: https://www.tensorflow.org/tutorials
2. **PyTorch Tutorials**: https://pytorch.org/tutorials/

## 프로세스 진행 논리적 흐름
1. **문제 정의** → 이미지 분류, 텍스트 생성 등
2. **데이터 전처리** → 정규화, 증강, 토큰화
3. **모델 설계** → 레이어 구성 및 아키텍처 선택
4. **학습** → 손실 함수, 옵티마이저 설정
5. **평가 및 개선** → 과적합 방지, 성능 향상

## 권장 코드
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# MNIST 데이터 로드
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 데이터 전처리
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# CNN 모델 구축
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# 컴파일
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 학습
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2
)

# 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n테스트 정확도: {test_acc:.4f}")
```

## 관련 업계 회사
1. **NVIDIA** - GPU 및 딥러닝 하드웨어/소프트웨어
2. **DeepMind (Google)** - AlphaGo, AlphaFold 개발
3. **OpenAI** - GPT, DALL-E 등 대규모 딥러닝 모델
