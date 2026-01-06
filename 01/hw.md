# Homework - Python 기초

## 이 homework의 의미
Python 기초 문법과 데이터 분석 라이브러리를 실습하면서 프로그래밍 사고력과 데이터 처리 능력을 키우는 과제야. 실무에서 바로 써먹을 수 있는 기본기를 다지는 거지.

## 관련 정보 사이트
1. **Python 공식 문서**: https://docs.python.org/ko/3/
2. **Real Python**: https://realpython.com/

## 프로세스 진행 논리적 흐름
1. **기초 문법 학습** → 변수, 데이터 타입, 제어문 이해
2. **함수 작성** → 재사용 가능한 코드 블록 만들기
3. **라이브러리 활용** → NumPy, Pandas로 데이터 처리
4. **시각화** → Matplotlib으로 결과 표현
5. **실전 적용** → 작은 프로젝트로 통합 연습

## 권장 코드
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성 및 분석
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'score': [85, 92, 78]
}
df = pd.DataFrame(data)

# 기본 통계
print(df.describe())

# 시각화
plt.bar(df['name'], df['score'])
plt.title('Student Scores')
plt.xlabel('Name')
plt.ylabel('Score')
plt.show()
```

## 관련 업계 회사
1. **Google** - Python을 주력 언어로 사용하는 글로벌 테크 기업
2. **Netflix** - 데이터 분석과 추천 시스템에 Python 활용
3. **Instagram** - Django(Python 프레임워크) 기반 서비스
