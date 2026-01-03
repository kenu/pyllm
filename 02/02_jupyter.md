# Jupyter Notebook 기본 정보

## Jupyter Notebook이란?

Jupyter Notebook은 인터랙티브한 웹 기반 개발 환경으로, 코드 실행, 시각화, 텍스트 설명을 하나의 문서에서 결합할 수 있는 오픈소스 프로젝트입니다.

## 주요 특징

### 1. 인터랙티브 셀
- **코드 셀**: Python 코드를 실행하고 결과를 바로 확인
- **마크다운 셀**: 텍스트, 수식, 이미지 등을 포함한 문서화
- **출력 셀**: 실행 결과, 그래프, 테이블 등을 표시

### 2. 다양한 언어 지원
- Python (기본)
- R, Julia, Scala 등 40+ 언어 지원 (커널 통해)

### 3. 리터러시 프로그래밍
- 코드와 설명을 함께 작성하여 재현 가능한 연구 가능
- 순차적인 실행 흐름으로 데이터 분석 과정 추적

## 설치 방법

### 1. pip로 설치
```bash
pip install jupyter
```

### 2. conda로 설치
```bash
conda install -c conda-forge jupyter
```

### 3. uv로 설치 (권장)
```bash
uv add jupyter
```

## 기본 사용법

### 1. Jupyter Notebook 시작
```bash
jupyter notebook
```

### 2. Jupyter Lab 시작 (더 현대적인 인터페이스)
```bash
jupyter lab
```

### 3. 주요 단축키
- **Ctrl + Enter**: 셀 실행
- **Shift + Enter**: 셀 실행 후 다음 셀로 이동
- **Alt + Enter**: 셀 실행 후 새 셀 삽입
- **M**: 마크다운 셀로 변환
- **Y**: 코드 셀로 변환
- **A**: 위에 새 셀 삽입
- **B**: 아래에 새 셀 삽입
- **D, D**: 셀 삭제

## 기본 셀 타입

### 1. Code 셀
```python
# Python 코드 실행
import pandas as pd
import numpy as np

# 데이터 생성
data = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35]
})

print(data)
```

### 2. Markdown 셀
```markdown
# 제목
## 부제목

- 리스트 항목 1
- 리스트 항목 2

**굵은 글씨**와 *기울임꼴*

`코드` 강조
```

### 3. Raw 셀
- 포맷팅 없는 텍스트
- 주로 메모나 주석으로 사용

## 유용한 매직 명령어

### 1. %로 시작하는 라인 매직
```python
%matplotlib inline  # 그래프를 노트북 내에 표시
%load_ext autoreload  # 모듈 자동 리로드
%time  # 코드 실행 시간 측정
%whos  # 현재 변수 목록 표시
```

### 2. %%로 시작하는 셀 매직
```python
%%time  # 셀 전체 실행 시간 측정
%%bash  # bash 명령어 실행
%%html  # HTML 렌더링
```

## 데이터 과학을 위한 필수 라이브러리

### 1. 데이터 조작
```python
import pandas as pd  # 데이터프레임 조작
import numpy as np   # 수치 계산
```

### 2. 시각화
```python
import matplotlib.pyplot as plt  # 기본 시각화
import seaborn as sns           # 통계 시각화
import plotly.express as px     # 인터랙티브 시각화
```

### 3. 머신러닝
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
```

## 노트북 관리

### 1. 노트북 저장
- **Ctrl + S**: 수동 저장
- 자동 저장 기능 (기본 2분 간격)

### 2. 노트북 내보내기
- **File → Download as**: 다양한 형식으로 내보내기
  - HTML (.html)
  - PDF (.pdf)
  - Python (.py)
  - Markdown (.md)

### 3. 커널 관리
- **Kernel → Restart**: 커널 재시작
- **Kernel → Interrupt**: 실행 중인 코드 중단
- **Kernel → Change kernel**: 다른 커널로 전환

## 버전 관리

### 1. Git과의 통합
- `.ipynb_checkpoints` 폴더는 `.gitignore`에 추가
- 노트북은 JSON 형식이라 diff가 어려움

### 2. 노트북 정리
```bash
# 노트북 정리 도구 설치
pip install nbstripout

# 출력 결과 제거
nbstripout notebook.ipynb
```

## 모범 사례

### 1. 코드 구성
- 하나의 노트북은 하나의 분석 작업에 집중
- 셀은 논리적으로 분리하고 명확한 주석 추가

### 2. 성능 최적화
- 큰 데이터셋은 청크로 처리
- 불필요한 변수는 `del`로 메모리에서 제거
- `%time`으로 성능 병목 확인

### 3. 재현성 보장
- 라이브러리 버전 고정 (`requirements.txt`)
- 랜덤 시드 설정
- 데이터 소스 명시

## Jupyter Lab vs Jupyter Notebook

| 기능 | Jupyter Notebook | Jupyter Lab |
|------|------------------|-------------|
| 인터페이스 | 단일 문서 | 다중 문서, 파일 브라우저 |
| 확장성 | 제한적 | 높음 (플러그인 지원) |
| 레이아웃 | 고정 | 유연 (드래그 앤 드롭) |
| 터미널 | 미지원 | 내장 터미널 지원 |

## 엑셀 파일 다루기

### 1. 엑셀 파일 읽기
```python
import pandas as pd

# 엑셀 파일 읽기
df = pd.read_excel('data.xlsx')

# 특정 시트 읽기
df_sheet = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# 여러 시트 한번에 읽기
all_sheets = pd.read_excel('data.xlsx', sheet_name=['Sheet1', 'Sheet2'])
```

### 2. 엑셀 파일 쓰기
```python
# 데이터프레임을 엑셀로 저장
df.to_excel('output.xlsx', index=False)

# 여러 시트에 저장
with pd.ExcelWriter('multi_sheet.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1', index=False)
    df2.to_excel(writer, sheet_name='Sheet2', index=False)
```

### 3. 엑셀 파일 스타일링
```python
# openpyxl 설치 필요
pip install openpyxl

# 엑셀 파일 스타일 적용
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill

# 엑셀 파일 로드
book = load_workbook('output.xlsx')
sheet = book.active

# 스타일 적용
sheet['A1'].font = Font(bold=True, color='FF0000')
sheet['A1'].fill = PatternFill(start_color='FFFF00', end_color='FFFF00', fill_type='solid')

book.save('styled_output.xlsx')
```

### 4. 대용량 엑셀 파일 처리
```python
# chunk로 읽기 (메모리 절약)
chunk_size = 10000
chunks = pd.read_excel('large_file.xlsx', chunksize=chunk_size)

for chunk in chunks:
    # 각 청크 처리
    process_data(chunk)
```

### 5. 엑셀 수식 다루기
```python
# 엑셀 수식이 있는 파일 읽기
df = pd.read_excel('formulas.xlsx', engine='openpyxl')

# 수식 유지하면서 저장
df.to_excel('output_formulas.xlsx', engine='openpyxl')
```

## 추천 확장 프로그램

### 1. Jupyter Lab 확장
```bash
# JupyterLab 확장 관리자 설치
pip install jupyterlab

# 유용한 확장들
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install jupyterlab-plotly
```

### 2. Nbextensions (Notebook)
```bash
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
```

이 기본 정보들을 바탕으로 Jupyter Notebook을 효과적으로 사용하여 데이터 분석, 머신러닝, 시각화 등의 작업을 수행할 수 있습니다.
