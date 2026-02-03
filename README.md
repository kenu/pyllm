# PyLLM (Python for LLM)

PyLLM은 파이썬을 기반으로 데이터 분석, 머신러닝, 그리고 대형 언어 모델(LLM)을 활용하는 방법을 학습하기 위한 프로젝트입니다.

## 주요 학습 내용

이 프로젝트는 다음과 같은 주제들을 다룹니다:

1.  **Python 기초 및 활용**: 데이터 분석(Pandas, NumPy) 및 머신러닝(Scikit-learn) 기초
2.  **대형 언어 모델 (LLM)**: LLM의 구조, 작동 원리 및 실제 응용 사례
3.  **데이터베이스**: SQL(SQLite) 및 NoSQL(MongoDB) 활용과 LLM 통합
4.  **API 활용**: RESTful API 개념 및 LLM API(OpenAI 등) 연동 보안

## 개발 도구 및 생산성 향상

이 프로젝트에는 생산적인 학습을 위한 몇 가지 도구가 포함되어 있습니다:

- **Makefile**: 터미널에서 `make help`를 입력하여 사용 가능한 명령어를 확인하세요.
  - `make install`: 필요한 라이브러리 설치
  - `make run-latest`: 가장 최근에 수정한 파이썬 스크립트 실행
  - `make clean`: 임시 파일 및 캐시 삭제
- **환경 변수 관리**: `.env.example` 파일을 복사하여 `.env` 파일을 만드세요. API 키나 DB 접속 정보를 안전하게 관리할 수 있습니다.
- **자동 실행 스크립트**: `skills/recent_files.py`를 통해 최근 작업물을 빠르게 테스트할 수 있습니다.

### 📅 최근 한 달간 학습 요약 (2025.01 - 2025.02)

지난 한 달간 다음과 같은 실습과 학습을 진행했습니다:

- **생산성 및 도구**: CLI 도구 활용을 위한 `Makefile` 도입, API 키 관리를 위한 `.env` 환경 변수 설정 및 최근 작업 파일 자동 실행 스크립트 개발.
- **데이터 분석 및 시각화**: Pandas를 이용한 결측치 처리, Plotly 기반의 대화형 데이터 시각화, 통계학 기초 및 시계열 데이터 분석 실습.
- **데이터베이스 활용**: PyMySQL과 MongoDB를 활용한 SQL/NoSQL 연동, 실무적인 예외 처리 및 커넥션 풀 관리 기법 습득.
- **LLM API 연동**: OpenAI 및 Claude API 연동 시 발생할 수 있는 오류 대응 로직 구현 및 Claude Skills 활용법 학습.
- **학습 유틸리티**: 효율적인 학습 기록 관리를 위해 Markdown 문서를 Jupyter Notebook(`.ipynb`)으로 자동 변환하는 스크립트 제작.

## 시작하기

상세 학습 가이드는 아래 링크를 통해 확인하실 수 있습니다:

- [PyLLM 학습 가이드 (목차)](./pyllm/pyllm.md)

## 프로젝트 구조

```text
pyllm/
├── 01_python.md    # 파이썬 기초 및 데이터 분석
├── 02_llm.md       # LLM 구조 및 응용
├── 03_database.md  # 데이터베이스 및 LLM 통합
├── 04_api.md       # API 활용 및 보안
├── 05_timeseries.md  # 시계열 데이터 분석
├── 06_data_cleaning.md # 데이터 전처리
├── 07_visualization.md # 데이터 시각화
├── 08_interactive_dashboard.md # 대시보드
├── 09_machine_learning.md # 머신러닝
├── 10_deep_learning.md # 딥러닝
├── 11_database_advanced.md # DB 고급
├── 12_web_scraping_api.md # 웹 스크래핑
├── 13_cloud_deployment.md # 클라우드 배포
└── pyllm.md        # 메인 가이드 (인덱스)
```

## 라이선스

이 프로젝트는 학습 목적으로 제작되었습니다.
