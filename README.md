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
└── pyllm.md        # 메인 가이드 (인덱스)
```

## 라이선스

이 프로젝트는 학습 목적으로 제작되었습니다.
