# `uv`로 파이썬 환경 관리 및 패키지 설치

`uv`는 Rust로 작성된 매우 빠르고 효율적인 Python 패키지 인스톨러 및 의존성 분석기입니다. 기존의 `pip`, `venv`, `pip-tools`를 대체하여 Python 프로젝트의 의존성 관리를 간소화하고 속도를 향상시킵니다.

## `uv` 설치

`uv`는 독립 실행형 바이너리로 설치하거나 `pipx`를 통해 설치할 수 있습니다.

### 1. `pipx`를 사용하여 설치 (권장)

`pipx`는 Python 애플리케이션을 격리된 환경에 설치하고 관리하는 도구입니다.

```bash
# pipx 설치 (아직 설치되어 있지 않다면)
pip install pipx
pipx ensurepath

# uv 설치
pipx install uv
```

### 2. curl을 사용하여 독립 실행형 바이너리로 설치

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Homebrew를 사용하여 설치 (macOS, Linux)

```bash
brew install uv
```

## 가상 환경 생성 및 관리

`uv`는 `venv`와 유사하게 가상 환경을 생성하고 관리할 수 있습니다.

### 가상 환경 생성

특정 Python 인터프리터 버전을 사용하여 가상 환경을 생성할 수 있습니다.

```bash
# 현재 시스템의 기본 Python 버전을 사용하여 .venv 가상 환경 생성
uv venv

# 특정 Python 버전 (예: python3.9)을 사용하여 .venv 가상 환경 생성
# 시스템에 python3.9가 설치되어 있어야 합니다.
uv venv --python python3.9

# 다른 이름의 가상 환경 생성
uv venv my_project_venv --python python3.10
```

### 가상 환경 활성화

생성된 가상 환경을 활성화하는 방법은 기존 `venv`와 동일합니다.

```bash
# Linux / macOS
source .venv/bin/activate

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 가상 환경 비활성화

```bash
deactivate
```

## 패키지 설치

`uv`는 `pip`와 동일한 방식으로 `requirements.txt` 파일이나 직접 패키지 이름을 지정하여 패키지를 설치할 수 있습니다.

### 1. `requirements.txt` 파일로 패키지 설치

`requirements.txt` 파일에 필요한 패키지 목록이 있다고 가정합니다.

```requirements.txt
requests==2.31.0
pandas>=2.0.0,<3.0.0
numpy
```

```bash
# 가상 환경이 활성화된 상태에서 실행
uv pip install -r requirements.txt
```

### 2. 개별 패키지 설치

```bash
# 가상 환경이 활성화된 상태에서 실행
uv pip install black "django<5"
```

### 3. 패키지 업그레이드

```bash
# 모든 설치된 패키지 업그레이드
uv pip install --upgrade --all

# 특정 패키지 업그레이드
uv pip install --upgrade requests
```

### 4. 패키지 제거

```bash
uv pip uninstall requests
```

## 의존성 고정 (Locking)

`uv`는 `pip-tools`처럼 `requirements.in` 파일을 기반으로 `requirements.txt` (잠금 파일)를 생성할 수 있습니다.

```requirements.in
requests
django
```

```bash
# 의존성 해결 및 requirements.txt 파일 생성
uv pip compile requirements.in -o requirements.txt
```

이렇게 생성된 `requirements.txt` 파일은 프로젝트의 정확한 의존성 버전을 명시하므로, 다른 개발 환경에서도 동일한 의존성 트리를 보장할 수 있습니다.

## 기타 유용한 `uv` 명령어

*   **설치된 패키지 목록 보기:**
    ```bash
    uv pip list
    ```
*   **패키지 정보 보기:**
    ```bash
    uv pip show requests
    ```
*   **`uv` 버전 확인:**
    ```bash
    uv --version
    ```
*   **도움말:**
    ```bash
    uv --help
    uv pip --help
    ```

`uv`는 Python 개발 워크플로우를 크게 가속화하고 단순화할 수 있는 강력한 도구입니다.
