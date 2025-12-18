# API 활용

## API 기본 개념
API의 정의, 작동 원리 및 RESTful API의 구조를 이해합니다.

**설명:**
API(Application Programming Interface)는 소프트웨어 애플리케이션들이 서로 통신하기 위한 규칙의 집합입니다. RESTful API는 웹에서 널리 사용되는 아키텍처 스타일로, HTTP 메서드(GET, POST, PUT, DELETE 등)를 사용해 자원(Resource)을 관리합니다.

## LLM API 사용법
OpenAI API와 같은 LLM API를 활용하여 텍스트 생성 및 데이터 처리 기능을 구현하는 방법을 배웁니다.

**예제 (requests 라이브러리 사용):**
```python
import requests
import json

# 가상의 LLM API 엔드포인트와 API 키
# API_URL = "https://api.some-llm.com/v1/generate"
# API_KEY = "YOUR_API_KEY"

# headers = {
#     "Authorization": f"Bearer {API_KEY}",
#     "Content-Type": "application/json"
# }

# data = {
#     "prompt": "API를 사용하는 간단한 파이썬 코드 예제를 보여줘.",
#     "max_tokens": 100
# }

# response = requests.post(API_URL, headers=headers, data=json.dumps(data))

# if response.status_code == 200:
#     result = response.json()
#     print("\nLLM API 응답:")
#     # print(result['choices'][0]['text'])
# else:
#     print(f"API 요청 실패: {response.status_code}")

print("LLM API 예제는 실제 API 키와 엔드포인트가 필요합니다.")

```

## API 보안 및 관리
API 사용 시 고려해야 할 보안 문제와 효율적인 API 관리 방법을 논의합니다.

**설명:**
*   **인증(Authentication):** API 키, OAuth 2.0 등을 사용해 허가된 사용자만 API에 접근하도록 합니다.
*   **권한 부여(Authorization):** 사용자에게 필요한 최소한의 권한만 부여합니다.
*   **전송 데이터 암호화:** HTTPS를 사용해 API 요청과 응답 데이터를 암호화합니다.
*   **사용량 제한(Rate Limiting):** 특정 시간 동안의 API 호출 횟수를 제한하여 서비스 남용을 방지합니다.

```python
# API 키를 코드에 직접 하드코딩하는 것은 위험합니다.
# BAD_PRACTICE
# api_key = "sk-xxxxxxxxxxxxxxxxxxxx"

# GOOD_PRACTICE: 환경 변수 사용
import os
# api_key = os.getenv("MY_LLM_API_KEY")
# print("\n환경 변수에서 API 키를 로드하는 것이 안전합니다.")
```
