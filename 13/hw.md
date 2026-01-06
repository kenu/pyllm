# Homework - 클라우드 배포

## 이 homework의 의미
로컬에서 개발한 애플리케이션을 클라우드에 배포하여 실제 서비스로 운영하는 과제야. AWS, GCP, Azure 등 클라우드 플랫폼을 활용해 확장 가능한 인프라를 구축하는 거지.

## 관련 정보 사이트
1. **AWS Documentation**: https://docs.aws.amazon.com/
2. **Google Cloud Platform**: https://cloud.google.com/docs

## 프로세스 진행 논리적 흐름
1. **클라우드 선택** → AWS, GCP, Azure 중 선택
2. **컨테이너화** → Docker 이미지 생성
3. **배포 설정** → EC2, Cloud Run, App Service 등
4. **CI/CD 구축** → GitHub Actions, Jenkins 연동
5. **모니터링** → 로그 수집 및 성능 추적

## 권장 코드
```python
# Dockerfile 예제
"""
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# FastAPI 애플리케이션 예제
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="My API")

@app.get("/")
def read_root():
    return {"message": "Hello from Cloud!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/predict")
def predict(data: str):
    # ML 모델 예측 로직
    result = f"Prediction for: {data}"
    return {"result": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/mydb
    depends_on:
      - db
  
  db:
    image: postgres:15
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: mydb
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

## 관련 업계 회사
1. **AWS (Amazon)** - 클라우드 컴퓨팅 시장 선두
2. **Google Cloud Platform** - AI/ML 특화 클라우드 서비스
3. **Microsoft Azure** - 엔터프라이즈 클라우드 솔루션
