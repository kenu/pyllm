# 클라우드와 배포

## 클라우드 컴퓨팅 기초
클라우드 서비스의 기본 개념과 종류를 이해합니다.

### 1. 클라우드 서비스 모델
```python
# 클라우드 서비스 모델 비교
import pandas as pd

cloud_models = {
    'IaaS': {
        'full_name': 'Infrastructure as a Service',
        'examples': ['AWS EC2', 'Google Compute Engine', 'Azure VM'],
        'responsibility': 'OS, 미들웨어, 런타임, 데이터, 애플리케이션',
        'flexibility': '높음',
        'management': '사용자'
    },
    'PaaS': {
        'full_name': 'Platform as a Service',
        'examples': ['Heroku', 'Google App Engine', 'Azure App Service'],
        'responsibility': '데이터, 애플리케이션',
        'flexibility': '중간',
        'management': '공유'
    },
    'SaaS': {
        'full_name': 'Software as a Service',
        'examples': ['Google Workspace', 'Microsoft 365', 'Salesforce'],
        'responsibility': '없음',
        'flexibility': '낮음',
        'management': '제공업체'
    },
    'FaaS': {
        'full_name': 'Function as a Service',
        'examples': ['AWS Lambda', 'Google Cloud Functions', 'Azure Functions'],
        'responsibility': '함수 코드',
        'flexibility': '매우 높음',
        'management': '제공업체'
    }
}

df_cloud = pd.DataFrame(cloud_models).T
print("=== 클라우드 서비스 모델 ===")
print(df_cloud)

# 클라우드 배포 전략
deployment_strategies = {
    '전통 배포': {
        '설명': '서버에 직접 애플리케이션 설치',
        '장점': '완전한 제어권',
        '단점': '확장성 부족, 수동 관리',
        '적합한 경우': '소규모 프로젝트, 특정 요구사항'
    },
    '컨테이너 배포': {
        '설명': 'Docker 컨테이너로 패키징하여 배포',
        '장점': '일관성, 이식성, 확장성',
        '단점': '학습 곡선, 오버헤드',
        '적합한 경우': '마이크로서비스, DevOps'
    },
    '서버리스': {
        '설명': '함수 단위로 배포, 서버 관리 불필요',
        '장점': '비용 효율, 자동 확장',
        '단점': '실행 시간 제한, 콜드 스타트',
        '적합한 경우': '이벤트 기반, 간헐적 작업'
    },
    'PaaS': {
        '설명': '플랫폼에 직접 배포',
        '장점': '간편함, 자동 확장',
        '단점': '벤더 종속성, 제어권 제한',
        '적합한 경우': '빠른 개발, 스타트업'
    }
}

df_strategies = pd.DataFrame(deployment_strategies).T
print("\n=== 배포 전략 비교 ===")
print(df_strategies)
```

## Docker 컨테이너화
애플리케이션을 Docker 컨테이너로 패키징합니다.

### 1. Dockerfile 작성
```python
# Dockerfile 예제 생성
dockerfile_example = '''
# Python 애플리케이션 Dockerfile
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# 의존성 파일 복사
COPY requirements.txt .

# Python 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 포트 노출
EXPOSE 8000

# 환경 변수
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 실행 명령
CMD ["python", "main.py"]
'''

# requirements.txt 예제
requirements_example = '''
fastapi==0.104.1
uvicorn==0.24.0
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
matplotlib==3.7.2
seaborn==0.12.2
'''

# docker-compose.yml 예제
docker_compose_example = '''
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/mydb
    depends_on:
      - db
    volumes:
      - ./data:/app/data

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=mydb
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
'''

print("=== Docker 설정 파일 예제 ===")
print("Dockerfile:")
print(dockerfile_example)
print("\nrequirements.txt:")
print(requirements_example)
print("\ndocker-compose.yml:")
print(docker_compose_example)
```

### 2. Docker 명령어 및 관리
```python
# Docker 명령어 가이드
docker_commands = {
    '이미지 빌드': 'docker build -t myapp:latest .',
    '컨테이너 실행': 'docker run -p 8000:8000 myapp:latest',
    '백그라운드 실행': 'docker run -d -p 8000:8000 --name myapp_container myapp:latest',
    '컨테이너 목록': 'docker ps',
    '모든 컨테이너': 'docker ps -a',
    '로그 확인': 'docker logs myapp_container',
    '컨테이너 접속': 'docker exec -it myapp_container bash',
    '이미지 목록': 'docker images',
    '이미지 삭제': 'docker rmi myapp:latest',
    '컨테이너 삭제': 'docker rm myapp_container',
    '볼륨 목록': 'docker volume ls',
    '네트워크 목록': 'docker network ls'
}

print("=== Docker 기본 명령어 ===")
for cmd, desc in docker_commands.items():
    print(f"{cmd}: {desc}")

# Docker Compose 명령어
compose_commands = {
    '서비스 시작': 'docker-compose up -d',
    '서비스 중지': 'docker-compose down',
    '서비스 재시작': 'docker-compose restart',
    '로그 확인': 'docker-compose logs -f',
    '서비스 빌드': 'docker-compose build',
    '이미지 재빌드': 'docker-compose up --build',
    '서비스 스케일링': 'docker-compose up --scale web=3'
}

print("\n=== Docker Compose 명령어 ===")
for cmd, desc in compose_commands.items():
    print(f"{cmd}: {desc}")
```

## AWS 클라우드 배포
AWS 서비스를 활용한 클라우드 배포를 구현합니다.

### 1. AWS EC2 배포
```python
# AWS EC2 배포 스크립트 예제
ec2_deploy_script = '''
#!/bin/bash

# EC2 인스턴스 설정 스크립트

# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# Python 및 필요한 패키지 설치
sudo apt install -y python3 python3-pip python3-venv nginx

# 프로젝트 디렉토리 생성
sudo mkdir -p /var/www/myapp
sudo chown $USER:$USER /var/www/myapp

# 가상환경 생성 및 활성화
cd /var/www/myapp
python3 -m venv venv
source venv/bin/activate

# Git에서 코드 클론
git clone https://github.com/username/myapp.git .

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
echo "DATABASE_URL=postgresql://user:password@localhost/mydb" >> .env
echo "SECRET_KEY=your-secret-key" >> .env

# Systemd 서비스 파일 생성
sudo tee /etc/systemd/system/myapp.service > /dev/null <<EOF
[Unit]
Description=My Application
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/myapp
Environment=PATH=/var/www/myapp/venv/bin
ExecStart=/var/www/myapp/venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# 서비스 활성화 및 시작
sudo systemctl daemon-reload
sudo systemctl enable myapp
sudo systemctl start myapp

# Nginx 설정
sudo tee /etc/nginx/sites-available/myapp > /dev/null <<EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# 사이트 활성화
sudo ln -s /etc/nginx/sites-available/myapp /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

echo "배포 완료!"
'''

print("=== EC2 배포 스크립트 ===")
print(ec2_deploy_script)
```

### 2. AWS Lambda 서버리스 배포
```python
# Lambda 함수 예제
lambda_function = '''
import json
import boto3
import pandas as pd

def lambda_handler(event, context):
    """
    Lambda 핸들러 함수
    """
    try:
        # 이벤트 데이터 추출
        http_method = event.get('httpMethod', 'GET')
        path = event.get('path', '/')
        
        if http_method == 'GET' and path == '/health':
            return {
                'statusCode': 200,
                'body': json.dumps({'status': 'healthy'})
            }
        
        elif http_method == 'POST' and path == '/process':
            # 데이터 처리
            body = json.loads(event.get('body', '{}'))
            data = body.get('data', [])
            
            # Pandas로 데이터 처리
            df = pd.DataFrame(data)
            result = {
                'count': len(df),
                'mean': df.mean().to_dict() if not df.empty else {},
                'processed_at': pd.Timestamp.now().isoformat()
            }
            
            return {
                'statusCode': 200,
                'body': json.dumps(result)
            }
        
        else:
            return {
                'statusCode': 404,
                'body': json.dumps({'error': 'Not found'})
            }
            
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
'''

# Lambda 배포 스크립트
lambda_deploy_script = '''
#!/bin/bash

# Lambda 함수 배포 스크립트

# 변수 설정
FUNCTION_NAME="myapp-function"
ROLE_ARN="arn:aws:iam::123456789012:role/lambda-execution-role"
HANDLER="lambda_function.lambda_handler"
RUNTIME="python3.11"

# 배포 패키지 생성
zip -r deployment.zip lambda_function.py requirements.txt

# Lambda 함수 생성 또는 업데이트
aws lambda get-function --function-name $FUNCTION_NAME > /dev/null 2>&1

if [ $? -eq 0 ]; then
    # 함수 업데이트
    aws lambda update-function-code \\
        --function-name $FUNCTION_NAME \\
        --zip-file fileb://deployment.zip
    echo "Lambda 함수 업데이트 완료"
else
    # 함수 생성
    aws lambda create-function \\
        --function-name $FUNCTION_NAME \\
        --runtime $RUNTIME \\
        --role $ROLE_ARN \\
        --handler $HANDLER \\
        --zip-file fileb://deployment.zip \\
        --timeout 30 \\
        --memory-size 256
    echo "Lambda 함수 생성 완료"
fi

# API Gateway 연결 (선택사항)
aws lambda add-permission \\
    --function-name $FUNCTION_NAME \\
    --action lambda:InvokeFunction \\
    --principal apigateway.amazonaws.com \\
    --statement-id apigateway-invoke

echo "배포 완료!"
'''

print("=== Lambda 함수 예제 ===")
print(lambda_function)
print("\n=== Lambda 배포 스크립트 ===")
print(lambda_deploy_script)
```

## Google Cloud Platform 배포
GCP 서비스를 활용한 클라우드 배포를 구현합니다.

### 1. Google Cloud Functions
```python
# Cloud Functions 예제
cloud_function = '''
import functions_framework
import pandas as pd
from google.cloud import storage

@functions_framework.http
def process_data(request):
    """
    HTTP 트리거 Cloud Function
    """
    # 요청 파라미터 추출
    bucket_name = request.args.get('bucket')
    file_name = request.args.get('file')
    
    if not bucket_name or not file_name:
        return {'error': 'bucket and file parameters required'}, 400
    
    try:
        # Cloud Storage에서 데이터 읽기
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        
        # CSV 데이터 처리
        data = blob.download_as_text()
        df = pd.read_csv(pd.StringIO(data))
        
        # 데이터 분석
        analysis = {
            'rows': len(df),
            'columns': len(df.columns),
            'summary': df.describe().to_dict(),
            'processed_at': pd.Timestamp.now().isoformat()
        }
        
        return analysis, 200
        
    except Exception as e:
        return {'error': str(e)}, 500

@functions_framework.cloud_event
def process_pubsub(cloud_event):
    """
    Pub/Sub 트리거 Cloud Function
    """
    # 이벤트 데이터 추출
    data = cloud_event.data
    
    # 메시지 처리
    message = base64.b64decode(data).decode('utf-8')
    message_data = json.loads(message)
    
    # 데이터 처리 로직
    print(f"Processing message: {message_data}")
    
    # 처리 결과
    result = {
        'message_id': cloud_event.id,
        'processed_at': pd.Timestamp.now().isoformat(),
        'status': 'processed'
    }
    
    return result
'''

print("=== Google Cloud Functions 예제 ===")
print(cloud_function)
```

### 2. Google App Engine 배포
```python
# app.yaml 설정 파일
app_yaml = '''
runtime: python311

instance_class: F2

# 자동 확장 설정
automatic_scaling:
  min_instances: 1
  max_instances: 10
  cpu_utilization:
    target_utilization: 0.75

# 환경 변수
env_variables:
  DATABASE_URL: postgresql://user:password@localhost/dbname
  SECRET_KEY: your-secret-key

# 핸들러
handlers:
- url: /static
  static_dir: static

- url: /.*
  script: auto

# 라이브러리
libraries:
- name: pandas
  version: "2.1.3"
- name: numpy
  version: "1.24.3"
'''

# main.py App Engine 예제
app_engine_main = '''
from flask import Flask, request, jsonify
import pandas as pd
import os

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({'message': 'Welcome to my App Engine app!'})

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    try:
        data = request.get_json()
        df = pd.DataFrame(data.get('data', []))
        
        if df.empty:
            return jsonify({'error': 'No data provided'}), 400
        
        analysis = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'summary': df.describe().to_dict()
        }
        
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
'''

print("=== App Engine 설정 ===")
print("app.yaml:")
print(app_yaml)
print("\nmain.py:")
print(app_engine_main)
```

## CI/CD 파이프라인
자동화된 배포 파이프라인을 구축합니다.

### 1. GitHub Actions
```python
# GitHub Actions 워크플로우
github_actions_workflow = '''
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest --cov=. --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: |
          username/myapp:latest
          username/myapp:${{ github.sha }}
    
    - name: Deploy to production
      run: |
        echo "Deploying to production..."
        # 실제 배포 스크립트
        # ssh user@server 'docker pull username/myapp:latest && docker-compose up -d'
'''

print("=== GitHub Actions CI/CD 파이프라인 ===")
print(github_actions_workflow)
```

### 2. Jenkins 파이프라인
```python
# Jenkinsfile 예제
jenkins_pipeline = '''
pipeline {
    agent any
    
    environment {
        DOCKER_REGISTRY = 'your-registry.com'
        DOCKER_CREDENTIALS = credentials('docker-creds')
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Setup Environment') {
            steps {
                sh '''
                    python -m venv venv
                    . venv/bin/activate
                    pip install -r requirements.txt
                '''
            }
        }
        
        stage('Run Tests') {
            steps {
                sh '''
                    . venv/bin/activate
                    pytest --cov=. --junitxml=test-results.xml
                '''
            }
            post {
                always {
                    publishTestResults testResultsPattern: 'test-results.xml'
                    publishCoverage adapters: [coberturaAdapter('coverage.xml')]
                }
            }
        }
        
        stage('Build Docker Image') {
            steps {
                script {
                    docker_image = docker.build(
                        "myapp:${env.BUILD_NUMBER}",
                        "."
                    )
                }
            }
        }
        
        stage('Push to Registry') {
            steps {
                script {
                    docker.withRegistry(
                        "https://${DOCKER_REGISTRY}",
                        "${DOCKER_CREDENTIALS}"
                    ) {
                        docker_image.push()
                        docker_image.push('latest')
                    }
                }
            }
        }
        
        stage('Deploy to Staging') {
            steps {
                sh '''
                    # 스테이징 환경 배포
                    kubectl set image deployment/myapp-staging \\
                        myapp=${DOCKER_REGISTRY}/myapp:${BUILD_NUMBER} \\
                        --namespace=staging
                    
                    # 롤아웃 확인
                    kubectl rollout status deployment/myapp-staging \\
                        --namespace=staging
                '''
            }
        }
        
        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            input {
                message "Deploy to production?"
            }
            steps {
                sh '''
                    # 프로덕션 환경 배포
                    kubectl set image deployment/myapp-production \\
                        myapp=${DOCKER_REGISTRY}/myapp:${BUILD_NUMBER} \\
                        --namespace=production
                    
                    # 롤아웃 확인
                    kubectl rollout status deployment/myapp-production \\
                        --namespace=production
                '''
            }
        }
    }
    
    post {
        success {
            slackSend(
                color: 'good',
                message: "✅ Pipeline succeeded for ${env.JOB_NAME} - ${env.BUILD_NUMBER}"
            )
        }
        failure {
            slackSend(
                color: 'danger',
                message: "❌ Pipeline failed for ${env.JOB_NAME} - ${env.BUILD_NUMBER}"
            )
        }
    }
}
'''

print("=== Jenkins 파이프라인 ===")
print(jenkins_pipeline)
```

## 모니터링 및 로깅
배포된 애플리케이션을 모니터링하고 로그를 관리합니다.

### 1. Prometheus와 Grafana
```python
# Prometheus 설정
prometheus_config = '''
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'myapp'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
'''

# Grafana 대시보드 설정
grafana_dashboard = '''
{
  "dashboard": {
    "title": "My Application Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m])",
            "legendFormat": "Error Rate"
          }
        ]
      }
    ]
  }
}
'''

print("=== 모니터링 설정 ===")
print("Prometheus 설정:")
print(prometheus_config)
print("\nGrafana 대시보드:")
print(grafana_dashboard)
```

### 2. 애플리케이션 모니터링
```python
# 모니터링 데코레이터
import time
import functools
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# 메트릭 정의
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active connections')

def monitor_performance(func):
    """성능 모니터링 데코레이터"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            REQUEST_COUNT.labels(method='GET', status='200').inc()
            return result
        except Exception as e:
            REQUEST_COUNT.labels(method='GET', status='500').inc()
            raise
        finally:
            REQUEST_DURATION.observe(time.time() - start_time)
    
    return wrapper

# 모니터링 미들웨어 예제
class MonitoringMiddleware:
    def __init__(self, app):
        self.app = app
    
    def __call__(self, environ, start_response):
        start_time = time.time()
        
        def custom_start_response(status, headers):
            # 메트릭 기록
            REQUEST_COUNT.labels(
                method=environ.get('REQUEST_METHOD', 'GET'),
                status=status.split(' ')[0]
            ).inc()
            
            REQUEST_DURATION.observe(time.time() - start_time)
            
            return start_response(status, headers)
        
        return self.app(environ, custom_start_response)

# 로깅 설정
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log(self, level, message, **kwargs):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'message': message,
            **kwargs
        }
        
        self.logger.info(json.dumps(log_data))
    
    def info(self, message, **kwargs):
        self.log('INFO', message, **kwargs)
    
    def error(self, message, **kwargs):
        self.log('ERROR', message, **kwargs)

# 사용 예제
logger = StructuredLogger('myapp')

@monitor_performance
def process_data(data):
    """데이터 처리 함수"""
    logger.info('Processing data', data_count=len(data))
    
    # 데이터 처리 로직
    result = len(data) * 2
    
    logger.info('Data processed', result=result)
    return result

# 모니터링 서버 시작
if __name__ == "__main__":
    start_http_server(8001)  # Prometheus 메트릭 포트
    print("Monitoring server started on port 8001")
```

## 보안 및 최적화
배포된 애플리케이션의 보안을 강화하고 성능을 최적화합니다.

### 1. 보안 설정
```python
# 보안 미들웨어
from functools import wraps
import hashlib
import secrets
import time

class SecurityMiddleware:
    def __init__(self, app):
        self.app = app
        self.rate_limit = {}
    
    def __call__(self, environ, start_response):
        # 보안 헤더 추가
        def custom_start_response(status, headers):
            security_headers = [
                ('X-Content-Type-Options', 'nosniff'),
                ('X-Frame-Options', 'DENY'),
                ('X-XSS-Protection', '1; mode=block'),
                ('Strict-Transport-Security', 'max-age=31536000; includeSubDomains'),
                ('Content-Security-Policy', "default-src 'self'")
            ]
            
            headers.extend(security_headers)
            return start_response(status, headers)
        
        # 속도 제한
        client_ip = environ.get('REMOTE_ADDR', 'unknown')
        if not self.check_rate_limit(client_ip):
            start_response('429 Too Many Requests', [])
            return [b'Rate limit exceeded']
        
        return self.app(environ, custom_start_response)
    
    def check_rate_limit(self, client_ip, limit=100, window=3600):
        """속도 제한 확인"""
        now = time.time()
        
        if client_ip not in self.rate_limit:
            self.rate_limit[client_ip] = []
        
        # 오래된 요청 제거
        self.rate_limit[client_ip] = [
            req_time for req_time in self.rate_limit[client_ip]
            if now - req_time < window
        ]
        
        # 제한 확인
        if len(self.rate_limit[client_ip]) >= limit:
            return False
        
        self.rate_limit[client_ip].append(now)
        return True

# API 키 인증
def require_api_key(func):
    """API 키 인증 데코레이터"""
    @wraps(func)
    def wrapper(request, *args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key or not validate_api_key(api_key):
            return {'error': 'Invalid API key'}, 401
        
        return func(request, *args, **kwargs)
    
    return wrapper

def validate_api_key(api_key):
    """API 키 검증"""
    # 실제로는 데이터베이스에서 확인
    valid_keys = ['key1', 'key2', 'key3']
    return api_key in valid_keys

# 데이터 암호화
from cryptography.fernet import Fernet

class DataEncryption:
    def __init__(self, key=None):
        if key is None:
            key = Fernet.generate_key()
        self.cipher = Fernet(key)
    
    def encrypt(self, data):
        """데이터 암호화"""
        if isinstance(data, str):
            data = data.encode()
        return self.cipher.encrypt(data)
    
    def decrypt(self, encrypted_data):
        """데이터 복호화"""
        decrypted = self.cipher.decrypt(encrypted_data)
        return decrypted.decode()

print("=== 보안 설정 예제 ===")
print("보안 미들웨어, API 인증, 데이터 암호화 구현 완료")
```

### 2. 성능 최적화
```python
# 캐싱 시스템
import redis
import pickle
from functools import wraps

class CacheManager:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
    
    def cache_result(self, key_prefix, expire_time=3600):
        """결과 캐싱 데코레이터"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 캐시 키 생성
                cache_key = f"{key_prefix}:{hash(str(args) + str(kwargs))}"
                
                # 캐시 확인
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    return pickle.loads(cached_result)
                
                # 함수 실행
                result = func(*args, **kwargs)
                
                # 결과 캐싱
                self.redis_client.setex(
                    cache_key,
                    expire_time,
                    pickle.dumps(result)
                )
                
                return result
            
            return wrapper
        return decorator
    
    def invalidate_cache(self, pattern):
        """캐시 무효화"""
        keys = self.redis_client.keys(pattern)
        if keys:
            self.redis_client.delete(*keys)

# 데이터베이스 연결 풀
import psycopg2
from psycopg2 import pool

class DatabasePool:
    def __init__(self, min_conn=1, max_conn=10, **db_config):
        self.pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=min_conn,
            maxconn=max_conn,
            **db_config
        )
    
    def get_connection(self):
        return self.pool.getconn()
    
    def release_connection(self, conn):
        self.pool.putconn(conn)
    
    def execute_query(self, query, params=None):
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            if query.strip().upper().startswith('SELECT'):
                result = cursor.fetchall()
                conn.commit()
                return result
            else:
                conn.commit()
                return cursor.rowcount
                
        finally:
            self.release_connection(conn)

# 비동기 처리
import asyncio
import aiohttp

class AsyncDataProcessor:
    def __init__(self, max_concurrent=10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_async(self, data_list):
        """비동기 데이터 처리"""
        tasks = []
        
        for data in data_list:
            task = self.process_single_data(data)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
    
    async def process_single_data(self, data):
        """단일 데이터 비동기 처리"""
        async with self.semaphore:
            # 비동기 처리 로직
            await asyncio.sleep(0.1)  # 시뮬레이션
            return len(data)

# 사용 예제
cache_manager = CacheManager()

@cache_manager.cache_result('data_analysis', expire_time=1800)
def analyze_data(data):
    """데이터 분석 (캐싱 적용)"""
    # 시간이 오래 걸리는 분석 작업
    time.sleep(1)
    return {'count': len(data), 'mean': sum(data) / len(data)}

print("=== 성능 최적화 예제 ===")
print("캐싱, 연결 풀, 비동기 처리 구현 완료")

print("\n=== 클라우드와 배포 예제 완료! ===")
print("1. 클라우드 컴퓨팅 기초")
print("2. Docker 컨테이너화")
print("3. AWS/GCP 배포")
print("4. CI/CD 파이프라인")
print("5. 모니터링 및 로깅")
print("6. 보안 및 성능 최적화")
