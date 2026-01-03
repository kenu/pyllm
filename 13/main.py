import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
import hashlib
import secrets

print("=== 클라우드와 배포 예제 ===")

# 1. 클라우드 서비스 모델 비교
print("\n=== 1. 클라우드 서비스 모델 ===")

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
print("클라우드 서비스 모델:")
print(df_cloud)

# 2. 배포 전략 비교
print("\n=== 2. 배포 전략 비교 ===")

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
print("배포 전략:")
print(df_strategies)

# 3. Docker 설정 파일 생성
print("\n=== 3. Docker 설정 ===")

# Dockerfile 예제
dockerfile_content = '''# Python 애플리케이션 Dockerfile
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
CMD ["python", "main.py"]'''

# requirements.txt 예제
requirements_content = '''fastapi==0.104.1
uvicorn==0.24.0
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
matplotlib==3.7.2
seaborn==0.12.2'''

# docker-compose.yml 예제
docker_compose_content = '''version: '3.8'

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
  redis_data:'''

print("Dockerfile:")
print(dockerfile_content)
print("\nrequirements.txt:")
print(requirements_content)
print("\ndocker-compose.yml:")
print(docker_compose_content)

# 4. Docker 명령어 가이드
print("\n=== 4. Docker 기본 명령어 ===")

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

for cmd, desc in docker_commands.items():
    print(f"{cmd}: {desc}")

# 5. AWS Lambda 함수 시뮬레이션
print("\n=== 5. AWS Lambda 함수 시뮬레이션 ===")

class LambdaSimulator:
    def __init__(self):
        self.functions = {}
    
    def create_function(self, name, handler, runtime='python3.11'):
        """Lambda 함수 생성"""
        self.functions[name] = {
            'handler': handler,
            'runtime': runtime,
            'invocations': 0
        }
        print(f"Lambda 함수 생성: {name}")
    
    def invoke_function(self, name, event):
        """Lambda 함수 호출"""
        if name not in self.functions:
            return {'statusCode': 404, 'body': json.dumps({'error': 'Function not found'})}
        
        func = self.functions[name]
        self.functions[name]['invocations'] += 1
        
        try:
            result = func['handler'](event, None)
            return result
        except Exception as e:
            return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}
    
    def get_metrics(self, name):
        """함수 메트릭"""
        if name in self.functions:
            return {
                'invocations': self.functions[name]['invocations'],
                'runtime': self.functions[name]['runtime']
            }
        return None

# Lambda 핸들러 함수
def lambda_handler(event, context):
    """Lambda 핸들러 함수"""
    try:
        http_method = event.get('httpMethod', 'GET')
        path = event.get('path', '/')
        
        if http_method == 'GET' and path == '/health':
            return {
                'statusCode': 200,
                'body': json.dumps({'status': 'healthy'})
            }
        
        elif http_method == 'POST' and path == '/process':
            body = json.loads(event.get('body', '{}'))
            data = body.get('data', [])
            
            df = pd.DataFrame(data)
            result = {
                'count': len(df),
                'mean': df.mean().to_dict() if not df.empty else {},
                'processed_at': datetime.now().isoformat()
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

# Lambda 시뮬레이션 실행
lambda_sim = LambdaSimulator()
lambda_sim.create_function('myapp-function', lambda_handler)

# 함수 호출 테스트
health_event = {'httpMethod': 'GET', 'path': '/health'}
health_result = lambda_sim.invoke_function('myapp-function', health_event)
print("Health check 결과:")
print(json.loads(health_result['body']))

process_event = {
    'httpMethod': 'POST',
    'path': '/process',
    'body': json.dumps({'data': [{'value': 10}, {'value': 20}, {'value': 30}]})
}
process_result = lambda_sim.invoke_function('myapp-function', process_event)
print("\n데이터 처리 결과:")
print(json.loads(process_result['body']))

# 6. CI/CD 파이프라인 시뮬레이션
print("\n=== 6. CI/CD 파이프라인 시뮬레이션 ===")

class CICDPipeline:
    def __init__(self):
        self.stages = []
        self.current_stage = 0
        self.pipeline_status = 'pending'
    
    def add_stage(self, name, action):
        """파이프라인 스테이지 추가"""
        self.stages.append({'name': name, 'action': action, 'status': 'pending'})
    
    def run_pipeline(self):
        """파이프라인 실행"""
        self.pipeline_status = 'running'
        
        for i, stage in enumerate(self.stages):
            print(f"실행 중: {stage['name']}")
            
            try:
                stage['action']()
                stage['status'] = 'success'
                print(f"✅ {stage['name']} 완료")
            except Exception as e:
                stage['status'] = 'failed'
                print(f"❌ {stage['name']} 실패: {e}")
                self.pipeline_status = 'failed'
                break
        
        if self.pipeline_status != 'failed':
            self.pipeline_status = 'success'
            print("🎉 파이프라인 성공!")
    
    def get_status(self):
        """파이프라인 상태"""
        return {
            'pipeline_status': self.pipeline_status,
            'stages': self.stages
        }

# CI/CD 스테이지 정의
def test_stage():
    """테스트 스테이지"""
    # 테스트 시뮬레이션
    test_results = [True, True, True]  # 모든 테스트 통과
    if not all(test_results):
        raise Exception("테스트 실패")
    time.sleep(0.5)  # 테스트 실행 시간 시뮬레이션

def build_stage():
    """빌드 스테이지"""
    # 빌드 시뮬레이션
    print("  Docker 이미지 빌드 중...")
    time.sleep(1)
    print("  이미지 푸시 중...")
    time.sleep(0.5)

def deploy_stage():
    """배포 스테이지"""
    # 배포 시뮬레이션
    print("  스테이징 환경 배포 중...")
    time.sleep(0.8)
    print("  프로덕션 환경 배포 중...")
    time.sleep(1.2)

# CI/CD 파이프라인 실행
pipeline = CICDPipeline()
pipeline.add_stage('Test', test_stage)
pipeline.add_stage('Build', build_stage)
pipeline.add_stage('Deploy', deploy_stage)

print("CI/CD 파이프라인 실행:")
pipeline.run_pipeline()

# 7. 모니터링 시스템 시뮬레이션
print("\n=== 7. 모니터링 시스템 시뮬레이션 ===")

class MonitoringSystem:
    def __init__(self):
        self.metrics = {
            'requests_total': 0,
            'requests_success': 0,
            'requests_error': 0,
            'response_times': [],
            'active_connections': 0
        }
    
    def record_request(self, success=True, response_time=None):
        """요청 기록"""
        self.metrics['requests_total'] += 1
        
        if success:
            self.metrics['requests_success'] += 1
        else:
            self.metrics['requests_error'] += 1
        
        if response_time:
            self.metrics['response_times'].append(response_time)
    
    def get_metrics(self):
        """메트릭 조회"""
        total_requests = self.metrics['requests_total']
        success_rate = (self.metrics['requests_success'] / total_requests * 100) if total_requests > 0 else 0
        
        avg_response_time = (sum(self.metrics['response_times']) / len(self.metrics['response_times'])) if self.metrics['response_times'] else 0
        
        return {
            'total_requests': total_requests,
            'success_rate': success_rate,
            'avg_response_time': avg_response_time,
            'active_connections': self.metrics['active_connections']
        }
    
    def check_alerts(self):
        """알람 확인"""
        metrics = self.get_metrics()
        alerts = []
        
        if metrics['success_rate'] < 95:
            alerts.append(f"성공률 낮음: {metrics['success_rate']:.1f}%")
        
        if metrics['avg_response_time'] > 1.0:
            alerts.append(f"응답 시간 느림: {metrics['avg_response_time']:.2f}s")
        
        return alerts

# 모니터링 시스템 실행
monitor = MonitoringSystem()

# 모의 요청 기록
for i in range(100):
    success = np.random.random() > 0.05  # 95% 성공률
    response_time = np.random.uniform(0.1, 2.0)
    monitor.record_request(success, response_time)

# 메트릭 확인
current_metrics = monitor.get_metrics()
print("현재 메트릭:")
print(f"총 요청: {current_metrics['total_requests']}")
print(f"성공률: {current_metrics['success_rate']:.1f}%")
print(f"평균 응답 시간: {current_metrics['avg_response_time']:.2f}s")

# 알람 확인
alerts = monitor.check_alerts()
if alerts:
    print("\n⚠️ 알람:")
    for alert in alerts:
        print(f"- {alert}")
else:
    print("\n✅ 알람 없음")

# 8. 보안 설정 시뮬레이션
print("\n=== 8. 보안 설정 시뮬레이션 ===")

class SecurityManager:
    def __init__(self):
        self.api_keys = {
            'key1': 'user1',
            'key2': 'user2',
            'key3': 'user3'
        }
        self.rate_limit = {}
    
    def validate_api_key(self, api_key):
        """API 키 검증"""
        return api_key in self.api_keys
    
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
    
    def generate_secure_token(self):
        """보안 토큰 생성"""
        return secrets.token_urlsafe(32)
    
    def hash_password(self, password):
        """비밀번호 해싱"""
        return hashlib.sha256(password.encode()).hexdigest()

# 보안 관리자 테스트
security = SecurityManager()

# API 키 검증
test_key = 'key1'
is_valid = security.validate_api_key(test_key)
print(f"API 키 '{test_key}' 검증: {'✅ 유효' if is_valid else '❌ 무효'}")

# 속도 제한 테스트
client_ip = '192.168.1.100'
for i in range(5):
    allowed = security.check_rate_limit(client_ip, limit=10)
    print(f"요청 {i+1}: {'✅ 허용' if allowed else '❌ 제한'}")

# 보안 토큰 생성
token = security.generate_secure_token()
print(f"생성된 보안 토큰: {token[:20]}...")

# 비밀번호 해싱
password = "user123"
hashed = security.hash_password(password)
print(f"비밀번호 해시: {hashed}")

print("\n=== 클라우드와 배포 예제 완료! ===")
print("1. 클라우드 서비스 모델 비교")
print("2. Docker 컨테이너화 설정")
print("3. AWS Lambda 함수 시뮬레이션")
print("4. CI/CD 파이프라인 자동화")
print("5. 모니터링 시스템 구축")
print("6. 보안 설정 및 관리")
