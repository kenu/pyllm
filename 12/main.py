import requests
import json
import pandas as pd
import time
from datetime import datetime
import re
from urllib.parse import urljoin
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print("=== 웹 스크래핑과 API 활용 예제 ===")

# 1. BeautifulSoup 시뮬레이션
print("\n=== 1. BeautifulSoup HTML 파싱 ===")

# 샘플 HTML
sample_html = """
<!DOCTYPE html>
<html>
<head>
    <title>뉴스 포털</title>
</head>
<body>
    <h1>오늘의 뉴스</h1>
    <div class="news-list">
        <article class="news-item">
            <h2 class="title">국회 의원회의 개최</h2>
            <p class="content">오늘 오후 2시 국회에서 중요 의안이 논의됩니다.</p>
            <div class="meta">
                <span class="author">김기자</span>
                <span class="date">2024-01-15</span>
                <span class="views">1,234</span>
            </div>
        </article>
        <article class="news-item">
            <h2 class="title">새로운 정책 발표</h2>
            <p class="content">정부가 새로운 경제 정책을 발표했습니다.</p>
            <div class="meta">
                <span class="author">이기자</span>
                <span class="date">2024-01-15</span>
                <span class="views">2,567</span>
            </div>
        </article>
    </div>
</body>
</html>
"""

# 간단한 HTML 파싱 함수
def parse_html_simple(html_content):
    """간단한 HTML 파싱 (BeautifulSoup 없이)"""
    news_data = []
    
    # 정규표현식으로 데이터 추출
    title_pattern = r'<h2 class="title">(.*?)</h2>'
    content_pattern = r'<p class="content">(.*?)</p>'
    author_pattern = r'<span class="author">(.*?)</span>'
    date_pattern = r'<span class="date">(.*?)</span>'
    views_pattern = r'<span class="views">(.*?)</span>'
    
    titles = re.findall(title_pattern, html_content)
    contents = re.findall(content_pattern, html_content)
    authors = re.findall(author_pattern, html_content)
    dates = re.findall(date_pattern, html_content)
    views = re.findall(views_pattern, html_content)
    
    # 데이터 조합
    for i in range(len(titles)):
        news_data.append({
            'title': titles[i].strip(),
            'content': contents[i].strip(),
            'author': authors[i].strip(),
            'date': dates[i].strip(),
            'views': int(views[i].replace(',', ''))
        })
    
    return news_data

# HTML 파싱 실행
news_data = parse_html_simple(sample_html)
df_news = pd.DataFrame(news_data)

print("추출된 뉴스 데이터:")
print(df_news)

# 2. API 연동 시뮬레이션
print("\n=== 2. REST API 연동 ===")

class MockAPIClient:
    """모의 API 클라이언트"""
    
    def __init__(self, base_url="https://api.example.com"):
        self.base_url = base_url
        self.mock_data = self._generate_mock_data()
    
    def _generate_mock_data(self):
        """모의 데이터 생성"""
        return {
            'users': [
                {'id': 1, 'name': '김철수', 'email': 'kim@example.com', 'department': '개발'},
                {'id': 2, 'name': '이영희', 'email': 'lee@example.com', 'department': '마케팅'},
                {'id': 3, 'name': '박민준', 'email': 'park@example.com', 'department': '영업'}
            ],
            'products': [
                {'id': 1, 'name': '노트북', 'price': 1200000, 'category': '전자제품'},
                {'id': 2, 'name': '마우스', 'price': 25000, 'category': '액세서리'},
                {'id': 3, 'name': '키보드', 'price': 80000, 'category': '액세서리'}
            ],
            'sales': [
                {'id': 1, 'user_id': 1, 'product_id': 1, 'quantity': 1, 'date': '2024-01-15'},
                {'id': 2, 'user_id': 2, 'product_id': 2, 'quantity': 2, 'date': '2024-01-15'},
                {'id': 3, 'user_id': 3, 'product_id': 3, 'quantity': 1, 'date': '2024-01-14'}
            ]
        }
    
    def get(self, endpoint, params=None):
        """GET 요청 시뮬레이션"""
        endpoint = endpoint.lstrip('/')
        
        if endpoint in self.mock_data:
            return {'data': self.mock_data[endpoint]}
        elif endpoint.startswith('users/') and endpoint.split('/')[1].isdigit():
            user_id = int(endpoint.split('/')[1])
            users = self.mock_data['users']
            user = next((u for u in users if u['id'] == user_id), None)
            return {'data': user} if user else {'error': 'User not found'}
        else:
            return {'error': 'Endpoint not found'}
    
    def post(self, endpoint, data=None):
        """POST 요청 시뮬레이션"""
        return {'success': True, 'message': 'Data created', 'data': data}

# API 클라이언트 사용
api_client = MockAPIClient()

# 사용자 데이터 가져오기
users_data = api_client.get('users')
if users_data and 'data' in users_data:
    users_df = pd.DataFrame(users_data['data'])
    print("사용자 데이터:")
    print(users_df)

# 제품 데이터 가져오기
products_data = api_client.get('products')
if products_data and 'data' in products_data:
    products_df = pd.DataFrame(products_data['data'])
    print("\n제품 데이터:")
    print(products_df)

# 판매 데이터 가져오기
sales_data = api_client.get('sales')
if sales_data and 'data' in sales_data:
    sales_df = pd.DataFrame(sales_data['data'])
    print("\n판매 데이터:")
    print(sales_df)

# 3. 데이터 통합 및 분석
print("\n=== 3. 데이터 통합 분석 ===")

def analyze_api_data(users_df, products_df, sales_df):
    """API 데이터 통합 분석"""
    # 데이터 병합
    merged_df = sales_df.merge(users_df, left_on='user_id', right_on='id', suffixes=('', '_user'))
    merged_df = merged_df.merge(products_df, left_on='product_id', right_on='id', suffixes=('', '_product'))
    
    # 분석
    analysis = {
        '총매출': (merged_df['quantity'] * merged_df['price']).sum(),
        '부서별 매출': merged_df.groupby('department')['price'].sum().to_dict(),
        '카테고리별 판매량': merged_df.groupby('category')['quantity'].sum().to_dict(),
        '일별 판매': merged_df.groupby('date')['quantity'].sum().to_dict()
    }
    
    return analysis, merged_df

# 데이터 분석 실행
analysis, merged_data = analyze_api_data(users_df, products_df, sales_df)

print("API 데이터 분석 결과:")
for key, value in analysis.items():
    print(f"{key}: {value}")

# 4. 에러 처리 및 재시도
print("\n=== 4. 에러 처리 및 재시도 ===")

def retry_on_failure(max_attempts=3, delay=1):
    """재시도 데코레이터"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (2 ** attempt)  # 지수 백오프
                        logging.warning(f"재시도 {attempt + 1}/{max_attempts} - {wait_time}초 후 재시도: {e}")
                        time.sleep(wait_time)
                    else:
                        logging.error(f"최종 실패: {e}")
            
            raise last_exception
        return wrapper
    return decorator

class RobustDataCollector:
    """강건한 데이터 수집기"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RobustDataCollector/1.0'
        })
    
    @retry_on_failure(max_attempts=3, delay=1)
    def fetch_with_retry(self, url, params=None):
        """재시도 포함 데이터 가져오기"""
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API 요청 실패: {e}")
    
    def simulate_api_call(self, endpoint):
        """API 호출 시뮬레이션"""
        # 30% 확률로 실패
        if np.random.random() < 0.3:
            raise Exception("API 호출 실패 (시뮬레이션)")
        
        return api_client.get(endpoint)

# 강건한 데이터 수집기 테스트
robust_collector = RobustDataCollector()

# 성공 케이스
try:
    result = robust_collector.simulate_api_call("users")
    print("성공적인 데이터 수집")
except Exception as e:
    print(f"데이터 수집 실패: {e}")

# 5. 데이터 품질 검증
print("\n=== 5. 데이터 품질 검증 ===")

class DataValidator:
    """데이터 검증기"""
    
    def __init__(self):
        self.validation_rules = {}
    
    def add_rule(self, field, rule_func, error_message):
        """검증 규칙 추가"""
        if field not in self.validation_rules:
            self.validation_rules[field] = []
        
        self.validation_rules[field].append({
            'func': rule_func,
            'message': error_message
        })
    
    def validate_data(self, data):
        """데이터 검증"""
        errors = []
        
        for item in data:
            item_errors = []
            
            for field, rules in self.validation_rules.items():
                if field in item:
                    value = item[field]
                    
                    for rule in rules:
                        if not rule['func'](value):
                            error_msg = f"{field}: {rule['message']} (값: {value})"
                            item_errors.append(error_msg)
            
            if item_errors:
                errors.append({
                    'item': item,
                    'errors': item_errors
                })
        
        return errors
    
    def generate_report(self, errors):
        """검증 보고서 생성"""
        return {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(news_data) if news_data else 0,
            'errors': errors[:3]  # 처음 3개 오류만 표시
        }

# 검증 규칙 설정
validator = DataValidator()

# 제목 검증
def is_valid_title(title):
    return isinstance(title, str) and len(title.strip()) > 0

validator.add_rule('title', is_valid_title, '제목은 비어있지 않은 문자열이어야 함')

# 조회수 검증
def is_valid_views(views):
    return isinstance(views, int) and views >= 0

validator.add_rule('views', is_valid_views, '조회수는 0 이상의 정수여야 함')

# 날짜 검증
def is_valid_date(date_str):
    try:
        pd.to_datetime(date_str)
        return True
    except:
        return False

validator.add_rule('date', is_valid_date, '유효한 날짜 형식이어야 함')

# 데이터 검증
validation_errors = validator.validate_data(news_data)
validation_report = validator.generate_report(validation_errors)

print("데이터 품질 검증 결과:")
print(f"총 오류 수: {validation_report['total_errors']}")
print(f"오류율: {validation_report['error_rate']:.2%}")

if validation_report['errors']:
    print("\n주요 오류:")
    for error in validation_report['errors']:
        for err_msg in error['errors']:
            print(f"- {err_msg}")

# 6. 자동화된 데이터 수집 시스템
print("\n=== 6. 자동화된 데이터 수집 시스템 ===")

class DataCollectionSystem:
    """데이터 수집 시스템"""
    
    def __init__(self):
        self.scrapers = {}
        self.api_clients = {}
        self.collected_data = {}
    
    def register_scraper(self, name, scraper_func):
        """스크래퍼 등록"""
        self.scrapers[name] = scraper_func
        logging.info(f"스크래퍼 등록: {name}")
    
    def register_api_client(self, name, client):
        """API 클라이언트 등록"""
        self.api_clients[name] = client
        logging.info(f"API 클라이언트 등록: {name}")
    
    def run_scraping_job(self, scraper_name):
        """스크래핑 작업 실행"""
        try:
            logging.info(f"스크래핑 시작: {scraper_name}")
            scraper = self.scrapers[scraper_name]
            data = scraper()
            
            if data:
                self.collected_data[f"{scraper_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"] = data
                logging.info(f"스크래핑 완료: {scraper_name}, 데이터 수: {len(data)}")
            else:
                logging.warning(f"스크래핑 실패: {scraper_name} - 데이터 없음")
                
        except Exception as e:
            logging.error(f"스크래핑 오류 {scraper_name}: {e}")
    
    def run_api_collection_job(self, client_name, endpoint):
        """API 데이터 수집 작업 실행"""
        try:
            logging.info(f"API 수집 시작: {client_name}/{endpoint}")
            client = self.api_clients[client_name]
            data = client.get(endpoint)
            
            if data and 'data' in data:
                self.collected_data[f"{client_name}_{endpoint}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"] = data['data']
                logging.info(f"API 수집 완료: {client_name}/{endpoint}, 데이터 수: {len(data['data'])}")
            else:
                logging.warning(f"API 수집 실패: {client_name}/{endpoint} - 데이터 없음")
                
        except Exception as e:
            logging.error(f"API 수집 오류 {client_name}/{endpoint}: {e}")
    
    def get_collected_data_summary(self):
        """수집된 데이터 요약"""
        summary = {}
        for key, data in self.collected_data.items():
            summary[key] = {
                'count': len(data),
                'type': type(data).__name__,
                'sample': data[0] if data else None
            }
        return summary

# 시스템 설정 및 실행
data_system = DataCollectionSystem()

# 스크래퍼 등록
def news_scraper():
    """뉴스 스크래퍼 (시뮬레이션)"""
    return parse_html_simple(sample_html)

data_system.register_scraper('news', news_scraper)

# API 클라이언트 등록
data_system.register_api_client('main_api', api_client)

# 작업 실행
print("데이터 수집 작업 실행:")
data_system.run_scraping_job('news')
data_system.run_api_collection_job('main_api', 'users')
data_system.run_api_collection_job('main_api', 'products')

# 수집된 데이터 요약
summary = data_system.get_collected_data_summary()
print("\n수집된 데이터 요약:")
for key, info in summary.items():
    print(f"{key}: {info['count']}개 ({info['type']})")

# 7. 스케줄링 시뮬레이션
print("\n=== 7. 스케줄링 시뮬레이션 ===")

def simulate_scheduled_tasks():
    """스케줄링 작업 시뮬레이션"""
    tasks = [
        ("09:00", "뉴스 스크래핑", lambda: data_system.run_scraping_job('news')),
        ("09:30", "사용자 API 수집", lambda: data_system.run_api_collection_job('main_api', 'users')),
        ("10:00", "제품 API 수집", lambda: data_system.run_api_collection_job('main_api', 'products')),
        ("23:00", "데이터 정리", lambda: print("오래된 데이터 정리 완료"))
    ]
    
    print("스케줄링된 작업:")
    for time, task_name, task_func in tasks:
        print(f"- {time}: {task_name}")
    
    # 첫 번째 작업 실행
    print("\n첫 번째 작업 실행:")
    tasks[0][2]()  # 뉴스 스크래핑 실행

simulate_scheduled_tasks()

print("\n=== 웹 스크래핑과 API 활용 예제 완료! ===")
print("1. HTML 파싱 및 데이터 추출")
print("2. REST API 연동 및 데이터 통합")
print("3. 에러 처리 및 재시도 메커니즘")
print("4. 데이터 품질 검증")
print("5. 자동화된 데이터 수집 시스템")
print("6. 스케줄링 기반 데이터 수집")
