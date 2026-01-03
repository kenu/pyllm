# 웹 스크래핑과 API 활용

## BeautifulSoup으로 데이터 수집
웹페이지에서 구조화된 데이터를 추출하는 방법을 익힙니다.

### 1. BeautifulSoup 기초
```python
from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
import json
from urllib.parse import urljoin, urlparse
import re

# 샘플 HTML 생성 (실제 웹사이트 대신)
sample_html = """
<!DOCTYPE html>
<html>
<head>
    <title>뉴스 포털</title>
</head>
<body>
    <header>
        <h1>오늘의 뉴스</h1>
        <nav>
            <ul>
                <li><a href="#politics">정치</a></li>
                <li><a href="#economy">경제</a></li>
                <li><a href="#society">사회</a></li>
            </ul>
        </nav>
    </header>
    
    <main>
        <section id="politics">
            <h2>정치 뉴스</h2>
            <article class="news-item">
                <h3 class="title">국회 의원회의 개최</h3>
                <p class="content">오늘 오후 2시 국회에서 중요 의안이 논의됩니다.</p>
                <div class="meta">
                    <span class="author">김기자</span>
                    <span class="date">2024-01-15</span>
                    <span class="views">1,234</span>
                </div>
            </article>
            <article class="news-item">
                <h3 class="title">새로운 정책 발표</h3>
                <p class="content">정부가 새로운 경제 정책을 발표했습니다.</p>
                <div class="meta">
                    <span class="author">이기자</span>
                    <span class="date">2024-01-15</span>
                    <span class="views">2,567</span>
                </div>
            </article>
        </section>
        
        <section id="economy">
            <h2>경제 뉴스</h2>
            <article class="news-item">
                <h3 class="title">주가 상승</h3>
                <p class="content">코스피 지수가 큰 폭으로 상승했습니다.</p>
                <div class="meta">
                    <span class="author">박기자</span>
                    <span class="date">2024-01-15</span>
                    <span class="views">3,890</span>
                </div>
            </article>
        </section>
        
        <section id="society">
            <h2>사회 뉴스</h2>
            <article class="news-item">
                <h3 class="title">날씨 변화</h3>
                <p class="content">내일부터 기온이 하락할 것으로 예상됩니다.</p>
                <div class="meta">
                    <span class="author">최기자</span>
                    <span class="date">2024-01-15</span>
                    <span class="views">987</span>
                </div>
            </article>
        </section>
    </main>
    
    <footer>
        <p>&copy; 2024 뉴스 포털</p>
    </footer>
</body>
</html>
"""

# BeautifulSoup 파싱
soup = BeautifulSoup(sample_html, 'html.parser')

print("=== BeautifulSoup 기초 ===")
print(f"페이지 제목: {soup.title.text}")
print(f"본문 텍스트: {soup.get_text()[:100]}...")

# 요소 찾기
print("\n=== 요소 찾기 ===")

# 태그로 찾기
headers = soup.find_all(['h1', 'h2', 'h3'])
print("제목들:")
for header in headers:
    print(f"- {header.name}: {header.text.strip()}")

# 클래스로 찾기
news_items = soup.find_all('article', class_='news-item')
print(f"\n뉴스 기사 수: {len(news_items)}")

# CSS 선택자로 찾기
titles = soup.select('.title')
print("뉴스 제목:")
for title in titles:
    print(f"- {title.text.strip()}")

# 속성으로 찾기
links = soup.find_all('a', href=True)
print(f"\n링크 수: {len(links)}")
for link in links:
    print(f"- {link.text}: {link['href']}")
```

### 2. 구조화된 데이터 추출
```python
# 뉴스 기사 데이터 추출
def extract_news_data(soup):
    """뉴스 기사에서 구조화된 데이터 추출"""
    news_data = []
    
    # 모든 뉴스 기사 찾기
    articles = soup.find_all('article', class_='news-item')
    
    for article in articles:
        try:
            # 제목
            title = article.find('h3', class_='title').text.strip()
            
            # 내용
            content = article.find('p', class_='content').text.strip()
            
            # 메타데이터
            meta = article.find('div', class_='meta')
            author = meta.find('span', class_='author').text.strip()
            date = meta.find('span', class_='date').text.strip()
            views = meta.find('span', class_='views').text.strip()
            views = int(views.replace(',', ''))
            
            # 섹션 (부모 요소에서 찾기)
            section = article.find_parent('section')
            section_name = section.find('h2').text.strip() if section else 'Unknown'
            
            news_data.append({
                'title': title,
                'content': content,
                'author': author,
                'date': date,
                'views': views,
                'section': section_name
            })
            
        except AttributeError as e:
            print(f"데이터 추출 오류: {e}")
            continue
    
    return news_data

# 데이터 추출
news_data = extract_news_data(soup)

# DataFrame으로 변환
df_news = pd.DataFrame(news_data)

print("\n=== 추출된 뉴스 데이터 ===")
print(df_news)

# 데이터 분석
print("\n=== 뉴스 데이터 분석 ===")
print(f"총 뉴스 수: {len(df_news)}")
print(f"섹션별 뉴스 수:")
print(df_news['section'].value_counts())
print(f"\n평균 조회수: {df_news['views'].mean():.1f}")
print(f"가장 많이 본 뉴스: {df_news.loc[df_news['views'].idxmax(), 'title']}")
```

## Scrapy로 대규모 데이터 수집
Scrapy 프레임워크를 사용하여 체계적인 웹 크롤링을 구현합니다.

### 1. Scrapy 스파이더 구현
```python
# Scrapy 스파이더 예제 (개념적 구현)
import scrapy
from scrapy.crawler import CrawlerProcess
from urllib.parse import urljoin

class NewsSpider(scrapy.Spider):
    name = 'news_spider'
    start_urls = ['http://example-news.com']  # 실제 URL로 변경 필요
    
    def parse(self, response):
        """메인 페이지 파싱"""
        # 뉴스 기사 링크 추출
        news_links = response.css('article.news-item a::attr(href)').getall()
        
        for link in news_links:
            # 절대 URL로 변환
            full_url = urljoin(response.url, link)
            yield response.follow(full_url, callback=self.parse_news)
        
        # 다음 페이지 링크
        next_page = response.css('a.next-page::attr(href)').get()
        if next_page:
            yield response.follow(next_page, callback=self.parse)
    
    def parse_news(self, response):
        """개별 뉴스 기사 파싱"""
        yield {
            'title': response.css('h1.article-title::text').get().strip(),
            'content': response.css('.article-content::text').get().strip(),
            'author': response.css('.author-name::text').get().strip(),
            'date': response.css('.publish-date::text').get().strip(),
            'category': response.css('.category::text').get().strip(),
            'url': response.url,
            'scraped_at': pd.Timestamp.now().isoformat()
        }

# Scrapy 실행 (실제 실행은 별도 환경에서)
print("=== Scrapy 스파이더 구조 ===")
print("1. start_urls에서 시작")
print("2. parse() 메서드로 메인 페이지 분석")
print("3. parse_news() 메서드로 개별 기사 추출")
print("4. 다음 페이지로 이동하여 반복")
```

### 2. 데이터 파이프라인
```python
# Scrapy 데이터 파이프라인 예제
class NewsPipeline:
    """데이터 처리 파이프라인"""
    
    def process_item(self, item, spider):
        """아이템 처리"""
        # 데이터 정제
        item['title'] = self.clean_text(item['title'])
        item['content'] = self.clean_text(item['content'])
        
        # 날짜 형식 표준화
        item['date'] = self.standardize_date(item['date'])
        
        # 중복 체크
        if self.is_duplicate(item):
            raise scrapy.exceptions.DropItem("Duplicate item")
        
        return item
    
    def clean_text(self, text):
        """텍스트 정제"""
        if text:
            # 불필요한 공백 제거
            text = re.sub(r'\s+', ' ', text)
            # 특수문자 제거
            text = re.sub(r'[^\w\s]', '', text)
            return text.strip()
        return ''
    
    def standardize_date(self, date_str):
        """날짜 형식 표준화"""
        try:
            return pd.to_datetime(date_str).isoformat()
        except:
            return pd.Timestamp.now().isoformat()
    
    def is_duplicate(self, item):
        """중복 체크"""
        # 실제로는 데이터베이스나 파일에서 중복 체크
        return False

# 데이터 저장 파이프라인
class DatabasePipeline:
    """데이터베이스 저장 파이프라인"""
    
    def open_spider(self, spider):
        """스파이더 시작 시 초기화"""
        self.items = []
    
    def close_spider(self, spider):
        """스파이더 종료 시 데이터 저장"""
        if self.items:
            df = pd.DataFrame(self.items)
            df.to_csv('scraped_news.csv', index=False)
            print(f"{len(df)}개의 뉴스 기사 저장 완료")
    
    def process_item(self, item, spider):
        """아이템 처리 및 저장"""
        self.items.append(dict(item))
        return item

print("\n=== 데이터 파이프라인 구조 ===")
print("1. NewsPipeline: 데이터 정제 및 표준화")
print("2. DatabasePipeline: 데이터베이스 저장")
```

## REST API 연동
다양한 REST API를 연동하여 데이터를 수집하고 처리합니다.

### 1. 기본 API 연동
```python
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time

# API 기본 설정
class APIClient:
    def __init__(self, base_url, api_key=None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        # 기본 헤더 설정
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'DataAnalysisBot/1.0'
        })
        
        if api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {api_key}'
            })
    
    def get(self, endpoint, params=None):
        """GET 요청"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"GET 요청 오류: {e}")
            return None
    
    def post(self, endpoint, data=None):
        """POST 요청"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"POST 요청 오류: {e}")
            return None

# 샘플 API 클라이언트 (실제 API가 아닌 시뮬레이션)
class MockAPIClient(APIClient):
    def __init__(self):
        super().__init__("https://api.example.com", "mock_api_key")
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
        """모의 GET 요청"""
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

# API 클라이언트 사용
print("=== REST API 연동 ===")

api_client = MockAPIClient()

# 사용자 데이터 가져오기
users_data = api_client.get('users')
if users_data:
    users_df = pd.DataFrame(users_data['data'])
    print("사용자 데이터:")
    print(users_df)

# 제품 데이터 가져오기
products_data = api_client.get('products')
if products_data:
    products_df = pd.DataFrame(products_data['data'])
    print("\n제품 데이터:")
    print(products_df)

# 판매 데이터 가져오기
sales_data = api_client.get('sales')
if sales_data:
    sales_df = pd.DataFrame(sales_data['data'])
    print("\n판매 데이터:")
    print(sales_df)
```

### 2. 고급 API 연동
```python
# 페이지네이션 처리
class PaginatedAPIClient(APIClient):
    def get_all_pages(self, endpoint, params=None, max_pages=10):
        """모든 페이지 데이터 가져오기"""
        all_data = []
        page = 1
        
        while page <= max_pages:
            if params is None:
                params = {}
            params['page'] = page
            params['limit'] = 100  # 페이지당 100개
            
            response = self.get(endpoint, params)
            
            if not response or 'data' not in response:
                break
            
            data = response['data']
            if not data:
                break
            
            all_data.extend(data)
            page += 1
            
            # API 레이트 리밋 준수
            time.sleep(0.1)
        
        return all_data

# 실시간 데이터 스트리밍
class StreamingAPIClient(APIClient):
    def stream_data(self, endpoint, callback, duration=60):
        """실시간 데이터 스트리밍"""
        start_time = time.time()
        
        while time.time() - start_time < duration:
            response = self.get(endpoint)
            
            if response and 'data' in response:
                for item in response['data']:
                    callback(item)
            
            time.sleep(1)  # 1초 간격

# 데이터 처리 콜백 함수
def process_data_item(item):
    """데이터 아이템 처리"""
    print(f"처리된 아이템: {item}")
    # 실제로는 데이터베이스에 저장하거나 분석 수행

# 페이지네이션 예제
paginated_client = PaginatedAPIClient("https://api.example.com")
all_users = paginated_client.get_all_pages('users')

print(f"\n총 {len(all_users)}개의 사용자 데이터 수집")

# 데이터 통합 및 분석
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

print("\n=== API 데이터 분석 결과 ===")
for key, value in analysis.items():
    print(f"{key}: {value}")
```

## 자동화된 데이터 수집 시스템
주기적으로 데이터를 수집하고 처리하는 자동화 시스템을 구축합니다.

### 1. 스케줄링 시스템
```python
import schedule
import threading
import time
from datetime import datetime
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)

class DataCollectionSystem:
    def __init__(self):
        self.scrapers = {}
        self.api_clients = {}
        self.storage = DataStorage()
        self.running = False
    
    def register_scraper(self, name, scraper):
        """스크래퍼 등록"""
        self.scrapers[name] = scraper
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
            data = scraper.scrape()
            
            if data:
                self.storage.save_data(f"{scraper_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", data)
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
                self.storage.save_data(f"{client_name}_{endpoint}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", data['data'])
                logging.info(f"API 수집 완료: {client_name}/{endpoint}, 데이터 수: {len(data['data'])}")
            else:
                logging.warning(f"API 수집 실패: {client_name}/{endpoint} - 데이터 없음")
                
        except Exception as e:
            logging.error(f"API 수집 오류 {client_name}/{endpoint}: {e}")
    
    def setup_schedule(self):
        """스케줄 설정"""
        # 매시간 뉴스 스크래핑
        schedule.every().hour.do(self.run_scraping_job, 'news')
        
        # 매일 API 데이터 수집
        schedule.every().day.at("09:00").do(self.run_api_collection_job, 'main_api', 'users')
        schedule.every().day.at("09:30").do(self.run_api_collection_job, 'main_api', 'products')
        
        # 매주 데이터 정리
        schedule.every().sunday.at("23:00").do(self.storage.cleanup_old_data)
        
        logging.info("스케줄 설정 완료")
    
    def start(self):
        """시스템 시작"""
        self.running = True
        self.setup_schedule()
        
        def run_scheduler():
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # 1분마다 체크
        
        scheduler_thread = threading.Thread(target=run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        
        logging.info("데이터 수집 시스템 시작")
    
    def stop(self):
        """시스템 중지"""
        self.running = False
        logging.info("데이터 수집 시스템 중지")

# 데이터 저장소
class DataStorage:
    def __init__(self, storage_path='collected_data'):
        self.storage_path = storage_path
        import os
        os.makedirs(storage_path, exist_ok=True)
    
    def save_data(self, filename, data):
        """데이터 저장"""
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
        
        filepath = f"{self.storage_path}/{filename}.csv"
        df.to_csv(filepath, index=False)
        return filepath
    
    def cleanup_old_data(self, days_to_keep=7):
        """오래된 데이터 정리"""
        import os
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for filename in os.listdir(self.storage_path):
            filepath = os.path.join(self.storage_path, filename)
            file_time = datetime.fromtimestamp(os.path.getctime(filepath))
            
            if file_time < cutoff_date:
                os.remove(filepath)
                logging.info(f"오래된 파일 삭제: {filename}")

# 샘플 스크래퍼
class NewsScraper:
    def scrape(self):
        """뉴스 스크래핑 (시뮬레이션)"""
        # 실제로는 requests + BeautifulSoup 사용
        mock_news = [
            {'title': f'뉴스 {i}', 'content': f'내용 {i}', 'date': datetime.now().strftime('%Y-%m-%d')}
            for i in range(5)
        ]
        return mock_news

# 시스템 설정 및 실행
print("\n=== 자동화된 데이터 수집 시스템 ===")

# 시스템 초기화
data_system = DataCollectionSystem()

# 스크래퍼 등록
news_scraper = NewsScraper()
data_system.register_scraper('news', news_scraper)

# API 클라이언트 등록
api_client = MockAPIClient()
data_system.register_api_client('main_api', api_client)

# 스케줄 설정 (실제 실행은 주석 처리)
# data_system.start()

# 테스트 실행 (개별 작업)
print("테스트 실행:")
data_system.run_scraping_job('news')
data_system.run_api_collection_job('main_api', 'users')

print("\n스케줄 설정 예제:")
print("- 매시간: 뉴스 스크래핑")
print("- 매일 09:00: 사용자 API 데이터 수집")
print("- 매일 09:30: 제품 API 데이터 수집")
print("- 매주 일요일 23:00: 오래된 데이터 정리")
```

### 2. 에러 처리 및 재시도
```python
# 에러 처리 및 재시도 메커니즘
import random
from functools import wraps
import backoff

class DataCollectionError(Exception):
    """데이터 수집 오류"""
    pass

def retry_on_failure(max_attempts=3, backoff_factor=2):
    """재시도 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        wait_time = backoff_factor ** attempt
                        logging.warning(f"재시도 {attempt + 1}/{max_attempts} - {wait_time}초 후 재시도: {e}")
                        time.sleep(wait_time)
                    else:
                        logging.error(f"최종 실패: {e}")
            
            raise last_exception
        
        return wrapper
    return decorator

class RobustDataCollector:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RobustDataCollector/1.0'
        })
    
    @retry_on_failure(max_attempts=3, backoff_factor=2)
    def fetch_with_retry(self, url, params=None):
        """재시도 포함 데이터 가져오기"""
        response = self.session.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    
    def scrape_with_circuit_breaker(self, url, max_failures=5, timeout=60):
        """서킷 브레이커 패턴"""
        if not hasattr(self, '_failure_count'):
            self._failure_count = 0
            self._last_failure_time = None
        
        current_time = time.time()
        
        # 타임아웃 기간 동안은 요청 차단
        if (self._last_failure_time and 
            current_time - self._last_failure_time < timeout and
            self._failure_count >= max_failures):
            raise DataCollectionError("서킷 브레이커 활성화 - 요청 차단")
        
        try:
            result = self.fetch_with_retry(url)
            # 성공 시 실패 카운트 리셋
            self._failure_count = 0
            return result
            
        except Exception as e:
            self._failure_count += 1
            self._last_failure_time = current_time
            raise

# 강건한 데이터 수집기 테스트
print("\n=== 에러 처리 및 재시도 ===")

robust_collector = RobustDataCollector()

# 성공 케이스
try:
    result = robust_collector.fetch_with_retry("https://httpbin.org/get")
    print("성공적인 데이터 수집")
except Exception as e:
    print(f"데이터 수집 실패: {e}")

print("에러 처리 전략:")
print("1. 재시도 메커니즘 (지수 백오프)")
print("2. 서킷 브레이커 패턴")
print("3. 타임아웃 설정")
print("4. 상세한 로깅")
```

## 데이터 품질 관리
수집된 데이터의 품질을 관리하고 검증합니다.

### 1. 데이터 검증
```python
# 데이터 품질 검증
class DataValidator:
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
        warnings = []
        
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
        
        return errors, warnings
    
    def generate_report(self, validation_result):
        """검증 보고서 생성"""
        errors, warnings = validation_result
        
        report = {
            'total_items': len(errors) + len(warnings),
            'error_count': len(errors),
            'warning_count': len(warnings),
            'error_rate': len(errors) / (len(errors) + len(warnings)) if (len(errors) + len(warnings)) > 0 else 0,
            'errors': errors[:5],  # 처음 5개 오류만 표시
            'warnings': warnings[:5]
        }
        
        return report

# 검증 규칙 설정
validator = DataValidator()

# 이메일 검증
def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

validator.add_rule('email', is_valid_email, '유효하지 않은 이메일 형식')

# 가격 검증
def is_positive_price(price):
    return isinstance(price, (int, float)) and price > 0

validator.add_rule('price', is_positive_price, '가격은 양수여야 함')

# 날짜 검증
def is_valid_date(date_str):
    try:
        pd.to_datetime(date_str)
        return True
    except:
        return False

validator.add_rule('date', is_valid_date, '유효하지 않은 날짜 형식')

# 테스트 데이터
test_data = [
    {'name': '제품1', 'email': 'valid@example.com', 'price': 1000, 'date': '2024-01-15'},
    {'name': '제품2', 'email': 'invalid-email', 'price': -500, 'date': 'invalid-date'},
    {'name': '제품3', 'email': 'another@example.com', 'price': 2000, 'date': '2024-01-16'}
]

# 데이터 검증
validation_result = validator.validate_data(test_data)
validation_report = validator.generate_report(validation_result)

print("\n=== 데이터 품질 검증 ===")
print(f"전체 아이템: {validation_report['total_items']}")
print(f"오류 수: {validation_report['error_count']}")
print(f"오류율: {validation_report['error_rate']:.2%}")

if validation_report['errors']:
    print("\n주요 오류:")
    for error in validation_report['errors']:
        for err_msg in error['errors']:
            print(f"- {err_msg}")
```

### 2. 데이터 모니터링
```python
# 데이터 수집 모니터링
class DataCollectionMonitor:
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'data_quality_scores': []
        }
    
    def record_request(self, success=True, response_time=None, quality_score=None):
        """요청 기록"""
        self.metrics['total_requests'] += 1
        
        if success:
            self.metrics['successful_requests'] += 1
        else:
            self.metrics['failed_requests'] += 1
        
        if response_time:
            self.metrics['response_times'].append(response_time)
        
        if quality_score:
            self.metrics['data_quality_scores'].append(quality_score)
    
    def get_success_rate(self):
        """성공률 계산"""
        if self.metrics['total_requests'] == 0:
            return 0
        return self.metrics['successful_requests'] / self.metrics['total_requests']
    
    def get_avg_response_time(self):
        """평균 응답 시간"""
        if not self.metrics['response_times']:
            return 0
        return sum(self.metrics['response_times']) / len(self.metrics['response_times'])
    
    def get_avg_quality_score(self):
        """평균 데이터 품질 점수"""
        if not self.metrics['data_quality_scores']:
            return 0
        return sum(self.metrics['data_quality_scores']) / len(self.metrics['data_quality_scores'])
    
    def generate_dashboard(self):
        """대시보드 데이터 생성"""
        return {
            'success_rate': self.get_success_rate(),
            'avg_response_time': self.get_avg_response_time(),
            'avg_quality_score': self.get_avg_quality_score(),
            'total_requests': self.metrics['total_requests'],
            'failed_requests': self.metrics['failed_requests']
        }

# 모니터링 데모
monitor = DataCollectionMonitor()

# 모의 요청 기록
for i in range(10):
    success = random.random() > 0.2  # 80% 성공률
    response_time = random.uniform(0.1, 2.0)
    quality_score = random.uniform(0.7, 1.0)
    
    monitor.record_request(success, response_time, quality_score)

dashboard = monitor.generate_dashboard()

print("\n=== 데이터 수집 모니터링 ===")
print(f"성공률: {dashboard['success_rate']:.2%}")
print(f"평균 응답 시간: {dashboard['avg_response_time']:.2f}초")
print(f"평균 품질 점수: {dashboard['avg_quality_score']:.2f}")
print(f"총 요청 수: {dashboard['total_requests']}")
print(f"실패 요청 수: {dashboard['failed_requests']}")

print("\n=== 웹 스크래핑과 API 활용 예제 완료! ===")
print("1. BeautifulSoup으로 HTML 파싱")
print("2. Scrapy로 대규모 데이터 수집")
print("3. REST API 연동 및 데이터 통합")
print("4. 자동화된 데이터 수집 시스템")
print("5. 에러 처리 및 재시도 메커니즘")
print("6. 데이터 품질 관리 및 모니터링")
