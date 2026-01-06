# Homework - 웹 스크래핑 & API

## 이 homework의 의미
웹에서 데이터를 자동으로 수집하고 API를 통해 외부 서비스와 연동하는 과제야. 실시간 데이터를 확보하고 다양한 소스를 통합하는 데이터 엔지니어링 스킬을 익히는 거지.

## 관련 정보 사이트
1. **Beautiful Soup Documentation**: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
2. **Requests Library**: https://requests.readthedocs.io/

## 프로세스 진행 논리적 흐름
1. **타겟 분석** → 수집할 웹사이트/API 조사
2. **HTML 파싱** → BeautifulSoup으로 데이터 추출
3. **API 호출** → REST API 요청 및 응답 처리
4. **데이터 저장** → CSV, JSON, 데이터베이스에 저장
5. **자동화** → 스케줄러로 주기적 수집

## 권장 코드
```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json

# 웹 스크래핑 예제
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # 예: 제목 추출
    titles = soup.find_all('h2', class_='title')
    data = [title.text.strip() for title in titles]
    
    return data

# API 호출 예제
def fetch_api_data(api_url, params=None):
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'application/json'
    }
    
    response = requests.get(api_url, params=params, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None

# 공공 API 예제 (날씨)
# api_key = 'YOUR_API_KEY'
# weather_url = f'https://api.openweathermap.org/data/2.5/weather'
# params = {'q': 'Seoul', 'appid': api_key, 'units': 'metric'}
# weather_data = fetch_api_data(weather_url, params)

# 데이터 저장
# df = pd.DataFrame(weather_data)
# df.to_csv('weather_data.csv', index=False)

print("웹 스크래핑 및 API 호출 준비 완료!")
```

## 관련 업계 회사
1. **Scrapy** - 오픈소스 웹 크롤링 프레임워크
2. **Apify** - 웹 스크래핑 및 자동화 플랫폼
3. **Postman** - API 개발 및 테스트 도구
