# Homework - 데이터베이스

## 이 homework의 의미
관계형 데이터베이스의 기초부터 LLM과의 통합까지, 데이터 저장과 검색의 전 과정을 마스터하는 과제야. Docker를 활용한 실전 환경 구축으로 현업 스킬을 익히는 거지.

## 관련 정보 사이트
1. **SQLZoo**: https://sqlzoo.net/
2. **PostgreSQL Tutorial**: https://www.postgresqltutorial.com/

## 프로세스 진행 논리적 흐름
1. **SQL 기초 학습** → CREATE, INSERT, SELECT, UPDATE, DELETE
2. **데이터 모델링** → 테이블 설계 및 관계 정의
3. **Docker 환경 구축** → MariaDB 컨테이너 실행
4. **CRUD 구현** → Python으로 데이터베이스 조작
5. **LLM 통합** → 자연어를 SQL로 변환

## 권장 코드
```python
import sqlite3

# 데이터베이스 연결
conn = sqlite3.connect('mydb.db')
cursor = conn.cursor()

# 테이블 생성
cursor.execute('''
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    price INTEGER,
    stock INTEGER
)
''')

# 데이터 삽입
cursor.execute("INSERT INTO products (name, price, stock) VALUES (?, ?, ?)",
               ('노트북', 1500000, 10))

# 데이터 조회
cursor.execute("SELECT * FROM products WHERE price > ?", (1000000,))
results = cursor.fetchall()

for row in results:
    print(row)

conn.commit()
conn.close()
```

## 관련 업계 회사
1. **Oracle** - 세계 최대 데이터베이스 솔루션 기업
2. **MongoDB** - NoSQL 데이터베이스 선두 주자
3. **AWS (Amazon RDS)** - 클라우드 데이터베이스 서비스 제공
