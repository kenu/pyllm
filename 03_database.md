# 데이터베이스

## 관계형 데이터베이스 기초
SQL 및 관계형 데이터베이스의 기본 개념과 데이터 모델링 방법을 학습합니다.

**예제 (Python 내장 SQLite 사용):**
```python
import sqlite3

# 데이터베이스 연결 (없으면 'test.db' 파일 생성)
conn = sqlite3.connect('test.db')
cursor = conn.cursor()

# 테이블 생성
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE
)
''')

# 데이터 삽입
try:
    cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)", ('Alice', 'alice@example.com'))
    cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)", ('Bob', 'bob@example.com'))
except sqlite3.IntegrityError:
    print("데이터가 이미 존재합니다.")


# 데이터 조회
cursor.execute("SELECT * FROM users")
rows = cursor.fetchall()

print("\nUsers Table:")
for row in rows:
    print(row)

# 연결 종료
conn.close()
```

## 데이터베이스와 LLM 통합
LLM과 데이터베이스를 연동하여 데이터를 저장하고 검색하는 방법을 배웁니다.

**예제 (개념 코드):**
```python
import sqlite3
# from some_llm_library import LanguageModel

# llm = LanguageModel(model_name="gemini-pro")

# 1. 사용자의 자연어 질문을 SQL로 변환
# user_question = "Alice의 이메일 주소는 무엇인가요?"
# sql_query = llm.generate_sql(user_question, table_schema="users(id, name, email)")
# print(f"생성된 SQL: {sql_query}")
# >> 예상 결과: "SELECT email FROM users WHERE name = 'Alice'"

# 2. 데이터베이스에서 SQL 실행
# conn = sqlite3.connect('test.db')
# cursor = conn.cursor()
# cursor.execute(sql_query)
# result = cursor.fetchone()
# print(f"DB 조회 결과: {result[0]}")
# conn.close()
```

## MariaDB와 Docker 활용 (CRUD 예제)
Docker Compose를 사용하여 MariaDB를 실행하고, Python으로 CRUD 작업을 수행하는 방법을 학습합니다.

상세 예제 코드는 `pyllm/03/` 디렉토리에서 확인할 수 있습니다.

### 1. MariaDB 실행 (Docker Compose)
`docker-compose.yml` 파일이 있는 위치에서 다음 명령어를 실행합니다.
```bash
docker-compose up -d
```

### 2. Python 라이브러리 설치
```bash
pip install -r requirements.txt
```

### 3. CRUD 예제 실행
```bash
python crud.py
```

## NoSQL 데이터베이스 활용
MongoDB와 같은 NoSQL 데이터베이스의 특징과 LLM과의 연동 방법을 탐구합니다.

**예제 (pymongo 사용):**
```python
# pip install pymongo
from pymongo import MongoClient

# MongoDB 연결
# client = MongoClient('mongodb://localhost:27017/')
# db = client['mydatabase']
# collection = db['posts']

# 문서(JSON과 유사) 삽입
# post1 = {"author": "Alice", "text": "첫 번째 블로그 게시물입니다."}
# post2 = {"author": "Bob", "tags": ["mongodb", "python"], "text": "NoSQL은 유연해요."}
# collection.insert_many([post1, post2])

# 데이터 검색
# print("\nMongoDB 'posts' 컬렉션:")
# for post in collection.find({"author": "Bob"}):
#    print(post)

# client.close()
print("MongoDB 예제는 로컬에 MongoDB가 설치되어 있어야 실행 가능합니다.")
```
