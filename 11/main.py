import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import hashlib
import base64

print("=== 데이터베이스 연동 심화 예제 ===")

# 1. SQLite 데이터베이스 설정
print("\n=== 1. 데이터베이스 설정 ===")

conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

# 테이블 생성
cursor.execute('''
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department_id INTEGER,
    position TEXT,
    salary INTEGER,
    hire_date DATE,
    manager_id INTEGER
)
''')

cursor.execute('''
CREATE TABLE departments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    location TEXT,
    budget INTEGER
)
''')

# 샘플 데이터 삽입
departments_data = [
    (1, '개발팀', '서울', 1000000),
    (2, '마케팅팀', '부산', 500000),
    (3, '인사팀', '대구', 300000),
    (4, '영업팀', '대전', 800000)
]

employees_data = [
    (1, '김철수', 1, '팀장', 80000, '2020-01-15', None),
    (2, '이영희', 1, '선임', 65000, '2020-03-22', 1),
    (3, '박민준', 1, '주니어', 45000, '2021-02-01', 1),
    (4, '최지아', 2, '팀장', 70000, '2019-06-10', None),
    (5, '정서연', 2, '선임', 55000, '2020-08-15', 4),
    (6, '강동훈', 3, '팀장', 60000, '2018-09-20', None),
    (7, '윤미래', 4, '팀장', 75000, '2019-11-05', None),
    (8, '한상준', 4, '선임', 58000, '2020-12-01', 7)
]

cursor.executemany('INSERT INTO departments VALUES (?, ?, ?, ?)', departments_data)
cursor.executemany('INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?, ?)', employees_data)
conn.commit()

print("데이터베이스 설정 완료")
print(f"부서: {len(departments_data)}개")
print(f"직원: {len(employees_data)}개")

# 2. 고급 SQL 쿼리
print("\n=== 2. 고급 SQL 쿼리 ===")

# 복잡한 JOIN
query1 = '''
SELECT 
    e.name AS 직원명,
    e.position AS 직급,
    d.name AS 부서명,
    e.salary AS 급여,
    CASE 
        WHEN e.salary > 70000 THEN '고급'
        WHEN e.salary > 50000 THEN '중급'
        ELSE '초급'
    END AS 급여등급
FROM employees e
JOIN departments d ON e.department_id = d.id
ORDER BY e.salary DESC
'''

df1 = pd.read_sql_query(query1, conn)
print("직원-부서 정보:")
print(df1)

# 집계 함수와 그룹화
query2 = '''
SELECT 
    d.name AS 부서명,
    COUNT(e.id) AS 직원수,
    AVG(e.salary) AS 평균급여,
    MAX(e.salary) AS 최고급여,
    MIN(e.salary) AS 최저급여
FROM departments d
LEFT JOIN employees e ON d.id = e.department_id
GROUP BY d.id, d.name
HAVING COUNT(e.id) > 0
'''

df2 = pd.read_sql_query(query2, conn)
print("\n부서별 통계:")
print(df2)

# 서브쿼리
query3 = '''
SELECT 
    name AS 직원명,
    salary AS 급여,
    (SELECT AVG(salary) FROM employees) AS 전체평균급여,
    ROUND(salary - (SELECT AVG(salary) FROM employees), 0) AS 급여차이
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees)
ORDER BY 급여차이 DESC
'''

df3 = pd.read_sql_query(query3, conn)
print("\n평균 이상 급여 직원:")
print(df3)

# 3. MongoDB 시뮬레이션
print("\n=== 3. NoSQL (MongoDB) 시뮬레이션 ===")

class MongoDBSimulator:
    def __init__(self):
        self.collections = {}
    
    def insert_one(self, collection_name, document):
        if collection_name not in self.collections:
            self.collections[collection_name] = []
        document['_id'] = len(self.collections[collection_name]) + 1
        self.collections[collection_name].append(document)
        return document
    
    def find(self, collection_name, query=None):
        if collection_name not in self.collections:
            return []
        
        if query is None:
            return self.collections[collection_name]
        
        results = []
        for doc in self.collections[collection_name]:
            match = True
            for key, value in query.items():
                if key not in doc or doc[key] != value:
                    match = False
                    break
            if match:
                results.append(doc)
        return results
    
    def aggregate(self, collection_name, pipeline):
        if collection_name not in self.collections:
            return []
        
        docs = self.collections[collection_name]
        
        # 간단한 $group 파이프라인 처리
        for stage in pipeline:
            if '$group' in stage:
                group_spec = stage['$group']
                grouped = {}
                
                for doc in docs:
                    if '_id' in group_spec:
                        group_key = str(doc.get(group_spec['_id'], 'default'))
                    else:
                        group_key = 'all'
                    
                    if group_key not in grouped:
                        grouped[group_key] = {}
                        for field, expr in group_spec.items():
                            if field != '_id':
                                if expr == '$sum':
                                    grouped[group_key][field] = 0
                                elif expr == '$avg':
                                    grouped[group_key][field] = 0
                                    grouped[group_key][field + '_count'] = 0
                    
                    for field, expr in group_spec.items():
                        if field != '_id':
                            field_name = expr[1:] if expr.startswith('$') else field
                            if field_name in doc:
                                value = doc[field_name]
                                if expr == '$sum':
                                    grouped[group_key][field] += value
                                elif expr == '$avg':
                                    grouped[group_key][field] += value
                                    grouped[group_key][field + '_count'] += 1
                
                # 평균값 계산
                for group_data in grouped.values():
                    for field in list(group_data.keys()):
                        if field.endswith('_count'):
                            base_field = field.replace('_count', '')
                            if base_field in group_data:
                                group_data[base_field] = group_data[base_field] / group_data[field]
                            del group_data[field]
                
                docs = [{'_id': k, **v} for k, v in grouped.items()]
        
        return docs

# MongoDB 시뮬레이션
mongo_sim = MongoDBSimulator()

# 문서 지향 데이터 모델
employees_mongo = [
    {
        "name": "김철수",
        "email": "kim@company.com",
        "position": "팀장",
        "department": {
            "name": "개발팀",
            "location": "서울"
        },
        "salary": 80000,
        "skills": ["Python", "JavaScript", "React"],
        "hire_date": "2020-01-15",
        "status": "active"
    },
    {
        "name": "이영희",
        "email": "lee@company.com",
        "position": "선임",
        "department": {
            "name": "개발팀",
            "location": "서울"
        },
        "salary": 65000,
        "skills": ["Java", "Spring", "MySQL"],
        "hire_date": "2020-03-22",
        "status": "active"
    },
    {
        "name": "최지아",
        "email": "choi@company.com",
        "position": "팀장",
        "department": {
            "name": "마케팅팀",
            "location": "부산"
        },
        "salary": 70000,
        "skills": ["Marketing", "Analytics", "SEO"],
        "hire_date": "2019-06-10",
        "status": "active"
    }
]

# 데이터 삽입
for emp in employees_mongo:
    mongo_sim.insert_one('employees', emp)

print("MongoDB 데이터 삽입 완료")

# 데이터 조회
all_employees = mongo_sim.find('employees')
print("\n모든 직원:")
for emp in all_employees:
    print(f"- {emp['name']} ({emp['position']})")

# 특정 조건 조회
dev_team = mongo_sim.find('employees', {"department.name": "개발팀"})
print("\n개발팀 직원:")
for emp in dev_team:
    print(f"- {emp['name']}: {', '.join(emp['skills'])}")

# 집계 파이프라인
pipeline = [
    {
        "$group": {
            "_id": "$department.name",
            "count": {"$sum": 1},
            "avg_salary": {"$avg": "$salary"}
        }
    },
    {
        "$sort": {"avg_salary": -1}
    }
]

dept_stats = mongo_sim.aggregate('employees', pipeline)
print("\n부서별 통계 (MongoDB):")
for stat in dept_stats:
    print(f"- {stat['_id']}: {stat['count']}명, 평균급여: {stat['avg_salary']:.0f}")

# 4. ETL 파이프라인
print("\n=== 4. ETL 파이프라인 ===")

class DataPipeline:
    def __init__(self, sql_conn, nosql_db):
        self.sql_conn = sql_conn
        self.nosql_db = nosql_db
    
    def extract_from_sql(self, query):
        return pd.read_sql_query(query, self.sql_conn)
    
    def transform_data(self, df):
        transformed = []
        for _, row in df.iterrows():
            doc = {
                "employee_id": row['id'],
                "name": row['name'],
                "position": row['position'],
                "department": {
                    "id": row['department_id'],
                    "name": row['department_name']
                },
                "salary": row['salary'],
                "hire_date": row['hire_date']
            }
            transformed.append(doc)
        return transformed
    
    def load_to_nosql(self, documents, collection_name):
        collection = self.nosql_db
        loaded_count = 0
        
        for doc in documents:
            collection.insert_one(collection_name, doc)
            loaded_count += 1
        
        return loaded_count
    
    def run_pipeline(self, sql_query, target_collection):
        print("ETL 파이프라인 시작...")
        
        # Extract
        raw_data = self.extract_from_sql(sql_query)
        print(f"추출된 레코드: {len(raw_data)}")
        
        # Transform
        transformed_data = self.transform_data(raw_data)
        print(f"변환된 문서: {len(transformed_data)}")
        
        # Load
        loaded_count = self.load_to_nosql(transformed_data, target_collection)
        print(f"적재된 문서: {loaded_count}")
        
        return loaded_count

# ETL 실행
etl_query = '''
SELECT 
    e.id,
    e.name,
    e.position,
    e.salary,
    e.hire_date,
    e.department_id,
    d.name as department_name
FROM employees e
JOIN departments d ON e.department_id = d.id
'''

pipeline = DataPipeline(conn, mongo_sim)
result_count = pipeline.run_pipeline(etl_query, 'employees_etl')

# ETL 결과 확인
etl_results = mongo_sim.find('employees_etl')
print(f"\nETL 결과: {len(etl_results)}개 문서")
for result in etl_results[:2]:
    print(f"- {result['name']} ({result['department']['name']})")

# 5. 데이터베이스 보안
print("\n=== 5. 데이터베이스 보안 ===")

# 비밀번호 해싱
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# 간단한 데이터 암호화
def simple_encrypt(data):
    if isinstance(data, str):
        data_bytes = data.encode()
    else:
        data_bytes = str(data).encode()
    
    # 간단한 XOR 암호화 (실제로는 더 안전한 방법 사용)
    key = b'secret_key_12345'
    encrypted = bytes([b ^ key[i % len(key)] for i, b in enumerate(data_bytes)])
    return base64.b64encode(encrypted).decode()

def simple_decrypt(encrypted_data):
    encrypted_bytes = base64.b64decode(encrypted_data.encode())
    key = b'secret_key_12345'
    decrypted = bytes([b ^ key[i % len(key)] for i, b in enumerate(encrypted_bytes)])
    return decrypted.decode()

# 보안 데모
password = "user123"
hashed_password = hash_password(password)
print(f"비밀번호 해시: {hashed_password}")

salary = 80000
encrypted_salary = simple_encrypt(salary)
print(f"암호화된 급여: {encrypted_salary}")

decrypted_salary = simple_decrypt(encrypted_salary)
print(f"복호화된 급여: {decrypted_salary}")

# 6. 성능 최적화
print("\n=== 6. 성능 최적화 ===")

# 인덱스 생성
cursor.execute('CREATE INDEX idx_employees_salary ON employees(salary)')
cursor.execute('CREATE INDEX idx_employees_department ON employees(department_id)')

print("인덱스 생성 완료")

# 쿼리 실행 계획 분석
explain_query = "EXPLAIN QUERY PLAN SELECT * FROM employees WHERE salary > 60000"
cursor.execute(explain_query)
plan = cursor.fetchall()

print("\n쿼리 실행 계획:")
for step in plan:
    print(f"  {step}")

# 성능 테스트
import time

def benchmark_query(query, name):
    start_time = time.time()
    result = pd.read_sql_query(query, conn)
    end_time = time.time()
    
    execution_time = (end_time - start_time) * 1000
    print(f"- {name}: {execution_time:.2f}ms ({len(result)}건)")
    return execution_time

# 쿼리 성능 비교
print("\n쿼리 성능 벤치마크:")
benchmark_query("SELECT * FROM employees", "전체 직원 조회")
benchmark_query("SELECT * FROM employees WHERE salary > 60000", "급여 필터링")
benchmark_query("SELECT * FROM employees WHERE department_id = 1", "부서 필터링")

# 데이터베이스 연결 종료
conn.close()
print("\n데이터베이스 연결 종료")

print("\n=== 데이터베이스 연동 심화 예제 완료! ===")
print("1. 고급 SQL 쿼리 (JOIN, 서브쿼리, 집계)")
print("2. NoSQL (MongoDB) 시뮬레이션")
print("3. ETL 파이프라인 구축")
print("4. 데이터 보안 (해싱, 암호화)")
print("5. 성능 최적화 (인덱스, 실행 계획)")
