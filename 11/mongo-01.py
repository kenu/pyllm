# MongoDB 연동 (설치되지 않은 경우 시뮬레이션)
import sqlite3
import pandas as pd
import numpy as np

# 샘플 데이터베이스 생성
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
    manager_id INTEGER,
    FOREIGN KEY (department_id) REFERENCES departments(id),
    FOREIGN KEY (manager_id) REFERENCES employees(id)
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

cursor.execute('''
CREATE TABLE projects (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department_id INTEGER,
    start_date DATE,
    end_date DATE,
    budget INTEGER,
    status TEXT
)
''')

cursor.execute('''
CREATE TABLE employee_projects (
    employee_id INTEGER,
    project_id INTEGER,
    role TEXT,
    hours_worked INTEGER,
    PRIMARY KEY (employee_id, project_id),
    FOREIGN KEY (employee_id) REFERENCES employees(id),
    FOREIGN KEY (project_id) REFERENCES projects(id)
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

projects_data = [
    (1, '웹사이트 리뉴얼', 1, '2023-01-01', '2023-06-30', 200000, '진행중'),
    (2, '모바일 앱 개발', 1, '2023-03-15', '2023-09-15', 300000, '진행중'),
    (3, '마케팅 캠페인', 2, '2023-02-01', '2023-05-31', 150000, '완료'),
    (4, '신규 고객 확보', 4, '2023-01-15', '2023-12-31', 250000, '진행중')
]

employee_projects_data = [
    (1, 1, 'PM', 160),
    (2, 1, '개발자', 200),
    (3, 1, '개발자', 180),
    (2, 2, '리드 개발자', 220),
    (3, 2, '개발자', 190),
    (4, 3, '기획자', 120),
    (5, 3, '디자이너', 100),
    (7, 4, 'PM', 80),
    (8, 4, '영업사원', 160)
]

cursor.executemany('INSERT INTO departments VALUES (?, ?, ?, ?)', departments_data)
cursor.executemany('INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?, ?)', employees_data)
cursor.executemany('INSERT INTO projects VALUES (?, ?, ?, ?, ?, ?, ?)', projects_data)
cursor.executemany('INSERT INTO employee_projects VALUES (?, ?, ?, ?)', employee_projects_data)

conn.commit()

print("=== 데이터베이스 설정 완료 ===")
print("테이블: employees, departments, projects, employee_projects")
try:
    from pymongo import MongoClient
    MONGO_AVAILABLE = True
    print("MongoDB 사용 가능")
except ImportError:
    print("MongoDB가 설치되지 않았습니다. 시뮬레이션으로 진행합니다.")
    MONGO_AVAILABLE = False

# MongoDB 데이터 모델링 클래스
class MongoDBSimulator:
    """MongoDB 시뮬레이터 (설치되지 않은 경우 사용)"""

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

        # 간단한 쿼리 처리
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
        """간단한 집계 파이프라인 시뮬레이션"""
        if collection_name not in self.collections:
            return []

        docs = self.collections[collection_name]

        # $group 파이프라인 처리
        for stage in pipeline:
            if '$group' in stage:
                group_spec = stage['$group']
                grouped = {}

                for doc in docs:
                    # 그룹 키 생성
                    if '_id' in group_spec:
                        group_id = group_spec['_id']
                        if isinstance(group_id, str) and group_id.startswith('$'):
                            group_key = str(doc.get(group_id[1:], 'default'))
                        else:
                            group_key = str(group_id)
                    else:
                        group_key = 'all'

                    if group_key not in grouped:
                        grouped[group_key] = {'_id': group_key}
                        # 집계 필드 초기화
                        for field, expr in group_spec.items():
                            if field != '_id':
                                if isinstance(expr, dict):
                                    for op, value in expr.items():
                                        if op == '$sum':
                                            grouped[group_key][field] = 0 if value != 1 else 0
                                        elif op == '$avg':
                                            grouped[group_key][field] = 0
                                            grouped[group_key][field + '_count'] = 0
                                        elif op == '$max':
                                            grouped[group_key][field] = float('-inf')
                                        elif op == '$min':
                                            grouped[group_key][field] = float('inf')

                    # 값 집계
                    for field, expr in group_spec.items():
                        if field != '_id':
                            if isinstance(expr, dict):
                                for op, value in expr.items():
                                    if op == '$sum':
                                        if value == 1:
                                            grouped[group_key][field] = grouped[group_key].get(field, 0) + 1
                                        elif isinstance(value, str) and value.startswith('$'):
                                            field_name = value[1:]
                                            if field_name in doc:
                                                grouped[group_key][field] = grouped[group_key].get(field, 0) + doc[field_name]
                                    elif op == '$avg':
                                        if isinstance(value, str) and value.startswith('$'):
                                            field_name = value[1:]
                                            if field_name in doc:
                                                grouped[group_key][field] = grouped[group_key].get(field, 0) + doc[field_name]
                                                grouped[group_key][field + '_count'] = grouped[group_key].get(field + '_count', 0) + 1
                                    elif op == '$max':
                                        if isinstance(value, str) and value.startswith('$'):
                                            field_name = value[1:]
                                            if field_name in doc:
                                                current = grouped[group_key].get(field, float('-inf'))
                                                grouped[group_key][field] = max(current, doc[field_name])
                                    elif op == '$min':
                                        if isinstance(value, str) and value.startswith('$'):
                                            field_name = value[1:]
                                            if field_name in doc:
                                                current = grouped[group_key].get(field, float('inf'))
                                                grouped[group_key][field] = min(current, doc[field_name])

                # 평균값 계산
                for group_data in grouped.values():
                    for field in list(group_data.keys()):
                        if field.endswith('_count'):
                            base_field = field[:-6]  # Remove '_count'
                            if base_field in group_data and group_data[field] > 0:
                                group_data[base_field] = group_data[base_field] / group_data[field]
                            del group_data[field]

                docs = list(grouped.values())

        return docs

# MongoDB 연결 또는 시뮬레이션
if MONGO_AVAILABLE:
    try:
        client = MongoClient('mongodb://root:example@localhost:27017/', serverSelectionTimeoutMS=1000)
        db = client['company_db']
        print("MongoDB 연결 성공")
    except:
        print("MongoDB 서버에 연결할 수 없습니다. 시뮬레이션으로 진행합니다.")
        MONGO_AVAILABLE = False

if not MONGO_AVAILABLE:
    # 시뮬레이터 사용
    class MockDB:
        def __init__(self):
            self.simulator = MongoDBSimulator()

        def __getitem__(self, collection_name):
            return MockCollection(self.simulator, collection_name)

    class MockCollection:
        def __init__(self, simulator, name):
            self.simulator = simulator
            self.name = name

        def insert_one(self, document):
            return self.simulator.insert_one(self.name, document)

        def find(self, query=None):
            return MockCursor(self.simulator.find(self.name, query))

        def aggregate(self, pipeline):
            return self.simulator.aggregate(self.name, pipeline)

    class MockCursor:
        def __init__(self, documents):
            self.documents = documents

        def __iter__(self):
            return iter(self.documents)

    db = MockDB()
if MONGO_AVAILABLE:
    try:
        client = MongoClient('mongodb://root:example@localhost:27017/', serverSelectionTimeoutMS=1000)
        db = client['company_db']
        print("MongoDB 연결 성공")
    except:
        print("MongoDB 서버에 연결할 수 없습니다. 시뮬레이션으로 진행합니다.")
        MONGO_AVAILABLE = False
    # 시뮬레이터 사용
    class MockDB:
        def __init__(self):
            self.simulator = MongoDBSimulator()

        def __getitem__(self, collection_name):
            return MockCollection(self.simulator, collection_name)

    class MockCollection:
        def __init__(self, simulator, name):
            self.simulator = simulator
            self.name = name

        def insert_one(self, document):
            return self.simulator.insert_one(self.name, document)

        def find(self, query=None):
            return MockCursor(self.simulator.find(self.name, query))

        def aggregate(self, pipeline):
            return self.simulator.aggregate(self.name, pipeline)

    class MockCursor:
        def __init__(self, documents):
            self.documents = documents

        def __iter__(self):
            return iter(self.documents)

    db = MockDB()

# MongoDB 데이터 모델링
print("\n=== MongoDB 데이터 모델링 ===")

# 직원 컬렉션 데이터 구조
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
        "projects": [
            {"name": "웹사이트 리뉴얼", "role": "PM", "hours": 160},
            {"name": "모바일 앱 개발", "role": "아키텍트", "hours": 80}
        ],
        "hire_date": "2020-01-15",
        "status": "active",
        "performance": {
            "2023": {"rating": 4.5, "goals": 8},
            "2022": {"rating": 4.2, "goals": 7}
        }
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
        "projects": [
            {"name": "웹사이트 리뉴얼", "role": "개발자", "hours": 200},
            {"name": "모바일 앱 개발", "role": "백엔드 개발자", "hours": 180}
        ],
        "hire_date": "2020-03-22",
        "status": "active",
        "performance": {
            "2023": {"rating": 4.3, "goals": 7},
            "2022": {"rating": 4.1, "goals": 6}
        }
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
        "projects": [
            {"name": "마케팅 캠페인", "role": "기획자", "hours": 120}
        ],
        "hire_date": "2019-06-10",
        "status": "active",
        "performance": {
            "2023": {"rating": 4.6, "goals": 9},
            "2022": {"rating": 4.4, "goals": 8}
        }
    }
]

# 데이터 삽입
employees_collection = db['employees']
for emp in employees_mongo:
    result = employees_collection.insert_one(emp)
    print(f"삽입된 문서 ID: {result['_id']}")

# 데이터 조회
print("\n=== MongoDB 데이터 조회 ===")

# 모든 직원 조회
all_employees = list(employees_collection.find())
print("모든 직원:")
for emp in all_employees:
    print(f"- {emp['name']} ({emp['position']})")

# 특정 조건으로 조회
dev_team = list(employees_collection.find({"department.name": "개발팀"}))
print("\n개발팀 직원:")
for emp in dev_team:
    print(f"- {emp['name']}: {', '.join(emp['skills'])}")

# 복잡한 쿼리
high_performers = list(employees_collection.find({
    "performance.2023.rating": {"$gte": 4.5},
    "skills": {"$in": ["Python", "Java"]}
}))
print("\n고성과자 (Python/Java 스킬):")
for emp in high_performers:
    print(f"- {emp['name']}: {emp['performance']['2023']['rating']}")

# MongoDB 집계 파이프라인
print("\n=== MongoDB 집계 파이프라인 ===")

# 부서별 통계
pipeline1 = [
    {
        "$group": {
            "_id": "$department.name",
            "count": {"$sum": 1},
            "avg_salary": {"$avg": "$salary"},
            "total_projects": {"$sum": {"$size": "$projects"}}
        }
    },
    {
        "$sort": {"avg_salary": -1}
    }
]

dept_stats = list(employees_collection.aggregate(pipeline1))
print("부서별 통계:")
for stat in dept_stats:
    print(f"- {stat['_id']}: {stat['count']}명, 평균급여: {stat['avg_salary']:.0f}")

# 스킬별 분석
pipeline2 = [
    {"$unwind": "$skills"},
    {
        "$group": {
            "_id": "$skills",
            "count": {"$sum": 1},
            "avg_salary": {"$avg": "$salary"}
        }
    },
    {"$sort": {"count": -1}}
]

skill_stats = list(employees_collection.aggregate(pipeline2))
print("\n스킬별 분석:")
for stat in skill_stats:
    print(f"- {stat['_id']}: {stat['count']}명, 평균급여: {stat['avg_salary']:.0f}")

# ETL (Extract, Transform, Load) 파이프라인
class DataPipeline:
    def __init__(self, sql_conn, nosql_db):
        self.sql_conn = sql_conn
        self.nosql_db = nosql_db

    def extract_from_sql(self, query):
        """SQL에서 데이터 추출"""
        return pd.read_sql_query(query, self.sql_conn)

    def transform_data(self, df):
        """데이터 변환"""
        # 중첩 구조로 변환
        transformed = []

        for _, row in df.iterrows():
            doc = {
                "employee_id": row['id'],
                "name": row['name'],
                "position": row['position'],
                "department": {
                    "id": row['department_id'],
                    "name": row['department_name'],
                    "location": row['location']
                },
                "salary": row['salary'],
                "hire_date": row['hire_date'],
                "projects": []
            }
            transformed.append(doc)

        return transformed

    def load_to_nosql(self, documents, collection_name):
        """NoSQL에 데이터 적재"""
        collection = self.nosql_db[collection_name]

        # 기존 데이터 삭제
        if hasattr(collection, 'delete_many'):
            collection.delete_many({})

        # 새 데이터 삽입
        for doc in documents:
            collection.insert_one(doc)

        return len(documents)

    def run_pipeline(self, sql_query, target_collection):
        """전체 파이프라인 실행"""
        print("ETL 파이프라인 시작...")

        # Extract
        print("1. 데이터 추출...")
        raw_data = self.extract_from_sql(sql_query)
        print(f"   추출된 레코드: {len(raw_data)}")

        # Transform
        print("2. 데이터 변환...")
        transformed_data = self.transform_data(raw_data)
        print(f"   변환된 문서: {len(transformed_data)}")

        # Load
        print("3. 데이터 적재...")
        loaded_count = self.load_to_nosql(transformed_data, target_collection)
        print(f"   적재된 문서: {loaded_count}")

        print("ETL 파이프라인 완료!")
        return loaded_count

# 복합 SQL 쿼리
complex_query = '''
SELECT
    e.id,
    e.name,
    e.position,
    e.salary,
    e.hire_date,
    e.department_id,
    d.name as department_name,
    d.location,
    COUNT(p.id) as project_count,
    COALESCE(SUM(ep.hours_worked), 0) as total_hours
FROM employees e
JOIN departments d ON e.department_id = d.id
LEFT JOIN employee_projects ep ON e.id = ep.employee_id
LEFT JOIN projects p ON ep.project_id = p.id
GROUP BY e.id, e.name, e.position, e.salary, e.hire_date,
         e.department_id, d.name, d.location
'''

# 파이프라인 실행
pipeline = DataPipeline(conn, db)
result_count = pipeline.run_pipeline(complex_query, 'employees_etl')

# 결과 확인
print(f"\n파이프라인 결과: {result_count}개 문서 처리")
etl_results = list(db['employees_etl'].find())
for result in etl_results[:2]:  # 처음 2개만 표시
    print(f"- {result['name']} ({result['department']['name']})")


# 실시간 데이터 동기화 시뮬레이션
import time
from datetime import datetime

class DataSync:
    def __init__(self, sql_conn, nosql_db):
        self.sql_conn = sql_conn
        self.nosql_db = nosql_db
        self.last_sync = None

    def detect_changes(self):
        """변경된 데이터 감지"""
        if self.last_sync is None:
            # 전체 동기화
            query = "SELECT * FROM employees"
        else:
            # 증분 동기화 (시뮬레이션)
            query = f"SELECT * FROM employees WHERE id > {self.last_sync}"

        return pd.read_sql_query(query, self.sql_conn)

    def sync_to_nosql(self, changes):
        """NoSQL로 동기화"""
        collection = self.nosql_db['realtime_sync']

        synced_count = 0
        for _, row in changes.iterrows():
            doc = {
                "employee_id": row['id'],
                "name": row['name'],
                "position": row['position'],
                "salary": row['salary'],
                "sync_timestamp": datetime.now().isoformat(),
                "sync_type": "incremental" if self.last_sync else "full"
            }

            # upsert (update or insert)
            existing = collection.find_one({"employee_id": row['id']})
            if existing:
                collection.update_one({"_id": existing['_id']}, {"$set": doc})
            else:
                collection.insert_one(doc)

            synced_count += 1

        return synced_count

    def run_sync(self):
        """동기화 실행"""
        print(f"\n[{datetime.now()}] 데이터 동기화 시작...")

        changes = self.detect_changes()
        if len(changes) > 0:
            synced = self.sync_to_nosql(changes)
            print(f"동기화된 레코드: {synced}")
            self.last_sync = changes['id'].max() if len(changes) > 0 else self.last_sync
        else:
            print("변경된 데이터 없음")

        print(f"동기화 완료: {datetime.now()}")

# 실시간 동기화 시뮬레이션
sync = DataSync(conn, db)

# 초기 동기화
sync.run_sync()

# 주기적 동기화 시뮬레이션
print("\n실시간 동기화 시뮬레이션 (3회)")
for i in range(3):
    time.sleep(1)  # 1초 대기
    sync.run_sync()

# NoSQL 성능 최적화
print("\n=== NoSQL 성능 최적화 ===")

# 인덱스 생성 시뮬레이션
def create_indexes(collection):
    """인덱스 생성 시뮬레이션"""
    indexes = [
        {"field": "department.name", "type": "single"},
        {"field": "salary", "type": "single"},
        {"field": "skills", "type": "multikey"},
        {"field": "performance.2023.rating", "type": "nested"}
    ]

    print("생성된 인덱스:")
    for idx in indexes:
        print(f"- {idx['field']} ({idx['type']})")

    return indexes

# 쿼리 성능 테스트
def benchmark_queries(collection):
    """쿼리 성능 벤치마크"""
    queries = [
        {
            "name": "부서별 조회",
            "query": {"department.name": "개발팀"}
        },
        {
            "name": "급여 범위 조회",
            "query": {"salary": {"$gte": 60000}}
        },
        {
            "name": "스킬 기반 조회",
            "query": {"skills": {"$in": ["Python", "Java"]}}
        }
    ]

    print("\n쿼리 성능 벤치마크:")
    for q in queries:
        start_time = time.time()
        results = list(collection.find(q['query']))
        end_time = time.time()

        print(f"- {q['name']}: {(end_time - start_time)*1000:.2f}ms ({len(results)}건)")

# 인덱스 생성 및 성능 테스트
create_indexes(db['employees'])
benchmark_queries(db['employees'])
