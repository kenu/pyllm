# Homework - 데이터베이스 고급

## 이 homework의 의미
복잡한 SQL 쿼리 작성, 인덱싱, 트랜잭션 관리 등 데이터베이스의 고급 기능을 마스터하는 과제야. 대용량 데이터를 효율적으로 처리하고 성능을 최적화하는 실무 스킬을 익히는 거지.

## 관련 정보 사이트
1. **Use The Index, Luke**: https://use-the-index-luke.com/
2. **SQL Performance Explained**: https://sql-performance-explained.com/

## 프로세스 진행 논리적 흐름
1. **고급 JOIN 학습** → INNER, LEFT, RIGHT, FULL OUTER JOIN
2. **서브쿼리 활용** → 중첩 쿼리로 복잡한 조건 처리
3. **인덱스 최적화** → 쿼리 성능 향상
4. **트랜잭션 관리** → ACID 속성 이해 및 구현
5. **쿼리 튜닝** → EXPLAIN으로 실행 계획 분석

## 권장 코드
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

# 테이블 생성
cursor.execute('''
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT,
    department_id INTEGER,
    salary INTEGER,
    manager_id INTEGER
)
''')

cursor.execute('''
CREATE TABLE departments (
    id INTEGER PRIMARY KEY,
    name TEXT,
    budget INTEGER
)
''')

# 데이터 삽입
departments = [(1, '개발', 5000000), (2, '영업', 3000000)]
employees = [
    (1, '김철수', 1, 6000000, None),
    (2, '이영희', 1, 5500000, 1),
    (3, '박민준', 2, 4500000, None)
]

cursor.executemany("INSERT INTO departments VALUES (?, ?, ?)", departments)
cursor.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?)", employees)

# 고급 쿼리: JOIN과 서브쿼리
query = '''
SELECT 
    e.name AS 직원명,
    d.name AS 부서명,
    e.salary AS 급여,
    (SELECT AVG(salary) FROM employees WHERE department_id = e.department_id) AS 부서평균급여
FROM employees e
JOIN departments d ON e.department_id = d.id
WHERE e.salary > (SELECT AVG(salary) FROM employees)
ORDER BY e.salary DESC
'''

df = pd.read_sql_query(query, conn)
print(df)

conn.close()
```

## 관련 업계 회사
1. **Percona** - MySQL/PostgreSQL 성능 최적화 전문
2. **Cockroach Labs** - 분산 SQL 데이터베이스
3. **Redis Labs** - 인메모리 데이터베이스 솔루션
