# 대규모 데이터 처리

## Dask로 병렬 데이터 처리
Dask를 사용하여 대용량 데이터를 효율적으로 처리합니다.

### 1. Dask 기초 개념
```python
import dask
import dask.dataframe as dd
import dask.array as da
import pandas as pd
import numpy as np
from dask.distributed import Client
import time

print("=== Dask 기초 ===")

# Dask 클라이언트 설정
client = Client(n_workers=4, threads_per_worker=2)
print(f"Dask 클라이언트: {client}")

# 대용량 데이터 생성 (시뮬레이션)
def create_large_dataset(size=10_000_000):
    """대용량 데이터셋 생성"""
    np.random.seed(42)
    data = {
        'id': range(size),
        'value': np.random.randn(size),
        'category': np.random.choice(['A', 'B', 'C', 'D'], size),
        'timestamp': pd.date_range('2020-01-01', periods=size, freq='1min')
    }
    return pd.DataFrame(data)

# Pandas DataFrame 생성
print("Pandas DataFrame 생성...")
start_time = time.time()
df_pandas = create_large_dataset(1_000_000)  # 100만 개
pandas_time = time.time() - start_time
print(f"Pandas 생성 시간: {pandas_time:.2f}초")

# Dask DataFrame 생성
print("Dask DataFrame 생성...")
start_time = time.time()
df_dask = dd.from_pandas(df_pandas, npartitions=4)
dask_time = time.time() - start_time
print(f"Dask 생성 시간: {dask_time:.2f}초")

# 기본 연산 비교
print("\n=== 기본 연산 비교 ===")

# Pandas 연산
start_time = time.time()
pandas_mean = df_pandas['value'].mean()
pandas_compute_time = time.time() - start_time
print(f"Pandas 평균 계산: {pandas_mean:.4f} ({pandas_compute_time:.4f}초)")

# Dask 연산
start_time = time.time()
dask_mean = df_dask['value'].mean().compute()
dask_compute_time = time.time() - start_time
print(f"Dask 평균 계산: {dask_mean:.4f} ({dask_compute_time:.4f}초)")

# 복잡한 연산
print("\n=== 복잡한 연산 ===")

def complex_operation(df):
    """복잡한 데이터 처리 연산"""
    result = df.groupby('category').agg({
        'value': ['mean', 'std', 'count'],
        'id': 'nunique'
    })
    return result

# Pandas 복잡 연산
start_time = time.time()
pandas_complex = complex_operation(df_pandas)
pandas_complex_time = time.time() - start_time
print(f"Pandas 복잡 연산: {pandas_complex_time:.2f}초")

# Dask 복잡 연산
start_time = time.time()
dask_complex = df_dask.groupby('category').agg({
    'value': ['mean', 'std', 'count'],
    'id': 'nunique'
}).compute()
dask_complex_time = time.time() - start_time
print(f"Dask 복잡 연산: {dask_complex_time:.2f}초")

print("\n연산 결과:")
print(dask_complex)
```

### 2. Dask Array 사용
```python
# Dask Array 예제
print("\n=== Dask Array ===")

# 대용량 배열 생성
x = da.random.random((10000, 10000), chunks=(1000, 1000))
y = da.random.random((10000, 10000), chunks=(1000, 1000))

print(f"Dask Array 형태: {x.shape}")
print(f"Chunk 크기: {x.chunksize}")

# 행렬 연산
print("행렬 곱셈...")
start_time = time.time()
z = da.dot(x, y)
result = z.compute()
compute_time = time.time() - start_time
print(f"행렬 곱셈 시간: {compute_time:.2f}초")
print(f"결과 형태: {result.shape}")

# 통계 연산
print("\n통계 연산...")
mean_val = x.mean().compute()
std_val = x.std().compute()
print(f"평균: {mean_val:.4f}")
print(f"표준편차: {std_val:.4f}")

# FFT 연산
print("\nFFT 연산...")
signal = da.random.random(1000000, chunks=100000)
fft_result = da.fft.fft(signal).compute()
print(f"FFT 결과 크기: {len(fft_result)}")
```

### 3. Dask 지연 연산
```python
# Dask 지연 연산
print("\n=== Dask 지연 연산 ===")

@dask.delayed
def process_data(data):
    """데이터 처리 함수"""
    time.sleep(0.1)  # 시뮬레이션
    return data * 2

@dask.delayed
def aggregate_results(results):
    """결과 집계 함수"""
    time.sleep(0.05)  # 시뮬레이션
    return sum(results)

# 여러 데이터 처리
data_chunks = [np.random.random(1000) for _ in range(10)]

# 지연 연산 생성
processed_chunks = [process_data(chunk) for chunk in data_chunks]
final_result = aggregate_results(processed_chunks)

# 실제 계산 실행
print("지연 연산 실행...")
start_time = time.time()
result = final_result.compute()
execution_time = time.time() - start_time
print(f"실행 시간: {execution_time:.2f}초")
print(f"결과: {result:.4f}")

# 시각화
import matplotlib.pyplot as plt

# Dask 작업 그래프 시각화
print("\n작업 그래프:")
final_result.visualize('dask_graph.png')
print("작업 그래프 저장: dask_graph.png")
```

## Apache Spark 기초
Spark를 사용하여 분산 데이터 처리를 구현합니다.

### 1. PySpark 설정 및 기본 사용
```python
# PySpark 예제 (개념적 구현)
print("=== Apache Spark ===")

# SparkSession 설정 (실제 실행 시)
spark_config = '''
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, mean, stddev

# SparkSession 생성
spark = SparkSession.builder \\
    .appName("LargeScaleDataProcessing") \\
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \\
    .getOrCreate()

# Spark 설정 확인
print(f"Spark 버전: {spark.version}")
print(f"Spark 마스터: {spark.sparkContext.master}")
'''

print("SparkSession 설정:")
print(spark_config)

# Spark DataFrame 예제
spark_dataframe_example = '''
# 샘플 데이터 생성
data = [
    (1, "Alice", 25, "Engineering", 75000),
    (2, "Bob", 30, "Marketing", 65000),
    (3, "Charlie", 35, "Engineering", 80000),
    (4, "Diana", 28, "Sales", 55000),
    (5, "Eve", 32, "Marketing", 70000)
]

# 스키마 정의
schema = ["id", "name", "age", "department", "salary"]

# DataFrame 생성
df = spark.createDataFrame(data, schema)

# DataFrame 기본 정보
print("DataFrame 스키마:")
df.printSchema()

print("DataFrame 데이터:")
df.show()

# 기본 연산
print("기본 통계:")
df.describe().show()

# 필터링
print("Engineering 부서 직원:")
df.filter(col("department") == "Engineering").show()

# 그룹화
print("부서별 통계:")
dept_stats = df.groupBy("department").agg(
    count("*").alias("employee_count"),
    mean("age").alias("avg_age"),
    mean("salary").alias("avg_salary")
)
dept_stats.show()
'''

print("\nSpark DataFrame 예제:")
print(spark_dataframe_example)
```

### 2. Spark SQL 및 고급 연산
```python
# Spark SQL 예제
spark_sql_example = '''
# 임시 뷰 생성
df.createOrReplaceTempView("employees")

# SQL 쿼리 실행
print("SQL 쿼리 결과:")
result = spark.sql("""
    SELECT 
        department,
        COUNT(*) as employee_count,
        AVG(age) as avg_age,
        AVG(salary) as avg_salary,
        MAX(salary) as max_salary
    FROM employees
    GROUP BY department
    ORDER BY avg_salary DESC
""")
result.show()

# 창 함수 사용
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, desc

# 부서 내 급여 순위
window_spec = Window.partitionBy("department").orderBy(desc("salary"))

ranked_df = df.withColumn(
    "salary_rank",
    row_number().over(window_spec)
)

print("부서별 급여 순위:")
ranked_df.show()

# 조인 연산
departments_data = [
    ("Engineering", "Tech"),
    ("Marketing", "Business"),
    ("Sales", "Business")
]

dept_df = spark.createDataFrame(departments_data, ["dept_name", "category"])

# 조인
joined_df = df.join(
    dept_df,
    df.department == dept_df.dept_name,
    "left"
)

print("조인 결과:")
joined_df.show()
'''

print("\nSpark SQL 예제:")
print(spark_sql_example)
```

### 3. Spark MLlib 머신러닝
```python
# Spark MLlib 예제
spark_mllib_example = '''
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# 데이터 준비
data = [
    (1.0, 2.0, 3.0, 4.0),
    (2.0, 3.0, 4.0, 5.0),
    (3.0, 4.0, 5.0, 6.0),
    (4.0, 5.0, 6.0, 7.0),
    (5.0, 6.0, 7.0, 8.0)
]

df = spark.createDataFrame(data, ["feature1", "feature2", "feature3", "label"])

# 특성 벡터화
assembler = VectorAssembler(
    inputCols=["feature1", "feature2", "feature3"],
    outputCol="features"
)

data_df = assembler.transform(df)

# 선형 회귀 모델
lr = LinearRegression(featuresCol="features", labelCol="label")

# 모델 학습
lr_model = lr.fit(data_df)

# 예측
predictions = lr_model.transform(data_df)

print("예측 결과:")
predictions.select("features", "label", "prediction").show()

# 모델 평가
evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"
)

rmse = evaluator.evaluate(predictions)
print(f"RMSE: {rmse}")
'''

print("\nSpark MLlib 예제:")
print(spark_mllib_example)
```

## 분산 파일 시스템
HDFS와 같은 분산 파일 시스템을 활용합니다.

### 1. HDFS 기본 개념
```python
# HDFS 개념 및 명령어
hdfs_commands = {
    '디렉토리 생성': 'hdfs dfs -mkdir /user/data',
    '파일 업로드': 'hdfs dfs -put local_file.csv /user/data/',
    '파일 목록': 'hdfs dfs -ls /user/data',
    '파일 내용': 'hdfs dfs -cat /user/data/file.csv',
    '파일 다운로드': 'hdfs dfs -get /user/data/file.csv local_copy.csv',
    '파일 삭제': 'hdfs dfs -rm /user/data/file.csv',
    '디렉토리 삭제': 'hdfs dfs -rm -r /user/data',
    '디스크 사용량': 'hdfs dfs -du -h /user/data',
    '파일 복사': 'hdfs dfs -cp /user/data/file.csv /user/backup/',
    '파일 이동': 'hdfs dfs -mv /user/data/file.csv /user/archive/'
}

print("=== HDFS 기본 명령어 ===")
for cmd, desc in hdfs_commands.items():
    print(f"{cmd}: {desc}")

# HDFS Python 연동 예제
hdfs_python_example = '''
from hdfs import InsecureClient

# HDFS 클라이언트 설정
client = InsecureClient('http://namenode:9870', user='hadoop')

# 파일 목록 조회
files = client.list('/user/data')
print("HDFS 파일 목록:", files)

# 파일 읽기
with client.read('/user/data/large_file.csv') as reader:
    data = reader.read()
    print(f"파일 크기: {len(data)} bytes")

# 파일 쓰기
data_to_write = "Hello, HDFS!"
with client.write('/user/data/output.txt', overwrite=True) as writer:
    writer.write(data_to_write.encode('utf-8'))

# 디렉토리 생성
client.makedirs('/user/data/new_directory')

# 파일 정보
file_info = client.status('/user/data/output.txt')
print("파일 정보:", file_info)
'''

print("\nHDFS Python 연동:")
print(hdfs_python_example)
```

### 2. Parquet 및 ORC 포맷
```python
# Parquet 포맷 예제
print("\n=== Parquet 및 ORC 포맷 ===")

# Parquet 쓰기/읽기 예제
parquet_example = '''
# Pandas DataFrame 생성
import pandas as pd
import numpy as np

data = pd.DataFrame({
    'id': range(1000000),
    'value': np.random.randn(1000000),
    'category': np.random.choice(['A', 'B', 'C'], 1000000)
})

# Parquet 파일로 저장
data.to_parquet('large_data.parquet')

# Parquet 파일 읽기
df_parquet = pd.read_parquet('large_data.parquet')

# Dask로 Parquet 처리
import dask.dataframe as dd

ddf = dd.read_parquet('large_data.parquet')
result = ddf.groupby('category')['value'].mean().compute()
'''

print("Parquet 포맷 처리:")
print(parquet_example)

# 포맷 비교
format_comparison = {
    'CSV': {
        '장점': '간단함, 범용성',
        '단점': '용량 큼, 읽기 느림',
        '압축': '없음',
        '스키마': '없음'
    },
    'JSON': {
        '장점': '구조화됨, 범용성',
        '단점': '용량 큼, 파싱 느림',
        '압축': '없음',
        '스키마': '없음'
    },
    'Parquet': {
        '장점': '압축 효율, 컬럼 기반, 빠름',
        '단점': '복잡함',
        '압축': '있음',
        '스키마': '있음'
    },
    'ORC': {
        '장점': '압축 효율, ACID 지원',
        '단점': 'Hadoop 종속적',
        '압축': '있음',
        '스키마': '있음'
    }
}

df_formats = pd.DataFrame(format_comparison).T
print("\n파일 포맷 비교:")
print(df_formats)
```

## 스트리밍 데이터 처리
실시간 스트리밍 데이터를 처리합니다.

### 1. Kafka와 스트리밍
```python
# Kafka 스트리밍 예제
print("=== Kafka 스트리밍 ===")

# Kafka 프로듀서 예제
kafka_producer_example = '''
from kafka import KafkaProducer
import json
import time
import random

# Kafka 프로듀서 설정
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

# 데이터 생성 및 전송
def generate_sensor_data():
    """센서 데이터 생성"""
    return {
        'sensor_id': random.randint(1, 100),
        'temperature': random.uniform(20, 30),
        'humidity': random.uniform(40, 60),
        'timestamp': time.time()
    }

# 데이터 전송
for i in range(100):
    data = generate_sensor_data()
    producer.send('sensor_data', data)
    time.sleep(0.1)

producer.flush()
'''

print("Kafka 프로듀서:")
print(kafka_producer_example)

# Kafka 컨슈머 예제
kafka_consumer_example = '''
from kafka import KafkaConsumer
import json

# Kafka 컨슈머 설정
consumer = KafkaConsumer(
    'sensor_data',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# 데이터 수신
for message in consumer:
    data = message.value
    print(f"수신 데이터: {data}")
    
    # 데이터 처리
    if data['temperature'] > 25:
        print("경고: 온도가 높음!")
'''

print("\nKafka 컨슈머:")
print(kafka_consumer_example)
```

### 2. Spark Structured Streaming
```python
# Spark Structured Streaming 예제
spark_streaming_example = '''
# 스트리밍 DataFrame 생성
streaming_df = spark.readStream \\
    .format("kafka") \\
    .option("kafka.bootstrap.servers", "localhost:9092") \\
    .option("subscribe", "sensor_data") \\
    .load()

# 데이터 파싱
from pyspark.sql.functions import from_json, col

schema = "sensor_id INT, temperature DOUBLE, humidity DOUBLE, timestamp DOUBLE"

parsed_df = streaming_df.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

# 스트리밍 쿼리
query = parsed_df.writeStream \\
    .outputMode("append") \\
    .format("console") \\
    .option("truncate", "false") \\
    .start()

# 집계 스트리밍
agg_query = parsed_df \\
    .groupBy("sensor_id") \\
    .agg(
        count("*").alias("message_count"),
        mean("temperature").alias("avg_temperature"),
        mean("humidity").alias("avg_humidity")
    ) \\
    .writeStream \\
    .outputMode("complete") \\
    .format("console") \\
    .start()

# 쿼리 실행 대기
query.awaitTermination()
'''

print("\nSpark Structured Streaming:")
print(spark_streaming_example)
```

## 성능 최적화
대규모 데이터 처리 성능을 최적화합니다.

### 1. 파티셔닝 및 병렬화
```python
# 파티셔닝 전략
print("=== 성능 최적화 ===")

partitioning_strategies = {
    '해시 파티셔닝': {
        '설명': '해시 함수로 데이터 분산',
        '장점': '균등한 분산',
        '단점': '조인 시 셔플 필요',
        '적합': '조인이 많은 경우'
    },
    '레인지 파티셔닝': {
        '설명': '값 범위로 파티션',
        '장점': '범위 쿼리 빠름',
        '단점': '데이터 편향 가능',
        '적합': '범위 쿼리가 많은 경우'
    },
    '라운드 로빈': {
        '설명': '순차적으로 분산',
        '장점': '단순함',
        '단점': '데이터 편향',
        '적합': '작은 데이터셋'
    },
    '커스텀 파티셔닝': {
        '설명': '사용자 정의 로직',
        '장점': '최적화 가능',
        '단점': '복잡함',
        '적합': '특수한 요구사항'
    }
}

df_partitioning = pd.DataFrame(partitioning_strategies).T
print("파티셔닝 전략:")
print(df_partitioning)

# Dask 파티셔닝 예제
dask_partitioning_example = '''
# Dask DataFrame 파티셔닝
import dask.dataframe as dd

# CSV 파일 읽기 (자동 파티셔닝)
ddf = dd.read_csv('large_data_*.csv')

# 수동 파티셔닝
ddf_repartitioned = ddf.repartition(npartitions=8)

# 인덱스 기반 파티셔닝
ddf_indexed = ddf.set_index('timestamp').repartition(divisions=[...])

# 파티션 크기 기반 최적화
ddf_optimized = ddf.repartition(partition_size='100MB')
'''

print("\nDask 파티셔닝:")
print(dask_partitioning_example)
```

### 2. 캐싱 및 메모리 관리
```python
# 캐싱 전략
caching_strategies = {
    '메모리 캐싱': {
        '설명': '메모리에 데이터 저장',
        '장점': '가장 빠름',
        '단점': '메모리 제한',
        '사용': '자주 접근하는 데이터'
    },
    '디스크 캐싱': {
        '설명': '디스크에 데이터 저장',
        '장점': '대용량 가능',
        '단점': '느림',
        '사용': '중간 크기 데이터'
    },
    '하이브리드 캐싱': {
        '설명': '메모리+디스크',
        '장점': '균형',
        '단점': '복잡함',
        '사용': '일반적인 경우'
    },
    '분산 캐싱': {
        '설명': '여러 노드에 분산',
        '장점': '확장성',
        '단점': '네트워크 오버헤드',
        '사용': '클러스터 환경'
    }
}

df_caching = pd.DataFrame(caching_strategies).T
print("캐싱 전략:")
print(df_caching)

# Spark 캐싱 예제
spark_caching_example = '''
# DataFrame 캐싱
df.cache()  # 메모리 캐싱
df.persist()  # 기본 캐싱

# 다양한 캐싱 레벨
from pyspark.storagelevel import StorageLevel

df.persist(StorageLevel.MEMORY_ONLY)  # 메모리만
df.persist(StorageLevel.MEMORY_AND_DISK)  # 메모리+디스크
df.persist(StorageLevel.DISK_ONLY)  # 디스크만

# 캐싱 해제
df.unpersist()

# 캐싱 확인
df.is_cached
'''

print("\nSpark 캐싱:")
print(spark_caching_example)
```

### 3. 최적화 기법
```python
# 최적화 기법
optimization_techniques = {
    '조인 최적화': {
        '기법': '브로드캐스트 조인',
        '설명': '작은 테이블을 모든 노드에 복제',
        '효과': '네트워크 트래픽 감소',
        '적용': '작은 테이블 조인 시'
    },
    '필터 푸시다운': {
        '기법': '조기 필터링',
        '설명': '데이터 로드 전 필터링',
        '효과': '처리 데이터량 감소',
        '적용': 'WHERE 조건이 있을 때'
    },
    '컬럼 프루닝': {
        '기법': '필요한 컬럼만 선택',
        '설명': '불필요한 컬럼 제외',
        '효과': '메모리 사용량 감소',
        '적용': '일부 컬럼만 사용 시'
    },
    '배치 크기 최적화': {
        '기법': '적절한 배치 크기',
        '설명': '처리 단위 최적화',
        '효과': '처리 효율 향상',
        '적용': '대량 데이터 처리 시'
    }
}

df_optimization = pd.DataFrame(optimization_techniques).T
print("최적화 기법:")
print(df_optimization)

# 성능 모니터링
performance_monitoring = '''
# Dask 성능 모니터링
from dask.distributed import progress, Client

# 클라이언트 대시보드
client = Client()
print("대시보드:", client.dashboard_link)

# 진행률 모니터링
future = client.compute(computation)
progress(future)

# 성능 프로파일링
with dask.config.set(scheduler='processes', num_workers=4):
    result = computation.compute()

# Spark 성능 모니터링
# Spark UI: http://localhost:4040
# 실행 계획 확인
df.explain()

# 쿼리 최적화
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
'''

print("\n성능 모니터링:")
print(performance_monitoring)
```

## 실전 프로젝트: 대규모 로그 분석
대규모 로그 데이터를 분석하는 실전 프로젝트를 구현합니다.

### 1. 데이터 수집 및 전처리
```python
# 로그 데이터 분석 파이프라인
print("\n=== 실전 프로젝트: 대규모 로그 분석 ===")

class LogAnalysisPipeline:
    def __init__(self):
        self.processed_logs = []
    
    def generate_log_data(self, num_logs=100000):
        """로그 데이터 생성"""
        import random
        from datetime import datetime, timedelta
        
        log_levels = ['INFO', 'WARNING', 'ERROR', 'DEBUG']
        services = ['web', 'database', 'cache', 'queue']
        
        logs = []
        base_time = datetime.now()
        
        for i in range(num_logs):
            log_entry = {
                'timestamp': base_time + timedelta(seconds=i),
                'level': random.choice(log_levels),
                'service': random.choice(services),
                'message': f'Log message {i}',
                'user_id': random.randint(1, 1000),
                'response_time': random.uniform(0.1, 5.0),
                'status_code': random.choice([200, 201, 400, 404, 500])
            }
            logs.append(log_entry)
        
        return logs
    
    def preprocess_logs(self, logs):
        """로그 전처리"""
        df = pd.DataFrame(logs)
        
        # 시간 관련 피처 생성
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # 에러 로그 필터링
        error_logs = df[df['level'] == 'ERROR']
        
        # 느린 응답 필터링
        slow_responses = df[df['response_time'] > 2.0]
        
        return df, error_logs, slow_responses
    
    def analyze_logs(self, df):
        """로그 분석"""
        analysis = {}
        
        # 로그 레벨별 통계
        level_stats = df['level'].value_counts()
        analysis['level_distribution'] = level_stats.to_dict()
        
        # 서비스별 통계
        service_stats = df.groupby('service').agg({
            'response_time': ['mean', 'max', 'count'],
            'status_code': lambda x: (x >= 400).sum()
        })
        analysis['service_performance'] = service_stats.to_dict()
        
        # 시간대별 패턴
        hourly_pattern = df.groupby('hour').size()
        analysis['hourly_pattern'] = hourly_pattern.to_dict()
        
        # 에러율 계산
        error_rate = (df['level'] == 'ERROR').sum() / len(df) * 100
        analysis['error_rate'] = error_rate
        
        return analysis

# 파이프라인 실행
pipeline = LogAnalysisPipeline()

# 데이터 생성
print("로그 데이터 생성...")
logs = pipeline.generate_log_data(50000)
print(f"생성된 로그 수: {len(logs)}")

# 전처리
print("로그 전처리...")
df_logs, error_logs, slow_responses = pipeline.preprocess_logs(logs)

# 분석
print("로그 분석...")
analysis = pipeline.analyze_logs(df_logs)

# 결과 출력
print("\n=== 분석 결과 ===")
print(f"전체 로그 수: {len(df_logs)}")
print(f"에러 로그 수: {len(error_logs)}")
print(f"느린 응답 수: {len(slow_responses)}")
print(f"에러율: {analysis['error_rate']:.2f}%")

print("\n로그 레벨 분포:")
for level, count in analysis['level_distribution'].items():
    print(f"- {level}: {count}")

print("\n시간대별 패턴 (상위 5):")
sorted_hours = sorted(analysis['hourly_pattern'].items(), key=lambda x: x[1], reverse=True)[:5]
for hour, count in sorted_hours:
    print(f"- {hour}시: {count}건")
```

### 2. 분산 처리 확장
```python
# Dask를 사용한 분산 로그 분석
print("\n=== 분산 처리 확장 ===")

class DistributedLogAnalysis:
    def __init__(self):
        self.client = Client(n_workers=4)
    
    def distributed_analysis(self, log_files):
        """분산 로그 분석"""
        # 여러 파일에서 데이터 로드
        ddf = dd.read_csv(log_files, parse_dates=['timestamp'])
        
        # 분산 연산
        level_counts = ddf['level'].value_counts().compute()
        service_stats = ddf.groupby('service')['response_time'].agg(['mean', 'max']).compute()
        
        # 시계열 분석
        hourly_stats = ddf.groupby(ddf['timestamp'].dt.hour).size().compute()
        
        return {
            'level_counts': level_counts,
            'service_stats': service_stats,
            'hourly_stats': hourly_stats
        }
    
    def real_time_analysis(self, stream_data):
        """실시간 로그 분석"""
        # 실시간 데이터 처리 로직
        alerts = []
        
        for log_batch in stream_data:
            # 에러율 계산
            error_rate = (log_batch['level'] == 'ERROR').sum() / len(log_batch)
            
            # 알람 생성
            if error_rate > 0.1:  # 10% 이상 에러
                alerts.append({
                    'timestamp': pd.Timestamp.now(),
                    'type': 'HIGH_ERROR_RATE',
                    'value': error_rate
                })
            
            # 평균 응답 시간
            avg_response = log_batch['response_time'].mean()
            if avg_response > 3.0:  # 3초 이상
                alerts.append({
                    'timestamp': pd.Timestamp.now(),
                    'type': 'SLOW_RESPONSE',
                    'value': avg_response
                })
        
        return alerts

# 분산 분석 시뮬레이션
distributed_analyzer = DistributedLogAnalysis()

# 가상 로그 파일 생성
log_files = [f'logs_{i}.csv' for i in range(4)]
for i, file in enumerate(log_files):
    chunk_logs = logs[i*12500:(i+1)*12500]
    pd.DataFrame(chunk_logs).to_csv(file, index=False)

print(f"생성된 로그 파일: {log_files}")

# 분산 분석 실행 (개념적)
print("분산 로그 분석 실행...")
# result = distributed_analyzer.distributed_analysis(log_files)
# print("분산 분석 결과:", result)

print("분산 처리 구조:")
print("- 여러 파일 병렬 로드")
print("- 분산 집계 연산")
print("- 실시간 스트리밍 처리")
print("- 자동 알람 생성")

print("\n=== 대규모 데이터 처리 예제 완료! ===")
print("1. Dask 병렬 데이터 처리")
print("2. Apache Spark 분산 컴퓨팅")
print("3. 분산 파일 시스템 활용")
print("4. 스트리밍 데이터 처리")
print("5. 성능 최적화 기법")
print("6. 실전 로그 분석 프로젝트")
