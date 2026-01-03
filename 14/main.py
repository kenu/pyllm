import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import json

print("=== 대규모 데이터 처리 예제 ===")

# 1. Dask 기초 개념
print("\n=== 1. Dask 기초 ===")

# Dask 임포트 (설치되지 않은 경우 대비)
try:
    import dask
    import dask.dataframe as dd
    import dask.array as da
    from dask.distributed import Client
    DASK_AVAILABLE = True
    print(f"Dask 버전: {dask.__version__}")
except ImportError:
    print("Dask가 설치되지 않았습니다. 시뮬레이션으로 진행합니다.")
    DASK_AVAILABLE = False

# 대용량 데이터 생성 (시뮬레이션)
def create_large_dataset(size=1_000_000):
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
df_pandas = create_large_dataset(100_000)  # 10만 개
pandas_time = time.time() - start_time
print(f"Pandas 생성 시간: {pandas_time:.2f}초")
print(f"데이터 크기: {df_pandas.shape}")

# Dask DataFrame 생성 (가능한 경우)
if DASK_AVAILABLE:
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
    
    # Pandas 복잡 연산
    start_time = time.time()
    pandas_complex = df_pandas.groupby('category').agg({
        'value': ['mean', 'std', 'count'],
        'id': 'nunique'
    })
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

# 2. Dask Array 사용
print("\n=== 2. Dask Array ===")

if DASK_AVAILABLE:
    # 대용량 배열 생성
    x = da.random.random((1000, 1000), chunks=(250, 250))
    y = da.random.random((1000, 1000), chunks=(250, 250))
    
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

# 3. Spark 개념 (시뮬레이션)
print("\n=== 3. Apache Spark (시뮬레이션) ===")

# Spark DataFrame 시뮬레이션
class SparkDataFrameSimulator:
    def __init__(self, data):
        self.data = data
        self.partitions = 4
    
    def show(self, n=20):
        """데이터 표시"""
        return self.data.head(n)
    
    def groupBy(self, column):
        """그룹화"""
        return SparkGroupBySimulator(self.data, column)
    
    def filter(self, condition):
        """필터링"""
        # 간단한 필터링 시뮬레이션
        return SparkDataFrameSimulator(self.data[self.data[column] == condition])
    
    def count(self):
        """개수 세기"""
        return len(self.data)
    
    def agg(self, operations):
        """집계 연산"""
        return operations

class SparkGroupBySimulator:
    def __init__(self, data, column):
        self.data = data
        self.column = column
    
    def agg(self, operations):
        """집계 연산"""
        grouped = self.data.groupby(self.column)
        
        result = {}
        for col, ops in operations.items():
            if isinstance(ops, list):
                for op in ops:
                    if op == 'mean':
                        result[f"{col}_{op}"] = grouped[col].mean()
                    elif op == 'std':
                        result[f"{col}_{op}"] = grouped[col].std()
                    elif op == 'count':
                        result[f"{col}_{op}"] = grouped[col].count()
            elif ops == 'nunique':
                result[f"{self.column}_{ops}"] = grouped[self.column].nunique()
        
        return SparkDataFrameSimulator(pd.DataFrame(result))

# Spark 시뮬레이션 실행
spark_data = [
    (1, "Alice", 25, "Engineering", 75000),
    (2, "Bob", 30, "Marketing", 65000),
    (3, "Charlie", 35, "Engineering", 80000),
    (4, "Diana", 28, "Sales", 55000),
    (5, "Eve", 32, "Marketing", 70000)
]

spark_df = SparkDataFrameSimulator(pd.DataFrame(spark_data, 
                                              columns=["id", "name", "age", "department", "salary"]))

print("Spark DataFrame (시뮬레이션):")
print(spark_df.show())

# 그룹화 연산
print("\n부서별 통계:")
dept_stats = spark_df.groupBy("department").agg({
    "age": ["mean"],
    "salary": ["mean", "count"],
    "id": "nunique"
})
print(dept_stats.show())

# 4. 분산 파일 시스템 개념
print("\n=== 4. 분산 파일 시스템 ===")

# HDFS 명령어 가이드
hdfs_commands = {
    '디렉토리 생성': 'hdfs dfs -mkdir /user/data',
    '파일 업로드': 'hdfs dfs -put local_file.csv /user/data/',
    '파일 목록': 'hdfs dfs -ls /user/data',
    '파일 내용': 'hdfs dfs -cat /user/data/file.csv',
    '파일 다운로드': 'hdfs dfs -get /user/data/file.csv local_copy.csv',
    '파일 삭제': 'hdfs dfs -rm /user/data/file.csv',
    '디렉토리 삭제': 'hdfs dfs -rm -r /user/data',
    '디스크 사용량': 'hdfs dfs -du -h /user/data'
}

print("HDFS 기본 명령어:")
for cmd, desc in hdfs_commands.items():
    print(f"{cmd}: {desc}")

# 파일 포맷 비교
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

# 5. 스트리밍 데이터 처리
print("\n=== 5. 스트리밍 데이터 처리 ===")

# Kafka 시뮬레이션
class KafkaSimulator:
    def __init__(self):
        self.topics = {}
        self.consumers = {}
    
    def create_topic(self, topic_name):
        """토픽 생성"""
        if topic_name not in self.topics:
            self.topics[topic_name] = []
    
    def send_message(self, topic_name, message):
        """메시지 전송"""
        if topic_name in self.topics:
            self.topics[topic_name].append({
                'message': message,
                'timestamp': datetime.now()
            })
    
    def consume_messages(self, topic_name, consumer_id):
        """메시지 수신"""
        if topic_name in self.topics:
            if consumer_id not in self.consumers:
                self.consumers[consumer_id] = 0
            
            start_idx = self.consumers[consumer_id]
            messages = self.topics[topic_name][start_idx:]
            self.consumers[consumer_id] = len(self.topics[topic_name])
            
            return messages
        return []

# Kafka 시뮬레이션 실행
kafka_sim = KafkaSimulator()
kafka_sim.create_topic('sensor_data')

# 센서 데이터 생성 및 전송
def generate_sensor_data():
    """센서 데이터 생성"""
    return {
        'sensor_id': np.random.randint(1, 100),
        'temperature': np.random.uniform(20, 30),
        'humidity': np.random.uniform(40, 60),
        'timestamp': datetime.now().isoformat()
    }

print("센서 데이터 전송...")
for i in range(10):
    data = generate_sensor_data()
    kafka_sim.send_message('sensor_data', data)
    time.sleep(0.1)

# 데이터 수신
print("\n데이터 수신:")
messages = kafka_sim.consume_messages('sensor_data', 'consumer1')
for msg in messages:
    data = msg['message']
    print(f"수신: 센서 {data['sensor_id']}, 온도 {data['temperature']:.1f}°C")

# 6. 성능 최적화
print("\n=== 6. 성능 최적화 ===")

# 파티셔닝 전략
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
    }
}

df_partitioning = pd.DataFrame(partitioning_strategies).T
print("파티셔닝 전략:")
print(df_partitioning)

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
    '분산 캐싱': {
        '설명': '여러 노드에 분산',
        '장점': '확장성',
        '단점': '네트워크 오버헤드',
        '사용': '클러스터 환경'
    }
}

df_caching = pd.DataFrame(caching_strategies).T
print("\n캐싱 전략:")
print(df_caching)

# 7. 실전 프로젝트: 대규모 로그 분석
print("\n=== 7. 실전 프로젝트: 대규모 로그 분석 ===")

class LogAnalysisPipeline:
    def __init__(self):
        self.processed_logs = []
    
    def generate_log_data(self, num_logs=10000):
        """로그 데이터 생성"""
        log_levels = ['INFO', 'WARNING', 'ERROR', 'DEBUG']
        services = ['web', 'database', 'cache', 'queue']
        
        logs = []
        base_time = datetime.now()
        
        for i in range(num_logs):
            log_entry = {
                'timestamp': base_time + timedelta(seconds=i),
                'level': np.random.choice(log_levels),
                'service': np.random.choice(services),
                'message': f'Log message {i}',
                'user_id': np.random.randint(1, 1000),
                'response_time': np.random.uniform(0.1, 5.0),
                'status_code': np.random.choice([200, 201, 400, 404, 500])
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
logs = pipeline.generate_log_data(5000)
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

# 8. 최적화 기법
print("\n=== 8. 최적화 기법 ===")

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

# 성능 모니터링 시뮬레이션
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'processing_time': [],
            'memory_usage': [],
            'cpu_usage': []
        }
    
    def record_metric(self, metric_name, value):
        """메트릭 기록"""
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
    
    def get_summary(self):
        """메트릭 요약"""
        summary = {}
        for metric, values in self.metrics.items():
            if values:
                summary[metric] = {
                    'avg': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        return summary

# 성능 모니터링 시뮬레이션
monitor = PerformanceMonitor()

# 처리 시간 기록
for i in range(10):
    processing_time = np.random.uniform(0.5, 2.0)
    memory_usage = np.random.uniform(40, 80)
    cpu_usage = np.random.uniform(20, 60)
    
    monitor.record_metric('processing_time', processing_time)
    monitor.record_metric('memory_usage', memory_usage)
    monitor.record_metric('cpu_usage', cpu_usage)

# 성능 요약
performance_summary = monitor.get_summary()
print("\n성능 모니터링 요약:")
for metric, stats in performance_summary.items():
    print(f"{metric}:")
    print(f"  평균: {stats['avg']:.2f}")
    print(f"  범위: {stats['min']:.2f} - {stats['max']:.2f}")

# Dask 클라이언트 정리
if DASK_AVAILABLE:
    client.close()

print("\n=== 대규모 데이터 처리 예제 완료! ===")
print("1. Dask 병렬 데이터 처리")
print("2. Apache Spark 분산 컴퓨팅")
print("3. 분산 파일 시스템 활용")
print("4. 스트리밍 데이터 처리")
print("5. 성능 최적화 기법")
print("6. 실전 로그 분석 프로젝트")
