# Homework - 대규모 데이터 처리

## 이 homework의 의미
빅데이터를 효율적으로 처리하기 위한 분산 컴퓨팅 기술을 익히는 과제야. Spark, Dask 등을 활용해 테라바이트급 데이터를 분석하는 실무 능력을 키우는 거지.

## 관련 정보 사이트
1. **Apache Spark Documentation**: https://spark.apache.org/docs/latest/
2. **Dask Tutorial**: https://tutorial.dask.org/

## 프로세스 진행 논리적 흐름
1. **데이터 파티셔닝** → 데이터를 여러 노드에 분산
2. **병렬 처리** → 맵리듀스 패턴 적용
3. **메모리 관리** → 효율적인 리소스 사용
4. **집계 및 조인** → 분산 환경에서 연산 수행
5. **결과 저장** → HDFS, S3 등에 저장

## 권장 코드
```python
# PySpark 예제
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, sum

# Spark 세션 생성
spark = SparkSession.builder \
    .appName("BigDataProcessing") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# 대용량 데이터 로드 (예: CSV)
df = spark.read.csv("large_dataset.csv", header=True, inferSchema=True)

# 데이터 탐색
print(f"총 레코드 수: {df.count():,}")
df.printSchema()

# 데이터 처리
result = df.groupBy("category") \
    .agg(
        count("*").alias("count"),
        avg("price").alias("avg_price"),
        sum("quantity").alias("total_quantity")
    ) \
    .orderBy(col("total_quantity").desc())

# 결과 출력
result.show(10)

# 결과 저장
result.write.mode("overwrite").parquet("output/aggregated_data")

# Dask 예제 (Pandas 대안)
import dask.dataframe as dd

# 대용량 CSV 로드
ddf = dd.read_csv("large_dataset_*.csv")

# 병렬 처리
result = ddf.groupby('category').agg({
    'price': ['mean', 'sum'],
    'quantity': 'sum'
}).compute()

print(result)

spark.stop()
```

```python
# 스트리밍 데이터 처리
from pyspark.sql import SparkSession
from pyspark.sql.functions import window, col

spark = SparkSession.builder \
    .appName("StreamProcessing") \
    .getOrCreate()

# 스트리밍 소스 (예: Kafka)
stream_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "events") \
    .load()

# 윈도우 집계
windowed = stream_df \
    .groupBy(window(col("timestamp"), "1 minute")) \
    .count()

# 스트림 출력
query = windowed.writeStream \
    .outputMode("complete") \
    .format("console") \
    .start()

query.awaitTermination()
```

## 관련 업계 회사
1. **Databricks** - Apache Spark 기반 통합 분석 플랫폼
2. **Cloudera** - 하둡 및 빅데이터 솔루션
3. **Confluent** - Apache Kafka 스트리밍 플랫폼
