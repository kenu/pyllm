# MongoDB 연동 (설치되지 않은 경우 시뮬레이션)
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
                        group_key = str(doc.get(group_spec['_id'], 'default'))
                    else:
                        group_key = 'all'

                    if group_key not in grouped:
                        grouped[group_key] = {}
                        # 집계 필드 초기화
                        for field, expr in group_spec.items():
                            if field != '_id':
                                if expr.startswith('$'):
                                    field_name = expr[1:]
                                    if expr == '$sum':
                                        grouped[group_key][field] = 0
                                    elif expr == '$avg':
                                        grouped[group_key][field] = 0
                                        grouped[group_key][field + '_count'] = 0
                                    elif expr == '$max':
                                        grouped[group_key][field] = float('-inf')
                                    elif expr == '$min':
                                        grouped[group_key][field] = float('inf')

                    # 값 집계
                    for field, expr in group_spec.items():
                        if field != '_id':
                            if expr.startswith('$'):
                                field_name = expr[1:]
                                if field_name in doc:
                                    value = doc[field_name]
                                    if expr == '$sum':
                                        grouped[group_key][field] += value
                                    elif expr == '$avg':
                                        grouped[group_key][field] += value
                                        grouped[group_key][field + '_count'] += 1
                                    elif expr == '$max':
                                        grouped[group_key][field] = max(grouped[group_key][field], value)
                                    elif expr == '$min':
                                        grouped[group_key][field] = min(grouped[group_key][field], value)

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

# MongoDB 연결 또는 시뮬레이션
if MONGO_AVAILABLE:
    try:
        client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=1000)
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
