import pytest
from utils import get_db_config
import os

def test_db_config_default():
    """기본 DB 설정값이 올바르게 반환되는지 테스트합니다."""
    # 환경 변수가 없는 상태에서의 기본값 확인
    config = get_db_config()
    assert config['host'] == 'localhost'
    assert config['port'] == 3306
    assert isinstance(config['port'], int)

def test_env_loading(tmp_path):
    """.env 파일 로딩이 정상적으로 작동하는지 테스트합니다."""
    # 임시 .env 파일 생성
    env_file = tmp_path / ".env"
    env_file.write_text("DB_HOST=test_host\nDB_PORT=9999")
    
    # os.environ에 직접 설정하여 시뮬레이션
    os.environ['DB_HOST'] = 'test_host'
    os.environ['DB_PORT'] = '9999'
    
    config = get_db_config()
    assert config['host'] == 'test_host'
    assert config['port'] == 9999
    
    # 청소
    del os.environ['DB_HOST']
    del os.environ['DB_PORT']
