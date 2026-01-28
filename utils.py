import os
from pathlib import Path
from dotenv import load_dotenv

def load_env_vars():
    """프로젝트 루트의 .env 파일을 로드합니다."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # .env가 없으면 .env.example을 참고하라는 메시지 출력
        example_path = Path(__file__).parent / ".env.example"
        if example_path.exists():
            print(f"알림: .env 파일이 없습니다. {example_path}를 복사하여 .env 파일을 생성하세요.")

def get_db_config():
    """환경 변수에서 데이터베이스 설정 정보를 가져옵니다."""
    load_env_vars()
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 3306)),
        'database': os.getenv('DB_NAME', 'pyllmdb'),
        'user': os.getenv('DB_USER', 'pyllmuser'),
        'password': os.getenv('DB_PASSWORD', 'pyllmpassword')
    }
