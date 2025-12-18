import mysql.connector
from mysql.connector import Error
import time

def connect_to_db():
    """MariaDB 데이터베이스에 연결 시도 (재시도 로직 포함)"""
    max_retries = 5
    retry_delay = 5
    
    for i in range(max_retries):
        try:
            connection = mysql.connector.connect(
                host='localhost',
                port=3306,
                database='pyllmdb',
                user='pyllmuser',
                password='pyllmpassword'
            )
            if connection.is_connected():
                return connection
        except Error as e:
            print(f"연결 실패 ({i+1}/{max_retries}): {e}")
            if i < max_retries - 1:
                print(f"{retry_delay}초 후 다시 시도합니다...")
                time.sleep(retry_delay)
            else:
                print("최대 재시도 횟수를 초과했습니다.")
                return None

def create_table(connection):
    """테이블 생성"""
    try:
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(100) NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("테이블 생성 성공")
    except Error as e:
        print(f"테이블 생성 오류: {e}")

def insert_user(connection, name, email):
    """사용자 추가 (Create)"""
    try:
        cursor = connection.cursor()
        query = "INSERT INTO users (name, email) VALUES (%s, %s)"
        cursor.execute(query, (name, email))
        connection.commit()
        print(f"사용자 추가 성공: {name}")
    except Error as e:
        print(f"사용자 추가 오류: {e}")

def get_users(connection):
    """사용자 목록 조회 (Read)"""
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users")
        rows = cursor.fetchall()
        print("\n--- 사용자 목록 ---")
        for row in rows:
            print(row)
        print("------------------\n")
    except Error as e:
        print(f"조회 오류: {e}")

def update_user_email(connection, name, new_email):
    """사용자 이메일 수정 (Update)"""
    try:
        cursor = connection.cursor()
        query = "UPDATE users SET email = %s WHERE name = %s"
        cursor.execute(query, (new_email, name))
        connection.commit()
        print(f"사용자 수정 성공: {name} -> {new_email}")
    except Error as e:
        print(f"수정 오류: {e}")

def delete_user(connection, name):
    """사용자 삭제 (Delete)"""
    try:
        cursor = connection.cursor()
        query = "DELETE FROM users WHERE name = %s"
        cursor.execute(query, (name,))
        connection.commit()
        print(f"사용자 삭제 성공: {name}")
    except Error as e:
        print(f"삭제 오류: {e}")

def main():
    conn = connect_to_db()
    
    if conn and conn.is_connected():
        # 1. 테이블 생성
        create_table(conn)
        
        # 2. 데이터 삽입 (C)
        insert_user(conn, "Alice", "alice@example.com")
        insert_user(conn, "Bob", "bob@example.com")
        
        # 3. 데이터 조회 (R)
        get_users(conn)
        
        # 4. 데이터 수정 (U)
        update_user_email(conn, "Alice", "alice_new@example.com")
        get_users(conn)
        
        # 5. 데이터 삭제 (D)
        delete_user(conn, "Bob")
        get_users(conn)
        
        conn.close()
        print("연결 종료")

if __name__ == "__main__":
    main()
