import os
import subprocess
from pathlib import Path

def find_latest_python_file(root_dir):
    """프로젝트 내에서 가장 최근에 수정된 .py 파일을 찾습니다."""
    latest_file = None
    latest_time = 0
    
    # 제외할 디렉토리
    exclude_dirs = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache'}
    
    for root, dirs, files in os.walk(root_dir):
        # 제외 디렉토리 건너뛰기
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.endswith('.py') and file != 'recent_files.py':
                file_path = Path(root) / file
                mtime = file_path.stat().st_mtime
                if mtime > latest_time:
                    latest_time = mtime
                    latest_file = file_path
                    
    return latest_file

def main():
    root_dir = Path(__file__).parent.parent
    latest = find_latest_python_file(root_dir)
    
    if latest:
        rel_path = latest.relative_to(root_dir)
        print(f"가장 최근에 수정된 파일: {rel_path}")
        confirm = input("이 파일을 실행할까요? (Y/n): ").lower()
        
        if confirm in ('', 'y', 'yes'):
            print(f"실행 중: python3 {rel_path}")
            print("-" * 30)
            try:
                subprocess.run(['python3', str(latest)], check=True)
            except subprocess.CalledProcessError as e:
                print(f"\n오류 발생 (종료 코드 {e.returncode})")
            except KeyboardInterrupt:
                print("\n실행이 중단되었습니다.")
    else:
        print("파이썬 파일을 찾을 수 없습니다.")

if __name__ == "__main__":
    main()
