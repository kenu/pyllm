.PHONY: help install clean run-latest test

help:
	@echo "사용 가능한 명령:"
	@echo "  make install      - uv를 사용하여 모든 의존성 설치"
	@echo "  make clean        - __pycache__ 및 임시 파일 삭제"
	@echo "  make run-latest   - 가장 최근에 수정된 파이썬 스크립트 실행"
	@echo "  make test         - pytest를 사용하여 테스트 실행"

install:
	@if command -v uv > /dev/null; then \
		uv sync; \
	else \
		pip install -r 08/requirements.txt; \
	fi

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.log" -delete

run-latest:
	@python3 skills/recent_files.py

test:
	pytest .
