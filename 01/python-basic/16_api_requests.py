import requests


# requests 라이브러리를 사용하여 JSON 데이터 가져오는 함수
def fetch_json(url: str) -> dict:
    # GET 요청 보내기
    resp = requests.get(
        url,
        headers={
            "User-Agent": "python-basic/1.0",  # 클라이언트 정보
            "Accept": "application/json",      # JSON 응답 요청
        },
        timeout=10,  # 타임아웃 10초
    )
    resp.raise_for_status()  # HTTP 에러가 있으면 예외 발생
    return resp.json()       # JSON 응답을 딕셔너리로 변환하여 반환


def main():
    # GitHub API에서 CPython 저장소 정보 가져오기
    url = "https://api.github.com/repos/python/cpython"
    obj = fetch_json(url)

    # 필요한 정보 출력
    print("full_name:", obj.get("full_name"))           # 저장소 전체 이름
    print("stargazers_count:", obj.get("stargazers_count"))  # 스타 수
    print("open_issues_count:", obj.get("open_issues_count"))  # 열린 이슈 수


if __name__ == "__main__":
    main()
