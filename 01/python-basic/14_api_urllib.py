import json
import urllib.request


# urllib을 사용하여 JSON 데이터 가져오는 함수
def fetch_json(url: str) -> dict:
    # HTTP 요청 객체 생성
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "python-basic/1.0",  # 클라이언트 정보
            "Accept": "application/json",      # JSON 응답 요청
        },
        method="GET",  # GET 방식 요청
    )

    # 요청 보내고 응답 받기 (타임아웃 10초)
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = resp.read().decode("utf-8")  # 바이너리 데이터를 문자열로 변환
        return json.loads(data)              # JSON 문자열을 딕셔너리로 변환


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
