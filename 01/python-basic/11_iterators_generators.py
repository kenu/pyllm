# 제너레이터 함수: n부터 1까지 카운트다운
def countdown(n):
    while n > 0:
        yield n  # 현재 값을 반환하고 함수 실행을 일시 정지
        n -= 1


# 커스텀 이터레이터 클래스: range와 유사한 동작
class RangeLike:
    def __init__(self, start, end):
        self.start = start  # 시작값
        self.end = end      # 끝값

    def __iter__(self):
        cur = self.start
        while cur < self.end:
            yield cur  # 현재 값을 반환
            cur += 1   # 다음 값으로 이동


def main():
    # 제너레이터 사용 예시
    for x in countdown(3):
        print("countdown:", x)

    # 커스텀 이터레이터 사용 예시
    for x in RangeLike(2, 5):
        print("rangelike:", x)

    # 이터레이터 직접 사용
    it = iter([10, 20])  # 리스트에서 이터레이터 생성
    print(next(it))      # 다음 요소 출력: 10
    print(next(it))      # 다음 요소 출력: 20


if __name__ == "__main__":
    main()
