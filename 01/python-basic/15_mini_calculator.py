# 기본 사칙연산을 수행하는 계산기 함수
def calc(a: float, op: str, b: float) -> float:
    if op == "+":
        return a + b  # 덧셈
    if op == "-":
        return a - b  # 뺄셈
    if op == "*":
        return a * b  # 곱셈
    if op == "/":
        if b == 0:
            raise ZeroDivisionError("0으로 나누기 금지")  # 0으로 나누기 에러
        return a / b  # 나눗셈
    raise ValueError(f"지원 안 하는 연산자: {op}")  # 지원하지 않는 연산자 에러


def main():
    print("미니 계산기: 예) 3 + 4")
    print("끝내려면 quit")

    # 무한 루프로 사용자 입력 처리
    while True:
        line = input("> ").strip()  # 사용자 입력 받기
        if not line:                  # 빈 입력이면 다시 입력
            continue
        if line.lower() in {"q", "quit", "exit"}:  # 종료 명령어
            break

        # 입력을 공백으로 분리
        parts = line.split()
        if len(parts) != 3:  # 형식이 맞지 않으면
            print("형식: 숫자 연산자 숫자 (예: 3 + 4)")
            continue

        # 각 부분 분리
        a_s, op, b_s = parts
        try:
            a = float(a_s)  # 첫 번째 숫자로 변환
            b = float(b_s)  # 두 번째 숫자로 변환
            result = calc(a, op, b)  # 계산 수행
            print("=", result)      # 결과 출력
        except Exception as e:  # 에러 처리
            print("에러:", e)


if __name__ == "__main__":
    main()
