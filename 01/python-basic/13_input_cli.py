def main():
    # 사용자로부터 이름 입력 받기
    name = input("이름 입력: ").strip()  # 앞뒤 공백 제거

    # 나이 입력 받기 (숫자 검증 포함)
    while True:
        raw = input("나이 입력(숫자): ").strip()
        if raw.isdigit():      # 숫자인지 확인
            age = int(raw)     # 문자열을 정수로 변환
            break              # 유효한 입력이면 루프 탈출
        print("숫자만 넣어라")  # 오류 메시지

    # 나이에 따른 그룹 분류
    if age >= 20:
        group = "성인"     # 20세 이상
    else:
        group = "미성년"   # 20세 미만

    # 결과 출력
    print(f"{name} / {age} / {group}")


if __name__ == "__main__":
    main()
