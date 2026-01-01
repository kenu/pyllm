from typing import Optional


# 리스트에서 첫 번째 짝수를 찾는 함수
# Optional[int]: int 또는 None을 반환할 수 있음
def find_first_even(nums: list[int]) -> Optional[int]:
    for n in nums:
        if n % 2 == 0:  # 짝수인지 확인
            return n     # 첫 번째 짝수 반환
    return None        # 짝수가 없으면 None 반환


def main():
    # 짝수가 없는 경우
    print(find_first_even([1, 3, 5]))  # None 출력
    
    # 짝수가 있는 경우
    print(find_first_even([1, 4, 5]))  # 4 출력


if __name__ == "__main__":
    main()
