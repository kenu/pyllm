def main():
    squares = [x * x for x in range(1, 6)]
    evens = [x for x in range(10) if x % 2 == 0]
    matrix = [[i * 3 + j for j in range(3)] for i in range(3)]

    m = {"a": 1, "b": 2}
    swapped = {v: k for k, v in m.items()}

    s = {x % 3 for x in range(10)}

    print("squares:", squares)
    print("evens:", evens)
    print("matrix:", matrix)
    print("swapped:", swapped)
    print("set:", s)


if __name__ == "__main__":
    main()
