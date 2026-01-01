def main():
    n = 7

    if n % 2 == 0:
        print("even")
    elif n % 3 == 0:
        print("divisible by 3")
    else:
        print("other")

    total = 0
    for i in range(1, 6):
        total += i
    print("sum 1..5:", total)

    i = 0
    while i < 3:
        print("while:", i)
        i += 1

    for i in range(10):
        if i == 3:
            continue
        if i == 6:
            break
        print("loop:", i)


if __name__ == "__main__":
    main()
