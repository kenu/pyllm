def main():
    a = 10
    b = 3.14
    c = "hello"
    d = True
    e = None

    print(type(a), a)
    print(type(b), b)
    print(type(c), c)
    print(type(d), d)
    print(type(e), e)

    x, y = 1, 2
    x, y = y, x
    print("swap:", x, y)


if __name__ == "__main__":
    main()
