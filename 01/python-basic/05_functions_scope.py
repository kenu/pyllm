def add(a, b=0):
    return a + b


def make_adder(x):
    def inner(y):
        return x + y

    return inner


def main():
    print(add(3, 4))
    print(add(10))

    f = make_adder(5)
    print(f(2))

    def varargs(*args, **kwargs):
        return args, kwargs

    print(varargs(1, 2, a=3, b=4))


if __name__ == "__main__":
    main()
