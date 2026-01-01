def safe_div(a, b):
    try:
        return a / b
    except ZeroDivisionError as e:
        return f"error: {e}"
    finally:
        pass


def parse_int(s):
    try:
        return int(s)
    except ValueError:
        raise ValueError(f"not an int: {s}")


def main():
    print(safe_div(10, 2))
    print(safe_div(10, 0))

    try:
        print(parse_int("123"))
        print(parse_int("abc"))
    except ValueError as e:
        print("caught:", e)


if __name__ == "__main__":
    main()
