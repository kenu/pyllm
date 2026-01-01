from dataclasses import dataclass


class Counter:
    def __init__(self, start=0):
        self.value = start

    def inc(self, step=1):
        self.value += step
        return self.value


@dataclass
class User:
    name: str
    age: int


def main():
    c = Counter(10)
    print(c.inc())
    print(c.inc(5))

    u = User(name="Kim", age=20)
    print(u)


if __name__ == "__main__":
    main()
