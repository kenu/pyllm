import math
import random
from datetime import datetime


def main():
    print("pi:", math.pi)
    print("sqrt(16):", math.sqrt(16))

    random.seed(42)
    print("randint:", random.randint(1, 10))

    now = datetime.now()
    print("now:", now.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
