def main():
    s = "Python"

    print(s[0], s[-1])
    print(s[0:3])
    print(s.upper())
    print("-".join(["a", "b", "c"]))

    name = "Kim"
    age = 20
    print("name=%s age=%d" % (name, age))
    print("name={} age={}".format(name, age))
    print(f"name={name} age={age}")


if __name__ == "__main__":
    main()
