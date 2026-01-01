def main():
    nums = [1, 2, 3]
    nums.append(4)
    nums[0] = 10
    print("list:", nums)

    t = (1, 2, 3)
    print("tuple:", t)

    st = {1, 2, 2, 3}
    st.add(4)
    print("set:", st)

    m = {"a": 1, "b": 2}
    m["c"] = 3
    print("dict keys:", list(m.keys()))
    print("dict values:", list(m.values()))

    items = [{"name": "A", "score": 80}, {"name": "B", "score": 95}]
    best = max(items, key=lambda x: x["score"])
    print("best:", best)


if __name__ == "__main__":
    main()
