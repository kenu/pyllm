from pathlib import Path


def main():
    base = Path(__file__).resolve().parent
    p = base / "sample.txt"

    p.write_text("line1\nline2\n", encoding="utf-8")

    text = p.read_text(encoding="utf-8")
    print(text.strip())

    lines = p.read_text(encoding="utf-8").splitlines()
    print("lines:", lines)

    p.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
