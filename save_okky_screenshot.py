import base64
import os
from pathlib import Path

MCP_OUTPUT_PATH = "/var/folders/kk/vf0qzlwj339cwt0mlcdclh580000gn/T/windsurf/mcp_output_694a2e814e982164.txt"


def main() -> None:
    p = Path(MCP_OUTPUT_PATH)
    if not p.exists():
        raise SystemExit(f"Not found: {p}")

    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 2:
        raise SystemExit(f"Unexpected format (need >=2 lines): {p}")

    data_line = lines[1].strip()
    prefix = "data:image/png;base64,"
    if prefix not in data_line:
        raise SystemExit("No base64 PNG data line found")

    b64 = data_line.split(prefix, 1)[1]
    png_bytes = base64.b64decode(b64)

    out_path = Path(os.path.expanduser("~/Desktop/okky_screenshot.png"))
    out_path.write_bytes(png_bytes)
    print(str(out_path))


if __name__ == "__main__":
    main()
