#!/usr/bin/env python3

from pathlib import Path


MODEL = """layers 2
layer 4 6 relu
weights
0.50 -0.20 0.10 0.00
-0.30 0.80 0.20 -0.10
0.70 0.10 -0.40 0.30
0.20 0.20 0.50 -0.50
-0.60 0.40 0.30 0.20
0.10 -0.70 0.60 0.40
bias
0.10 -0.20 0.05 0.00 0.15 -0.05
layer 6 3 linear
weights
0.40 -0.30 0.20 0.70 -0.10 0.50
-0.20 0.60 -0.50 0.10 0.30 -0.40
0.30 0.20 0.40 -0.60 0.50 0.10
bias
0.05 -0.10 0.20
"""


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "models" / "tiny_mlp.txt"
    path.write_text(MODEL, encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
