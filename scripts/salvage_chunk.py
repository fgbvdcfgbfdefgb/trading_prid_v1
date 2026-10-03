#!/usr/bin/env python3
"""One-off repair tool: reads a (possibly corrupted, due to being killed
mid-write) gzip CSV chunk line-by-line, keeps everything up to the first
decode/parse error, and writes the good prefix out as a clean new file."""
import csv
import gzip
import sys

src, dst = sys.argv[1], sys.argv[2]
good_rows = []
header = None
try:
    with gzip.open(src, "rt", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) != 7:
                break
            try:
                float(row[0]); float(row[1]); float(row[2])
            except ValueError:
                break
            good_rows.append(row)
except Exception as e:
    print(f"[info] stopped reading {src} at row {len(good_rows)} due to: {e}")

with gzip.open(dst, "wt", newline="") as f:
    w = csv.writer(f)
    w.writerow(header or ["price", "volume", "time", "side", "order_type", "misc", "trade_id"])
    w.writerows(good_rows)

print(f"[salvage] {src} -> {dst}: kept {len(good_rows)} good rows")
