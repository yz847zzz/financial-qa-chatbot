# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_pipeline.processing.extractor import make_extractor

path = Path("E:/emo/workspace/pintrade/data/filings/AAPL/10-K/2023-11-03_000032019323000106/aapl-20230930.htm")
ext = make_extractor(path)
sections = ext.extract_sections()

print("All sections and their lengths:")
for k, v in sections.items():
    safe_k = k.encode("ascii", errors="replace").decode()
    safe_v = v[:80].encode("ascii", errors="replace").decode()
    print(f"  {len(v):>8} chars  {safe_k!r:30}  {safe_v!r}")
