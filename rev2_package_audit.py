# -*- coding: utf-8 -*-
"""Data fingerprints for the reproducibility package.

One output, outputs/results/rev2_data_fingerprints.csv: the md5, the byte
size and the row count of every raw data file this work reads, computed here
rather than copied from anywhere, so a reader can confirm they downloaded the
same files. The nine Olist files are fingerprinted individually, and each
row carries the publisher's licence and this package's redistribution
stance - nothing under a non-commercial licence is redistributed.

It reads; it changes nothing.

Usage:  python -m rev2_package_audit
"""
import hashlib
import io
import os
import sys

import pandas as pd

import config

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
R = config.RESULTS_PATH

# path, dataset, licence as the publisher states it, redistribution stance
DATA = [
    ("data/Online Retail.xlsx", "UCI Online Retail",
     "UCI ML Repository, CC BY 4.0", "link only, not redistributed"),
    ("data/online_retail_II.xlsx", "Online Retail II",
     "UCI ML Repository, CC BY 4.0", "link only, not redistributed"),
    ("data/superstore/superstore.csv", "Superstore",
     "Tableau sample workbook content, fictional teaching data",
     "link only, not redistributed"),
    ("data/bakery/bakery.csv", "Bakery (Carvalho & Maciel 2026)",
     "open-access supplementary material, CC BY", "link only, not redistributed"),
]
OLIST_DIR = "data/archive"
OLIST_LICENCE = ("Kaggle: Olist Brazilian E-Commerce, CC BY-NC-SA 4.0 - "
                 "NON-COMMERCIAL and share-alike")


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def rows_of(path):
    try:
        if path.lower().endswith((".xlsx", ".xls")):
            return "not counted (spreadsheet; sheet-dependent)"
        with open(path, "rb") as f:
            return str(sum(1 for _ in f) - 1) + " data rows"
    except OSError as e:
        return "unreadable: %s" % e


print("=" * 108)
print("1. DATA FINGERPRINTS")
print("=" * 108)
fp = []
for path, name, lic, stance in DATA:
    if not os.path.exists(path):
        print("  MISSING  %s" % path)
        fp.append(dict(Dataset=name, Path=path, Exists=False, MD5="", Bytes=0,
                       Rows="", Licence=lic, Redistribution=stance))
        continue
    h, size = md5(path), os.path.getsize(path)
    print("  %-36s %s  %12d bytes  %s" % (os.path.basename(path), h, size,
                                          rows_of(path)))
    fp.append(dict(Dataset=name, Path=path, Exists=True, MD5=h, Bytes=size,
                   Rows=rows_of(path), Licence=lic, Redistribution=stance))

if os.path.isdir(OLIST_DIR):
    print("\n  Olist, one fingerprint per file (%s):" % OLIST_LICENCE)
    for f in sorted(os.listdir(OLIST_DIR)):
        p = os.path.join(OLIST_DIR, f)
        if not os.path.isfile(p):
            continue
        h, size = md5(p), os.path.getsize(p)
        print("    %-44s %s  %10d bytes" % (f, h, size))
        fp.append(dict(Dataset="Olist: %s" % f, Path=p, Exists=True, MD5=h,
                       Bytes=size, Rows=rows_of(p), Licence=OLIST_LICENCE,
                       Redistribution="link only, not redistributed - "
                                      "non-commercial licence"))
pd.DataFrame(fp).to_csv(os.path.join(R, "rev2_data_fingerprints.csv"), index=False)

print()
print("written: %s" % os.path.join(R, "rev2_data_fingerprints.csv"))
