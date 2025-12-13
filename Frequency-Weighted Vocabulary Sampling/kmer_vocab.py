#!/usr/bin/env python3

import argparse
from pathlib import Path
from collections import Counter
from tqdm import tqdm

def kmer_iter(seq, k=5):
    L = len(seq)
    for i in range(L - k + 1):
        yield seq[i:i+k]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True, type=str)
    parser.add_argument("--out_file", required=True, type=str)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    files = sorted(train_dir.glob("*.txt"))

    print(f"[INFO] Found {len(files)} training files")
    print(f"[INFO] Building {args.k}-mer vocabulary...")

    counter = Counter()

    for fp in tqdm(files, desc="Scanning train_prefix"):
        with fp.open() as f:
            for line in f:
                seq = line.strip()
                if not seq:
                    continue
                for km in kmer_iter(seq, args.k):
                    counter[km] += 1

    print(f"[INFO] Writing output to {args.out_file}")

    with open(args.out_file, "w") as out:
        for kmer, count in counter.most_common():
            out.write(f"{kmer}\t{count}\n")

    print(f"[DONE] Wrote {len(counter)} unique 5-mers.")

if __name__ == "__main__":
    main()
