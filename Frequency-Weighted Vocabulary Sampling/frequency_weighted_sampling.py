#!/usr/bin/env python3

import argparse
from pathlib import Path
import math
import random
import shutil
from tqdm import tqdm

def load_kmer_freqs(path):
    freqs = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            # k-mer tokens are everything except last
            *kmer_tokens, cnt_str = parts
            try:
                cnt = int(cnt_str)
            except:
                print("BAD LINE:", parts)
                continue

            kmer = " ".join(kmer_tokens)
            freqs[kmer] = cnt
    return freqs


def kmer_iter(seq, k=5):
    L = len(seq)
    for i in range(L - k + 1):
        yield seq[i:i+k]

def score_file(fp, kmer_weights, k=5):
    kmers_seen = set()
    with open(fp) as f:
        for line in f:
            seq = line.strip()
            if not seq:
                continue
            for km in kmer_iter(seq, k):
                if km in kmer_weights:
                    kmers_seen.add(km)
    return sum(kmer_weights[km] for km in kmers_seen)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--kmer_vocab", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--fraction", type=float, required=True,
                        help="Fraction of files to keep (e.g., 0.1 for 10%)")
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--eps", type=float, default=1.0)
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading k-mer frequencies...")
    freqs = load_kmer_freqs(args.kmer_vocab)

    print(f"[INFO] Loaded {len(freqs)} 5-mers.")
    print("[INFO] Computing rarity weights...")

    kmer_weights = {k: 1.0 / ((v + args.eps)**args.alpha) for k, v in freqs.items()}

    files = sorted(train_dir.glob("*.txt"))
    N = len(files)
    keep = int(N * args.fraction)

    print(f"[INFO] Scoring {N} files (keeping top {keep})...")

    scored = []
    for fp in tqdm(files, desc="Scoring"):
        s = score_file(fp, kmer_weights, k=5)
        scored.append((fp, s))

    scored.sort(key=lambda x: x[1], reverse=True)
    selected = scored[:keep]

    print(f"[INFO] Copying {len(selected)} files to {out_dir}")

    for fp, _ in tqdm(selected, desc="Copying"):
        shutil.copy2(fp, out_dir / fp.name)

    print("[DONE] Frequency-weighted sampling complete.")
    print(f"[RESULT] Subset written to: {out_dir}")

if __name__ == "__main__":
    main()
