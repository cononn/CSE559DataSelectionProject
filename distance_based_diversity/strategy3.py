import argparse
import glob
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances


def farthest_first(X, k, metric="cosine", random_state=42):



    # convert to numpy-ish indexing; pairwise_distances handles sparse matrices
    n_samples = X.shape[0]
    if k >= n_samples:
        
        return list(range(n_samples))
    

    rng= np.random.RandomState(random_state)
    first = rng.randint(0, n_samples)
    
    selected = [first]

    # distance from all points to the first selected point
    min_dist =pairwise_distances(X, X[first],metric=metric).ravel()

    for _ in range(1, k):
        # pick the point with maximum distance to its nearest selected point
        
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        # update min distance with distance to the newly selected point
        
        d = pairwise_distances(X, X[next_idx], metric=metric).ravel()
        
        min_dist = np.minimum(min_dist, d)

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Strategy 3: distance-based diversity selection"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="subsets/strategy3_test100",
        help="Directory where selected files will be copied",
    )
    
    parser.add_argument(
        "--select_fraction",
        type=float,
        default=0.1,
        help="Fraction of files to select (0 < f <= 1).",
    )
    
    
    parser.add_argument(
        "--max_features",
        type=int,
        default=5000,
        help="Max TF-IDF features",
    )
    
    
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    
    args = parser.parse_args()

    file_paths = sorted(glob.glob(args.input_glob))
    
    
    if not file_paths:
        raise ValueError(f"No files found for pattern: {args.input_glob}")

    n_files = len(file_paths)
    n_select = max(1, int(round(args.select_fraction* n_files)))

    print(f"Found {n_files} files.")
    print(f"Selecting {n_select} files (~{args.select_fraction*100:.1f}% of data).")

   
   
    docs = []
    for fp in file_paths:
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                docs.append(f.read())
        except Exception as e:
            print(f"Warning: could not read {fp}: {e}")
            docs.append("")

    # Build TF-IDF vectors
    print("Building TF-IDF matrix...")
    
    
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        lowercase=True,
        stop_words=None,
        token_pattern=r"(?u)\b\w+\b",
    )
    
    
    X = vectorizer.fit_transform(docs).astype(np.float32)
    print(f"TF-IDF shape:", X.shape)

    # Farthest-first selection in TF-IDF space using cosine distance
    print("Running farthest-first selection (cosine distance)...")
    selected_indices = farthest_first(
        X, k=n_select, metric="cosine", random_state=args.random_state
    )

    print(f"Selected {len(selected_indices)} documents.")
    os.makedirs(args.output_dir, exist_ok=True)

    # Save list of selected files
    selected_list_path = os.path.join(args.output_dir, "selected_files.txt")
    with open(selected_list_path, "w") as f:
        for idx in selected_indices:
            f.write(file_paths[idx] + "\n")

    # Copy selected files into the output dir
    print(f"Copying selected files into {args.output_dir} ...")
    for idx in selected_indices:
        src = file_paths[idx]
        dst = os.path.join(args.output_dir, os.path.basename(src))
        if not os.path.exists(dst):
            try:
                with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                    fdst.write(fsrc.read())
            except Exception as e:
                print(f"Warning: failed to copy {src} -> {dst}: {e}")

    print("Done. Selected file list saved to:", selected_list_path)


if __name__ == "__main__":
    main()

