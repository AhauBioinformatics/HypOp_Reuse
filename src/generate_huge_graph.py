import random
import os
import math
import time

def generate_and_save_gnp(n, p, out_dir, progress_step=0.01):
    """
    Generate a G(n,p) random graph using the skip method on n nodes (1-based),
    with all edge weights = 1, and save to out_dir/graph_n{n}_p{p:.3f}.txt.
    The process prints two progress bars: first for counting edges,
    second for writing the file.

    Parameters:
      n            Number of nodes
      p            Edge probability
      out_dir      Output directory
      progress_step Progress update interval (0~1), e.g., 0.01 means progress every 1%
    """
    os.makedirs(out_dir, exist_ok=True)
    fname       = f"graph_n{n}_p{p:.3f}.txt"
    final_path  = os.path.join(out_dir, fname)
    temp_path   = final_path + ".tmp"
    log_q       = math.log1p(-p)
    step_nodes  = max(1, int(n * progress_step))

    # —— First pass: count edges —— #
    print("Starting edge count phase …")
    t0 = time.time()
    m = 0
    for u in range(1, n+1):
        v = u
        while True:
            # Skip length follows geometric distribution
            r    = random.random()
            skip = int(math.log(r) / log_q)
            v   += skip + 1
            if v > n:
                break
            m += 1
        # Progress update
        if u % step_nodes == 0 or u == n:
            elapsed = time.time() - t0
            print(f"  [Count] Processed node {u}/{n} ({u/n:.1%}), counted edges m={m:,}, elapsed {elapsed:.1f}s")

    # —— Second pass: write edges to temp file —— #
    print("\nStarting edge writing phase …")
    t1 = time.time()
    with open(temp_path, "w") as f:
        count_edges = 0
        for u in range(1, n+1):
            v = u
            while True:
                r    = random.random()
                skip = int(math.log(r) / log_q)
                v   += skip + 1
                if v > n:
                    break
                f.write(f"{u} {v} 1\n")
                count_edges += 1
            # Progress update
            if u % step_nodes == 0 or u == n:
                elapsed = time.time() - t1
                print(f"  [Write] Processed node {u}/{n} ({u/n:.1%}), wrote edges {count_edges:,}, elapsed {elapsed:.1f}s")

    # —— Merge and write final file —— #
    print("\nMerging and writing final file …")
    with open(final_path, "w") as fout, open(temp_path, "r") as fin:
        fout.write(f"{n} {m}\n")
        for line in fin:
            fout.write(line)
    os.remove(temp_path)

    total_time = time.time() - t0
    print(f"\nAll done: saved file {final_path}")
    print(f"  Nodes n = {n}, Edges m = {m:,}, Density≈{m/(n*(n-1)/2):.6f}")
    print(f"  Total elapsed time {total_time:.1f}s\n")


if __name__ == "__main__":
    n       = 300000
    p       = 0.0001
    out_dir = "data/maxcut_data/huge/"
    # Print progress every 1% nodes; you can adjust to 0.005 (0.5%) or other values
    generate_and_save_gnp(n, p, out_dir, progress_step=0.01)
