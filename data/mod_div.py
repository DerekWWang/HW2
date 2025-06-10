#!/usr/bin/env python3
import csv
import random
from math import gcd
from typing import List, Tuple

def generate_mod_examples(p: int) -> List[str]:
    """
    Generate all expressions of the form:
      a+b = c mod p
      a-b = c mod p
      a/b = c mod p   (only when b is invertible mod p)
    for 0 <= a, b < p.
    """
    examples = []
    for a in range(p):
        for b in range(p):
            # division (only when gcd(b,p)==1)
            if b != 0 and gcd(b, p) == 1:
                inv_b = pow(b, -1, p)      # modular inverse
                c_div = (a * inv_b) % p
                examples.append(f"{a}/{b}={c_div} mod {p}")
    return examples

def split_dataset(
    data: List[str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> Tuple[List[str], List[str], List[str]]:
    random.shuffle(data)
    n = len(data)
    n_train = int(train_frac * n)
    n_val   = int(val_frac   * n)
    train   = data[:n_train]
    val     = data[n_train:n_train + n_val]
    test    = data[n_train + n_val:]
    return train, val, test

def write_csv(lines: List[str], filename: str) -> None:
    """Write each line as a single-column CSV."""
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        for line in lines:
            writer.writerow([line])

if __name__ == "__main__":
    random.seed(42)            # for reproducible shuffling
    moduli = [97]
    for p in moduli:
        data = generate_mod_examples(p)[:3000]
        train, val, test = split_dataset(data, train_frac=0.8, val_frac=0.1)

        # write out files
        write_csv(train, f"train_mod{p}.csv")
        write_csv(val,   f"val_mod{p}.csv")
        write_csv(test,  f"test_mod{p}.csv")

        # print summary
        print(f"mod {p:3d} → total={len(data):5d} | "
              f"train={len(train):5d} | val={len(val):5d} | test={len(test):5d}")
