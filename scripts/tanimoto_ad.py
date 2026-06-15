"""
Add a nearest-neighbour Tanimoto AD column to a chemprop predictions CSV.

For each query compound, computes the maximum Tanimoto similarity to any
compound in the training set using Morgan fingerprints (radius 2, 2048 bits).
Higher similarity = more likely within the training domain.
"""

import argparse
import csv
import sys

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import AllChem, MolFromSmiles


def morgan_fp(smiles, radius=2, n_bits=2048):
    mol = MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_path", required=True, help="Training set CSV with smiles column")
    parser.add_argument("--preds_path", required=True, help="Predictions CSV to add AD column to")
    parser.add_argument("--smiles_column", default="smiles")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n_bits", type=int, default=2048)
    args = parser.parse_args()

    print("Loading training set fingerprints...")
    train_fps = []
    with open(args.train_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fp = morgan_fp(row[args.smiles_column], args.radius, args.n_bits)
            if fp is not None:
                train_fps.append(fp)
    print(f"  {len(train_fps)} training compounds")

    print("Loading query compounds...")
    with open(args.preds_path) as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())
    print(f"  {len(rows)} query compounds")

    col = f"tanimoto_nn_r{args.radius}_{args.n_bits}b"
    print(f"Computing nearest-neighbour Tanimoto similarity (column: {col})...")

    invalid = 0
    for i, row in enumerate(rows):
        if i % 2000 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
        fp = morgan_fp(row[args.smiles_column], args.radius, args.n_bits)
        if fp is None:
            row[col] = ""
            invalid += 1
        else:
            sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
            row[col] = round(max(sims), 6)

    if invalid:
        print(f"  Warning: {invalid} SMILES could not be parsed", file=sys.stderr)

    print(f"Writing results to {args.preds_path}...")
    with open(args.preds_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + [col])
        writer.writeheader()
        writer.writerows(rows)

    print("Done.")


if __name__ == "__main__":
    main()
