"""Builds a descriptor-prediction corpus for pretraining a chemprop 2 foundation model.

A CheMeleon-style foundation model needs no experimental data: the targets are
molecular descriptors computed from each structure, so a list of SMILES is the
only input. This script turns such a list into the CSV chemprop 2 trains on.

It runs in two stages, because computing descriptors for a million molecules takes
hours and should not have to start over:

    prepare   standardise structures, compute descriptors, write parquet chunks
    finalize  choose usable descriptor columns, winsorise and rescale them,
              write the training CSV

``prepare`` skips chunks it has already written, so it can be interrupted and
resumed. ``finalize`` reads the chunks twice: once to measure each column, once to
write the scaled values.

Run it with the chemprop 2 environment's interpreter, which has RDKit, Mordred and
descriptastorus::

    ~/Programs/miniconda3/envs/chemprop2/bin/python scripts/pretrain_corpus.py \\
        prepare -i chembl.csv --smiles-column canonical_smiles -o corpus/
    ~/Programs/miniconda3/envs/chemprop2/bin/python scripts/pretrain_corpus.py \\
        finalize -o corpus/

Start with a subsample (--limit 100000) to check the pipeline end to end before
committing to a full corpus.
"""

import argparse
import csv
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog('rdApp.*')

# Elements a small organic molecule is made of. Anything else (metals, boron
# clusters, exotic salts) is dropped: descriptors for them are poorly defined and
# they are not what a downstream model will be asked about.
ORGANIC = {'H', 'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Se', 'Br', 'I'}

CHUNK_PREFIX = 'chunk_'
KEYS_SUFFIX = '.keys'
STATS_FILENAME = 'descriptor_stats.json'


# --- structure standardisation -------------------------------------------

_LARGEST_FRAGMENT = None
_UNCHARGER = None


def standardize(smiles: str, min_mw: float, max_mw: float, max_atoms: int):
    """Returns ``(canonical_smiles, inchikey)`` or None if the molecule is unusable.

    Salts and mixtures are reduced to their largest fragment and the result is
    neutralised, so the corpus holds one canonical form of each compound rather
    than several salt forms of it.
    """
    global _LARGEST_FRAGMENT, _UNCHARGER
    if _LARGEST_FRAGMENT is None:
        _LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()
        _UNCHARGER = rdMolStandardize.Uncharger()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        mol = _LARGEST_FRAGMENT.choose(mol)
        mol = _UNCHARGER.uncharge(mol)
    except Exception:
        return None

    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return None
    if mol.GetNumHeavyAtoms() > max_atoms:
        return None
    if any(atom.GetSymbol() not in ORGANIC for atom in mol.GetAtoms()):
        return None

    mw = Descriptors.MolWt(mol)
    if not (min_mw <= mw <= max_mw):
        return None

    try:
        return Chem.MolToSmiles(mol), Chem.MolToInchiKey(mol)
    except Exception:
        return None


def _standardize_one(args):
    return standardize(*args)


# --- descriptors ----------------------------------------------------------

def mordred_descriptors(smiles_list, nproc):
    """~1600 Mordred 2D descriptors, the labels CheMeleon was trained on."""
    from mordred import Calculator, descriptors as mordred_module

    calc = Calculator(mordred_module, ignore_3D=True)
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    frame = calc.pandas(mols, nproc=nproc, quiet=True)

    # Mordred reports failures as error objects rather than NaN.
    return frame.apply(pd.to_numeric, errors='coerce').astype('float32')


_RDKIT2D = None


def _rdkit2d_one(smiles):
    global _RDKIT2D
    if _RDKIT2D is None:
        from descriptastorus.descriptors import rdNormalizedDescriptors
        _RDKIT2D = rdNormalizedDescriptors.RDKit2DNormalized()
    # process() returns a flat [ok, v1, ... vN]; the first element is a success
    # flag, not a descriptor.
    values = _RDKIT2D.process(smiles)
    return values[1:] if values and values[0] else None


def rdkit2d_descriptors(smiles_list, nproc):
    """200 CDF-normalised RDKit descriptors: much cheaper than Mordred, and a
    sensible choice for a first run that only has to prove the pipeline works."""
    from descriptastorus.descriptors import rdNormalizedDescriptors

    columns = [name for name, _ in rdNormalizedDescriptors.RDKit2DNormalized().columns]
    with Pool(nproc) as pool:
        rows = pool.map(_rdkit2d_one, smiles_list, chunksize=200)

    width = len(columns)
    rows = [row if row is not None else [np.nan] * width for row in rows]
    return pd.DataFrame(rows, columns=columns, dtype='float32')


DESCRIPTOR_SETS = {'mordred': mordred_descriptors, 'rdkit2d': rdkit2d_descriptors}


# --- prepare --------------------------------------------------------------

def read_smiles(path: str, column: str, limit: int):
    """Streams SMILES from a CSV (by column name) or a plain one-per-line file."""
    with open(path) as f:
        if path.endswith(('.csv', '.tsv')):
            reader = csv.DictReader(f, delimiter='\t' if path.endswith('.tsv') else ',')
            if column not in (reader.fieldnames or []):
                raise SystemExit(f'No column "{column}" in {path}; found: {reader.fieldnames}')
            source = (row[column] for row in reader)
        else:
            source = (line.split()[0] for line in f if line.strip())

        for i, smiles in enumerate(source):
            if limit and i >= limit:
                return
            if smiles:
                yield smiles


def prepare(args):
    os.makedirs(args.out_dir, exist_ok=True)
    describe = DESCRIPTOR_SETS[args.descriptors]

    # Structures already accepted, so a resumed run does not re-add them.
    seen = set()
    for keys_file in sorted(glob.glob(os.path.join(args.out_dir, f'*{KEYS_SUFFIX}'))):
        with open(keys_file) as f:
            seen.update(line.strip() for line in f if line.strip())
    if seen:
        print(f'resuming: {len(seen):,} structures already in the corpus')

    chunk_index = 0
    batch_smiles, batch_keys = [], []
    kept = rejected = duplicates = 0

    def flush():
        """Writes one chunk of descriptors, unless it is already on disk."""
        nonlocal chunk_index, batch_smiles, batch_keys
        if not batch_smiles:
            return
        path = os.path.join(args.out_dir, f'{CHUNK_PREFIX}{chunk_index:06d}.parquet')
        if not os.path.exists(path):
            frame = describe(batch_smiles, args.nproc)
            frame.insert(0, 'smiles', batch_smiles)
            frame.to_parquet(path, index=False)
            with open(path + KEYS_SUFFIX, 'w') as f:
                f.write('\n'.join(batch_keys) + '\n')
            print(f'  wrote {os.path.basename(path)}: {len(frame):,} molecules, '
                  f'{frame.shape[1] - 1} descriptors')
        chunk_index += 1
        batch_smiles, batch_keys = [], []

    standardise_args = (args.min_mw, args.max_mw, args.max_atoms)

    with Pool(args.nproc) as pool:
        stream = read_smiles(args.input, args.smiles_column, args.limit)
        work = ((s, *standardise_args) for s in stream)

        for result in pool.imap(_standardize_one, work, chunksize=500):
            if result is None:
                rejected += 1
                continue
            smiles, key = result
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            kept += 1
            batch_smiles.append(smiles)
            batch_keys.append(key)
            if len(batch_smiles) >= args.chunk_size:
                flush()
        flush()

    print(f'\nkept {kept:,} structures | {rejected:,} unusable | {duplicates:,} duplicates')
    print(f'chunks in {args.out_dir}')
    print(f'next: {sys.argv[0]} finalize -o {args.out_dir}')


# --- finalize -------------------------------------------------------------

def sample_values(chunks, sample_rows):
    """Reads a bounded sample of rows, to measure columns without loading the lot."""
    per_chunk = max(1, sample_rows // max(1, len(chunks)))
    frames = []
    for path in chunks:
        frame = pd.read_parquet(path)
        frames.append(frame.head(per_chunk).drop(columns=['smiles']))
    return pd.concat(frames, ignore_index=True)


def finalize(args):
    chunks = sorted(glob.glob(os.path.join(args.out_dir, f'{CHUNK_PREFIX}*.parquet')))
    if not chunks:
        raise SystemExit(f'No chunks in {args.out_dir}; run prepare first.')

    print(f'measuring {len(chunks)} chunks...')
    sample = sample_values(chunks, args.sample_rows)
    print(f'  sampled {len(sample):,} rows x {sample.shape[1]} descriptors')

    # A descriptor is usable when it is defined for most molecules and actually
    # varies. Mordred emits many columns that are constant or fail on most inputs.
    filled = sample.notna().mean()
    spread = sample.std(numeric_only=True)
    keep = [c for c in sample.columns
            if filled.get(c, 0) >= args.min_fill and spread.get(c, 0) > 0]
    dropped = sample.shape[1] - len(keep)
    print(f'  keeping {len(keep)} descriptors, dropping {dropped} '
          f'(sparse or constant)')
    if not keep:
        raise SystemExit('No usable descriptor columns.')

    # Winsorise before scaling: descriptor tails run to enormous values, and a
    # single outlier otherwise decides a column's scale.
    low = sample[keep].quantile(args.winsorize)
    high = sample[keep].quantile(1 - args.winsorize)
    clipped = sample[keep].clip(low, high, axis=1)
    mean, std = clipped.mean(), clipped.std().replace(0, 1)

    stats = {'descriptors': keep,
             'winsorize_quantile': args.winsorize,
             'low': low.to_dict(), 'high': high.to_dict(),
             'mean': mean.to_dict(), 'std': std.to_dict()}
    with open(os.path.join(args.out_dir, STATS_FILENAME), 'w') as f:
        json.dump(stats, f, indent=1)

    output = args.output or os.path.join(args.out_dir, 'pretrain.csv')
    rows = 0
    for i, path in enumerate(chunks):
        frame = pd.read_parquet(path)
        smiles = frame['smiles']
        values = frame[keep].clip(low, high, axis=1)
        values = ((values - mean) / std).astype('float32')
        # Anything still missing sits at the mean, i.e. contributes no signal.
        values = values.fillna(0.0).round(4)
        # Concatenated rather than inserted: with a thousand descriptor columns,
        # inserting into the frame leaves it fragmented and slow.
        values = pd.concat([smiles, values], axis=1)
        values.to_csv(output, mode='w' if i == 0 else 'a', header=(i == 0), index=False)
        rows += len(values)

    size_gb = os.path.getsize(output) / 1e9
    print(f'\nwrote {output}: {rows:,} molecules x {len(keep)} targets ({size_gb:.2f} GB)')
    print(f'column statistics: {os.path.join(args.out_dir, STATS_FILENAME)}')
    print_train_command(output, keep)


def print_train_command(data_path: str, targets):
    """Prints a training command matching CheMeleon's own architecture."""
    print('\nPretrain with (CheMeleon uses d_h=2048, depth=6, dropout=0):\n')
    print(f'  chemprop train \\\n'
          f'      --data-path {data_path} \\\n'
          f'      --smiles-columns smiles \\\n'
          f'      --target-columns $(head -1 {data_path} | cut -d, -f2- | tr "," " ") \\\n'
          f'      --task-type regression \\\n'
          f'      --message-hidden-dim 2048 --depth 6 --dropout 0 \\\n'
          f'      --epochs 50 --batch-size 64 --num-workers 0 \\\n'
          f'      --accelerator gpu \\\n'
          f'      --output-dir pretrained_foundation\n')
    print(f'{len(targets)} target columns. The model file it writes '
          f'(pretrained_foundation/model_0/best.pt) is what --from-foundation takes;\n'
          f'upload it on the Checkpoints page to use it from the web app.')


# --- cli ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='stage', required=True)

    p = sub.add_parser('prepare', help='standardise structures and compute descriptors')
    p.add_argument('-i', '--input', required=True, help='CSV or one-SMILES-per-line file')
    p.add_argument('-o', '--out-dir', required=True)
    p.add_argument('--smiles-column', default='smiles')
    p.add_argument('--descriptors', choices=sorted(DESCRIPTOR_SETS), default='mordred')
    p.add_argument('--chunk-size', type=int, default=5000)
    p.add_argument('--nproc', type=int, default=min(32, os.cpu_count() or 1),
                   help='worker processes (default 32, leaving cores for other jobs)')
    p.add_argument('--limit', type=int, default=0, help='read only the first N molecules')
    p.add_argument('--min-mw', type=float, default=100.0)
    p.add_argument('--max-mw', type=float, default=1000.0)
    p.add_argument('--max-atoms', type=int, default=100)
    p.set_defaults(func=prepare)

    p = sub.add_parser('finalize', help='select, winsorise and rescale the descriptors')
    p.add_argument('-o', '--out-dir', required=True)
    p.add_argument('--output', help='training CSV to write (default <out-dir>/pretrain.csv)')
    p.add_argument('--min-fill', type=float, default=0.95,
                   help='drop descriptors defined for fewer than this fraction')
    p.add_argument('--winsorize', type=float, default=0.001,
                   help='quantile clipped from each tail before scaling')
    p.add_argument('--sample-rows', type=int, default=200000,
                   help='rows sampled to measure the columns')
    p.set_defaults(func=finalize)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
