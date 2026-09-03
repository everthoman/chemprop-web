"""Applicability-domain estimation for predictions.

The applicability domain (AD) tells you how much a query molecule resembles the
molecules the model was actually trained on. A prediction for a molecule that
sits far outside the chemical space of the training set is an extrapolation and
should be treated with caution, regardless of the model's own uncertainty
estimate.

The measure used here is the classic *training-set similarity* AD:

  * Every molecule is encoded as a 2048-bit Morgan (ECFP4-style) fingerprint.
  * For a query molecule we find its single nearest neighbour in the training
    set and record the Tanimoto similarity to that neighbour.
  * A query is considered *within* the domain when this nearest-neighbour
    similarity is at least a threshold derived from the training set itself.

The threshold is data-driven rather than a magic constant: we look at how
similar each training molecule is to its own nearest neighbour and take a low
percentile of that distribution. A tightly clustered training set yields a high
threshold (so the model only trusts very close analogues), while a chemically
diverse training set yields a low threshold (the model legitimately covers a
wider space).
"""

import csv
import os
from typing import List, Optional, Sequence

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

# Fingerprint settings (radius 2 / 2048 bits == ECFP4, the de-facto standard for
# small-molecule similarity).
_MORGAN_RADIUS = 2
_MORGAN_BITS = 2048

# Deriving the threshold compares every reference with every other, so it is
# capped; 2000 molecules give a stable estimate of the distribution. Scoring a
# query is linear and uses the whole training set.
_MAX_REFERENCES = 2000

# Percentile of the training set's own nearest-neighbour similarity distribution
# used as the in-/out-of-domain cutoff. 5 means "95% of training molecules have
# a neighbour at least this similar".
_THRESHOLD_PERCENTILE = 5

# Fallback cutoff used when the training set is too small to derive a threshold.
_FALLBACK_THRESHOLD = 0.30

_RNG_SEED = 0


def _fingerprint(smiles: str):
    """Return a Morgan bit-vector fingerprint for ``smiles`` or ``None``."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, _MORGAN_RADIUS, nBits=_MORGAN_BITS)


def _derive_threshold(reference_fps: Sequence) -> float:
    """Derive an in-domain cutoff from the training set's own NN similarities."""
    n = len(reference_fps)
    if n < 2:
        return _FALLBACK_THRESHOLD
    nn_sims = []
    for i, fp in enumerate(reference_fps):
        others = reference_fps[:i] + reference_fps[i + 1:]
        sims = DataStructs.BulkTanimotoSimilarity(fp, others)
        nn_sims.append(max(sims))
    return float(np.percentile(nn_sims, _THRESHOLD_PERCENTILE))


class ApplicabilityDomain:
    """Scores query molecules against a fixed set of training references."""

    def __init__(self, reference_fps: Sequence, threshold: float):
        self._reference_fps = list(reference_fps)
        self.threshold = threshold
        self.num_references = len(self._reference_fps)

    @classmethod
    def from_training_smiles(cls, train_smiles: Sequence[str]) -> Optional["ApplicabilityDomain"]:
        """Build an AD scorer from training-set SMILES.

        A query is compared against every training molecule, so the similarity
        reported is exact. Only the threshold, which needs each reference's
        distance to its own nearest neighbour and so costs a pairwise pass, is
        derived from a bounded sample.

        That sample is drawn from the sorted molecules rather than the order they
        arrive in: two models trained on the same set write their training files
        in different orders, and sampling by position gave them different
        references, so the same molecule could be inside one model's domain and
        outside another's.

        Returns ``None`` when no valid reference molecules could be parsed.
        """
        fps = [fp for fp in (_fingerprint(s) for s in sorted(set(train_smiles)))
               if fp is not None]
        if not fps:
            return None

        sample = fps
        if len(fps) > _MAX_REFERENCES:
            rng = np.random.default_rng(_RNG_SEED)
            idx = sorted(rng.choice(len(fps), size=_MAX_REFERENCES, replace=False))
            sample = [fps[i] for i in idx]

        return cls(fps, _derive_threshold(sample))

    def score(self, smiles: str) -> Optional[dict]:
        """Score one query molecule.

        Returns a dict with ``similarity`` (nearest-neighbour Tanimoto, rounded),
        ``in_domain`` (bool) and ``threshold``; or ``None`` for invalid SMILES.
        """
        fp = _fingerprint(smiles)
        if fp is None:
            return None
        sims = DataStructs.BulkTanimotoSimilarity(fp, self._reference_fps)
        nn_sim = max(sims) if sims else 0.0
        return {
            'similarity': round(float(nn_sim), 3),
            'in_domain': bool(nn_sim >= self.threshold),
            'threshold': round(float(self.threshold), 3),
        }

    def score_all(self, smiles_list: Sequence[str]) -> List[Optional[dict]]:
        return [self.score(s) for s in smiles_list]


def load_training_smiles(preds_csv_path: str) -> List[str]:
    """Read the training-set SMILES from a ``{ckpt}_train_test_preds.csv`` file.

    Only rows whose ``split`` column equals ``train`` are returned (the model
    only ever saw the training molecules). Returns an empty list if the file is
    missing or lacks the expected columns.
    """
    if not preds_csv_path or not os.path.exists(preds_csv_path):
        return []
    smiles: List[str] = []
    with open(preds_csv_path, newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or 'smiles' not in reader.fieldnames:
            return []
        has_split = 'split' in reader.fieldnames
        for row in reader:
            if has_split and (row.get('split') or '').strip().lower() != 'train':
                continue
            smi = (row.get('smiles') or '').strip()
            if smi:
                smiles.append(smi)
    return smiles
