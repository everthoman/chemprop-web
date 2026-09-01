"""Atom-level attribution via gradient × activation (GradCAM-style)."""

import base64
import io
import logging
from typing import List, Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D, SimilarityMaps

from chemprop.utils import load_checkpoint, load_args


def plain_svg(smiles_str: str, width: int = 400, height: int = 300) -> Optional[str]:
    """Render a plain molecule SVG with no atom highlighting."""
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return None
    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        if svg.startswith('<?xml'):
            svg = svg[svg.index('<svg'):]
        return svg
    except Exception:
        return None


def _compute_atom_weights(model, smiles_str: str) -> Optional[np.ndarray]:
    """
    Compute per-atom contribution weights for a single SMILES using
    gradient × activation on the atom hidden states (GradCAM-style).
    Returns an array of shape [n_atoms], or None if unsupported.
    """
    mpn = model.encoder
    if mpn.features_only or mpn.reaction_solvent:
        return None

    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return None

    model.eval()

    with torch.enable_grad():
        output = model([[smiles_str]])  # [1, num_tasks]

        enc = mpn.encoder[0]
        if not hasattr(enc, '_atom_hiddens'):
            return None

        atom_hiddens = enc._atom_hiddens  # [total_atoms, hidden_size]
        a_scope = enc._a_scope

        # Sum all task outputs so gradients reflect overall influence
        pred = output[0].sum()
        grads = torch.autograd.grad(pred, atom_hiddens)[0]

        # GradCAM: grad * activation summed over hidden dim
        weights = (grads * atom_hiddens).sum(dim=-1)  # [total_atoms]

        a_start, a_size = a_scope[0]
        mol_weights = weights[a_start:a_start + a_size].detach().cpu().numpy()

    if len(mol_weights) != mol.GetNumAtoms():
        return None
    return mol_weights


def render_attribution_svg(smiles_str: str, weights: np.ndarray,
                           width: int = 400, height: int = 300) -> Optional[str]:
    """Render molecule with per-atom attribution weights as an inline SVG string."""
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return None

    max_abs = np.abs(weights).max()
    norm_weights = (weights / max_abs).tolist() if max_abs > 0 else weights.tolist()

    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        SimilarityMaps.GetSimilarityMapFromWeights(mol, norm_weights, drawer)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        # Strip XML declaration so the SVG embeds cleanly inline in HTML
        if svg.startswith('<?xml'):
            svg = svg[svg.index('<svg'):]
        return svg
    except Exception as e:
        logging.getLogger(__name__).warning(f"Attribution rendering failed for {smiles_str}: {e}")
        return None


def compute_attributions(model_paths: List[str], smiles_list: List[str],
                         device: torch.device = None) -> List[Optional[str]]:
    """
    Compute atom attribution SVGs for each SMILES, averaging weights across ensemble.
    Returns a list of SVG strings (or None for invalid/unsupported molecules).
    """
    if device is None:
        device = torch.device('cpu')

    svgs = []
    for smiles_str in smiles_list:
        if Chem.MolFromSmiles(smiles_str) is None:
            svgs.append(None)
            continue

        # Models trained with molecule-level features generators (rdkit_2d, morgan, etc.)
        # can't do atom-level attribution — fall back to a plain structure SVG.
        if model_paths:
            try:
                train_args = load_args(model_paths[0])
                if train_args.features_generator is not None:
                    svgs.append(plain_svg(smiles_str))
                    continue
            except Exception:
                pass

        all_weights = []
        for path in model_paths:
            try:
                model = load_checkpoint(path, device=device)
                w = _compute_atom_weights(model, smiles_str)
                if w is not None:
                    all_weights.append(w)
            except Exception as e:
                logging.getLogger(__name__).warning(f"Attribution failed for {path}: {e}")

        if not all_weights:
            svgs.append(plain_svg(smiles_str))
            continue

        avg_weights = np.mean(all_weights, axis=0)
        svgs.append(render_attribution_svg(smiles_str, avg_weights))

    return svgs
