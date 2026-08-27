"""Atom-level attribution via gradient × activation (GradCAM-style)."""

import base64
import io
import logging
import threading
from collections import OrderedDict
from typing import List, Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from chemprop.utils import load_checkpoint, load_args


# ---------------------------------------------------------------------------
# Per-process caches.
#
# The Checkpoints/Train scatter plots fire one /get_attribution request per
# hovered point. Without caching, every request re-read each ensemble member
# from disk and deserialized it (seconds per hover). We keep the loaded models
# and the rendered SVGs in memory so the first hover for a checkpoint pays the
# load cost and the rest are near-instant.
# ---------------------------------------------------------------------------

_MODEL_CACHE = OrderedDict()   # path -> loaded MoleculeModel
_MODEL_CACHE_MAX = 12
_ARGS_CACHE = {}               # path -> TrainArgs
_SVG_CACHE = OrderedDict()     # (path0, smiles) -> svg string
_SVG_CACHE_MAX = 500
_CACHE_LOCK = threading.Lock()


def _cached_load_args(path: str):
    args = _ARGS_CACHE.get(path)
    if args is None:
        args = load_args(path)
        _ARGS_CACHE[path] = args
    return args


def _cached_load_checkpoint(path: str, device: torch.device):
    with _CACHE_LOCK:
        model = _MODEL_CACHE.get(path)
        if model is not None:
            _MODEL_CACHE.move_to_end(path)
            return model
    model = load_checkpoint(path, device=device)
    with _CACHE_LOCK:
        _MODEL_CACHE[path] = model
        _MODEL_CACHE.move_to_end(path)
        while len(_MODEL_CACHE) > _MODEL_CACHE_MAX:
            _MODEL_CACHE.popitem(last=False)
    return model


def _plain_svg(smiles_str: str, width: int = 400, height: int = 300) -> Optional[str]:
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
    """Render molecule with per-atom attribution weights as an inline SVG string.

    Atoms are shaded from green (increases the prediction) to red (decreases it).
    We highlight atoms rather than drawing a SimilarityMaps contour grid: the
    contour SVG runs to hundreds of KB and janks the browser when it is swapped
    into the hover tooltip on every mouse move, whereas atom highlighting is a
    few KB and renders instantly.
    """
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return None

    max_abs = float(np.abs(weights).max())
    norm = weights / max_abs if max_abs > 0 else weights

    highlight_atoms = list(range(mol.GetNumAtoms()))
    atom_colors = {}
    for i, w in enumerate(norm):
        w = max(-1.0, min(1.0, float(w)))
        if w >= 0:  # white -> green
            atom_colors[i] = (1.0 - 0.65 * w, 1.0, 1.0 - 0.65 * w)
        else:       # white -> red
            atom_colors[i] = (1.0, 1.0 + 0.65 * w, 1.0 + 0.65 * w)

    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer, mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=atom_colors,
            highlightBonds=[],
        )
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

    cache_key0 = model_paths[0] if model_paths else None

    svgs = []
    for smiles_str in smiles_list:
        if Chem.MolFromSmiles(smiles_str) is None:
            svgs.append(None)
            continue

        svg_key = (cache_key0, smiles_str)
        with _CACHE_LOCK:
            cached = _SVG_CACHE.get(svg_key)
            if cached is not None:
                _SVG_CACHE.move_to_end(svg_key)
        if cached is not None:
            svgs.append(cached)
            continue

        # Models trained with molecule-level features generators (rdkit_2d, morgan, etc.)
        # can't do atom-level attribution — fall back to a plain structure SVG.
        if model_paths:
            try:
                train_args = _cached_load_args(model_paths[0])
                if train_args.features_generator is not None:
                    svg = _plain_svg(smiles_str)
                    _store_svg(svg_key, svg)
                    svgs.append(svg)
                    continue
            except Exception:
                pass

        all_weights = []
        for path in model_paths:
            try:
                model = _cached_load_checkpoint(path, device)
                w = _compute_atom_weights(model, smiles_str)
                if w is not None:
                    all_weights.append(w)
            except Exception as e:
                logging.getLogger(__name__).warning(f"Attribution failed for {path}: {e}")

        if not all_weights:
            svg = _plain_svg(smiles_str)
        else:
            avg_weights = np.mean(all_weights, axis=0)
            svg = render_attribution_svg(smiles_str, avg_weights)
        _store_svg(svg_key, svg)
        svgs.append(svg)

    return svgs


def _store_svg(key, svg: Optional[str]) -> None:
    if svg is None:
        return
    with _CACHE_LOCK:
        _SVG_CACHE[key] = svg
        _SVG_CACHE.move_to_end(key)
        while len(_SVG_CACHE) > _SVG_CACHE_MAX:
            _SVG_CACHE.popitem(last=False)
