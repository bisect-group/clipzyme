from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .utils import DEFAULT_DATASET_ROOT

DEFAULT_MAP_FILE = DEFAULT_DATASET_ROOT / "reaction_atom_mappings.parquet"
MAPPED_PROTON = re.compile(r"\[H\+:\d+\]")


def load_atom_maps(path: str | Path) -> dict[str, str]:
    """Read the EMULaToR LocalMapper maps: rxn_smiles -> atom-mapped reaction, for rows with status "mapped"."""
    frame = pd.read_parquet(
        path, columns=["rxn_smiles", "atom_mapped_rxn_smiles", "atom_mapping_status"]
    )
    frame = frame[
        (frame["atom_mapping_status"] == "mapped") & frame["atom_mapped_rxn_smiles"].notna()
    ]
    # The original code drops unmapped protons ("[H+]"), so a mapped proton becomes one too.
    mapped = frame["atom_mapped_rxn_smiles"].map(lambda m: MAPPED_PROTON.sub("[H+]", m))
    return dict(zip(frame["rxn_smiles"].str.strip(), mapped))


def clipzyme_reaction(mapped_reaction: str) -> tuple[list[str], list[str], str]:
    """Reactants and products as the EnzymeMapGraph dataset builds them (clipzyme/datasets/enzymemap.py)."""
    reactants, products = mapped_reaction.split(">>")
    reactants = sorted(s for s in reactants.split(".") if s and s != "[H+]")
    products = sorted(s for s in products.split(".") if s and s != "[H+]")
    products = [p for p in products if p not in reactants]
    return reactants, products, f"{'.'.join(reactants)}>>{'.'.join(products)}"


def _atom_maps(smiles: str) -> list[int]:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit cannot parse the mapped SMILES")
    return [atom.GetAtomMapNum() for atom in mol.GetAtoms()]


def validate_mapped_reaction(mapped_reaction: str) -> tuple[str | None, list, str | None]:
    """Run the original featurization. Return (reaction, bond_changes, error); reaction is None on failure."""
    from clipzyme.utils.pyg import from_mapped_smiles
    from clipzyme.utils.wln_processing import get_bond_changes

    try:
        reactants, products, reaction = clipzyme_reaction(mapped_reaction)
        if not reactants or not products:
            return None, [], "no reactants or products"
        reactant_maps = _atom_maps(".".join(reactants))
        product_maps = _atom_maps(".".join(products))
        if 0 in reactant_maps or 0 in product_maps:
            return None, [], "unmapped atoms"
        if len(set(reactant_maps)) != len(reactant_maps) or set(reactant_maps) != set(
            product_maps
        ):
            return None, [], "atom maps differ between reactants and products"
        bond_changes = sorted(get_bond_changes(reaction))
        if not bond_changes:
            return None, [], "no bond changes"
        reactant_graph, atom_map2index = from_mapped_smiles(
            ".".join(reactants), encode_no_edge=True
        )
        product_graph, _ = from_mapped_smiles(".".join(products), encode_no_edge=True)
        if reactant_graph.x.shape[0] != product_graph.x.shape[0]:
            return None, [], "graph node counts differ"
        for u, v, _ in bond_changes:
            atom_map2index[int(u)], atom_map2index[int(v)]
        return reaction, [[str(u), str(v), float(t)] for u, v, t in bond_changes], None
    except Exception as exc:
        return None, [], f"{type(exc).__name__}: {exc}"
