"""Input rebuild rule and the handling of rows without a usable atom map."""

import pandas as pd
import pytest

from emulator_bench import dataset_adapter
from emulator_bench.dataset_adapter import Split, build_split_inputs

GOOD = "[CH3:1][C:2](=[O:3])[OH:4].[OH:5][CH2:6][CH3:7]>>[CH3:1][C:2](=[O:3])[O:5][CH2:6][CH3:7].[OH2:4]"
PARTIAL = (
    "[CH3:1][C:2](=[O:3])[OH:4].[OH:5][CH2:6][CH3:7]>>[CH3:1][C:2](=[O:3])[O:5][CH2:6][CH3:7].O"
)
ROWS = {
    "CC(=O)O.OCC>>CC(=O)OCC.O": GOOD,
    "CC(=O)O.OCCC>>CC(=O)OCCC.O": PARTIAL,
    "CC(=O)O.OCCCC>>CC(=O)OCCCC.O": None,
}


def make_split(folder) -> Split:
    frame = pd.DataFrame({"rxn_smiles": list(ROWS), "ec_number": "3.1.1.1", "sequence": "MKV"})
    for name in dataset_adapter.SPLITS:
        frame.to_parquet(folder / f"{name}.parquet")
    return Split("g", "all", folder)


def atom_maps(path) -> dict:
    mapped = {k: v for k, v in ROWS.items() if v}
    pd.DataFrame({"rxn_smiles": list(mapped)}).to_parquet(
        path
    )  # the file identity is what matters
    return mapped


def test_unusable_rows_are_dropped_from_train_val_and_missed_in_test(tmp_path):
    split = make_split(tmp_path)
    stats = build_split_inputs(
        split, tmp_path / "out", atom_maps(tmp_path / "maps.parquet"), tmp_path / "maps.parquet"
    )
    for name in ("train", "val"):
        s = stats["splits"][name]
        assert (s["no_atom_map_rows"], s["failed_featurization_rows"], s["usable_rows"]) == (
            1,
            1,
            1,
        )
    s = stats["splits"]["test"]
    assert (
        s["no_atom_map_rows"],
        s["failed_featurization_rows"],
        s["scored_as_miss_rows"],
        s["usable_rows"],
    ) == (1, 1, 2, 1)
    queries = pd.read_parquet(tmp_path / "out/enzyme_retrieval/g/all/inputs/test_queries.parquet")
    assert len(queries) == 3 and queries["mapped"].notna().sum() == 1


def test_inputs_are_rebuilt_when_maps_or_sources_change(tmp_path):
    split = make_split(tmp_path)
    maps_file = tmp_path / "maps.parquet"
    maps = atom_maps(maps_file)
    out = tmp_path / "out"
    first = build_split_inputs(split, out, maps, maps_file)
    assert build_split_inputs(split, out, {}, maps_file) == first  # same files: reused
    pd.DataFrame({"rxn_smiles": list(maps) + ["x"]}).to_parquet(maps_file)
    with pytest.raises(ValueError, match="no usable rows"):  # the new map file forces a rebuild
        build_split_inputs(split, out, {}, maps_file)
    build_split_inputs(split, out, maps, maps_file)
    pd.read_parquet(tmp_path / "train.parquet").iloc[:1].to_parquet(tmp_path / "train.parquet")
    assert build_split_inputs(split, out, maps, maps_file)["splits"]["train"]["raw_rows"] == 1
