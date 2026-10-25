import pandas as pd

from emulator_bench.atom_maps import clipzyme_reaction, load_atom_maps
from emulator_bench.dataset_adapter import candidate_pool, discover_splits, read_rows
from emulator_bench.results import ec_level_accuracy, matched_levels
from emulator_bench.train import build_command


def write_split(folder, columns=("rxn_smiles", "ec_number", "sequence")):
    folder.mkdir(parents=True)
    for name in ("train", "val", "test"):
        pd.DataFrame({c: ["A>>B"] for c in columns}).to_parquet(folder / f"{name}.parquet")


def test_discover_splits_names_and_skips(tmp_path):
    write_split(tmp_path / "with_protein_information" / "random_splits_grouped_rxn_key")
    write_split(tmp_path / "with_protein_information" / "enzyme_sequence_splits" / "threshold_0.4")
    write_split(tmp_path / "uniprot_time_splits", columns=("rxn_smiles", "ec_number"))
    splits, skipped = discover_splits(tmp_path)
    assert sorted(s.id for s in splits) == [
        "enzyme_sequence_splits/threshold_0.4",
        "random_splits_grouped_rxn_key/all",
    ]
    assert skipped == ["uniprot_time_splits"]


def test_read_rows_counts_each_drop(tmp_path):
    pd.DataFrame(
        {
            "rxn_smiles": ["CC>>CCO", "CC>>CCO", "bad", "CC>>CO", "CC>>CCC"],
            "ec_number": ["1.1.1.1", "1.1.1.1", "1.1.1.1", "1.1.-.-", "2.7.1.1"],
            "sequence": ["MK", "MK", "MK", "MK", None],
        }
    ).to_parquet(tmp_path / "x.parquet")
    frame, stats = read_rows(tmp_path / "x.parquet")
    assert len(frame) == 1
    assert (
        stats["missing_value_rows"],
        stats["invalid_reaction_rows"],
        stats["incomplete_ec_rows"],
    ) == (1, 1, 1)
    assert stats["exact_duplicate_rows"] == 1


def test_ec_level_accuracy_uses_prefix_match():
    predictions = pd.DataFrame(
        {
            "true_ecs": [["1.2.3.4"], ["1.2.3.4", "3.1.1.1"]],
            "ranked_ecs": [["1.2.9.9", "1.2.3.4"]] * 2,
        }
    )
    assert matched_levels("1.2.3.4", ["1.2.9.9"]) == 2
    top1 = ec_level_accuracy(predictions)["k=1"]
    assert top1 == {"level_4": 0.0, "level_3": 0.0, "level_2": 50.0 + 25.0, "level_1": 75.0}
    assert ec_level_accuracy(predictions)["k=3"]["level_4"] == 75.0


def test_reaction_without_ranking_is_a_miss_at_every_k():
    predictions = pd.DataFrame(
        {"true_ecs": [["1.2.3.4"], ["1.2.3.4"]], "ranked_ecs": [["1.2.3.4"], []]}
    )
    accuracy = ec_level_accuracy(predictions)
    assert all(
        levels["level_4"] == 50.0 and levels["level_1"] == 50.0 for levels in accuracy.values()
    )


def test_clipzyme_reaction_drops_protons_and_products_in_reactants():
    reactants, products, reaction = clipzyme_reaction(
        "[CH3:1][OH:2].[H+]>>[CH3:1][OH:2].[CH4:3].[H+]"
    )
    assert reactants == ["[CH3:1][OH:2]"] and products == ["[CH4:3]"]
    assert reaction == "[CH3:1][OH:2]>>[CH4:3]"


def test_load_atom_maps_keeps_mapped_rows_only_and_plain_protons(tmp_path):
    pd.DataFrame(
        {
            "rxn_smiles": ["CC>>CO", "A>>B", "C>>D"],
            "atom_mapped_rxn_smiles": ["[CH3:1]>>[CH3:1].[H+:2]", None, "[CH4:1]>>[CH4:1]"],
            "atom_mapping_status": ["mapped", "failed", "failed"],
        }
    ).to_parquet(tmp_path / "maps.parquet")
    assert load_atom_maps(tmp_path / "maps.parquet") == {"CC>>CO": "[CH3:1]>>[CH3:1].[H+]"}


def test_train_command_uses_original_settings_and_run_folder(tmp_path):
    from argparse import Namespace

    args = Namespace(
        amp=False,
        frozen_esm=False,
        epochs=None,
        out_root=tmp_path,
        hub_dir=None,
        num_workers=2,
        seed=666,
    )
    command = build_command(args, "enzyme_sequence_splits", "threshold_0.4")
    text = " ".join(command)
    assert (
        "--precision bf16" in text
        and "--max_epochs 20" in text
        and "--accumulate_grad_batches 8" in text
    )
    assert (
        f"--checkpoint_dir {tmp_path}/enzyme_retrieval/enzyme_sequence_splits/threshold_0.4 "
        in text
    )
    assert "--experiment_name seed_666" in text
    assert "--freeze_esm" not in command and "--train" in command


def test_discover_splits_skips_backup_folders(tmp_path):
    base = tmp_path / "with_protein_information" / "enzyme_structure_splits"
    write_split(base / "threshold_0.86")
    write_split(base / "backup_2026-10-21" / "threshold_0.7")
    assert [s.id for s in discover_splits(tmp_path)[0]] == [
        "enzyme_structure_splits/threshold_0.86"
    ]


def test_candidate_pool_has_train_val_and_test_proteins():
    frames = {
        name: pd.DataFrame({"sequence": [seq], "ec_number": [ec]})
        for name, seq, ec in (
            ("train", "MK", "1.1.1.1"),
            ("val", "MKL", "2.1.1.1"),
            ("test", "MK", "1.1.1.2"),
        )
    }
    pool = candidate_pool(frames, max_length=3)
    assert pool.values.tolist() == [["MK", "1.1.1.1"], ["MK", "1.1.1.2"], ["MKL", "2.1.1.1"]]
    assert len(candidate_pool(frames, max_length=2)) == 2
