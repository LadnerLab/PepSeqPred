import argparse
import pytest
from pepseqpred.apps.train_ffnn_cli import _build_run_plans

pytestmark = pytest.mark.unit


def _args(**overrides):
    base = argparse.Namespace(
        train_mode="ensemble-kfold",
        n_folds=2,
        split_seeds=None,
        train_seeds=None,
        fold_seed=None,
        seed=42,
        ensemble_train_seeds=None,
        split_type="id-family",
        val_frac=0.5,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_build_run_plans_ensemble_set_pairs_map_to_full_kfold_sets():
    args = _args(
        train_mode="ensemble-kfold",
        n_folds=2,
        split_seeds="101,202",
        train_seeds="11,22",
        split_type="id-family",
    )
    ids = ["a1", "a2", "b1", "b2"]
    family_groups = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}

    plans, meta = _build_run_plans(args, ids, family_groups)

    assert len(plans) == 4  # 2 sets x 2 folds
    assert [plan.run_index for plan in plans] == [1, 2, 3, 4]
    assert meta["n_sets"] == 2
    assert meta["n_folds"] == 2
    assert meta["split_seeds"] == [101, 202]
    assert meta["train_seeds"] == [11, 22]
    assert meta["ensemble_seed_mode"] == "set-paired"

    set_1 = [plan for plan in plans if plan.ensemble_set_index == 1]
    assert len(set_1) == 2
    assert {plan.fold_index for plan in set_1} == {1, 2}
    assert all(plan.split_seed == 101 for plan in set_1)
    assert all(plan.train_seed == 11 for plan in set_1)
    assert all(plan.ensemble_set_split_seed == 101 for plan in set_1)
    assert all(plan.ensemble_set_train_seed == 11 for plan in set_1)
    assert all(
        plan.save_dir_name.startswith("set_001_split_101_train_11/")
        for plan in set_1
    )

    set_2 = [plan for plan in plans if plan.ensemble_set_index == 2]
    assert len(set_2) == 2
    assert {plan.fold_index for plan in set_2} == {1, 2}
    assert all(plan.split_seed == 202 for plan in set_2)
    assert all(plan.train_seed == 22 for plan in set_2)
    assert all(plan.ensemble_set_split_seed == 202 for plan in set_2)
    assert all(plan.ensemble_set_train_seed == 22 for plan in set_2)
    assert all(
        plan.save_dir_name.startswith("set_002_split_202_train_22/")
        for plan in set_2
    )


def test_build_run_plans_ensemble_set_pairs_require_matching_lengths():
    args = _args(
        train_mode="ensemble-kfold",
        n_folds=2,
        split_seeds="101,202",
        train_seeds="11",
        split_type="id",
    )
    ids = ["p1", "p2", "p3", "p4"]

    with pytest.raises(ValueError, match="same length"):
        _build_run_plans(args, ids, {})


def test_build_run_plans_ensemble_legacy_per_fold_train_seeds():
    args = _args(
        train_mode="ensemble-kfold",
        n_folds=2,
        split_seeds=None,
        train_seeds=None,
        fold_seed=17,
        seed=5,
        ensemble_train_seeds="11,12",
        split_type="id",
    )
    ids = ["p1", "p2", "p3", "p4"]

    plans, meta = _build_run_plans(args, ids, {})

    assert len(plans) == 2
    assert meta["ensemble_seed_mode"] == "legacy-per-fold-train-seeds"
    assert meta["ensemble_train_seeds_per_fold"] == [11, 12]
    assert all(plan.split_seed == 17 for plan in plans)
    assert [plan.train_seed for plan in plans] == [11, 12]
    assert [plan.ensemble_set_index for plan in plans] == [1, 1]


def test_build_run_plans_seeded_mode_keeps_prior_behavior():
    args = _args(
        train_mode="seeded",
        split_seeds="11,22",
        train_seeds="101,202",
        split_type="id",
        val_frac=0.5,
    )
    ids = ["p1", "p2", "p3", "p4"]

    plans, meta = _build_run_plans(args, ids, {})

    assert len(plans) == 2
    assert [plan.run_index for plan in plans] == [1, 2]
    assert [plan.split_seed for plan in plans] == [11, 22]
    assert [plan.train_seed for plan in plans] == [101, 202]
    assert all(plan.ensemble_set_index is None for plan in plans)
    assert meta["split_seeds"] == [11, 22]
    assert meta["train_seeds"] == [101, 202]
