"""Tests for condition-level train/val/test splitting."""

from __future__ import annotations

from graph_perturb.data.splits import make_splits


def test_no_control_in_held_out(processed, split_cfg):
    splits = make_splits(processed, split_cfg)
    for key in ("val", "test_single", "test_combo"):
        assert "ctrl" not in getattr(splits, key)
    assert "ctrl" not in splits.train


def test_train_non_empty(processed, split_cfg):
    splits = make_splits(processed, split_cfg)
    assert len(splits.train) >= 1


def test_singles_in_test_single_are_singles(processed, split_cfg):
    splits = make_splits(processed, split_cfg)
    for c in splits.test_single:
        assert not processed.is_combo(c)


def test_combos_in_test_combo_are_combos(processed, split_cfg):
    splits = make_splits(processed, split_cfg)
    for c in splits.test_combo:
        assert processed.is_combo(c)


def test_no_overlap_between_splits(processed, split_cfg):
    splits = make_splits(processed, split_cfg)
    parts = [
        set(splits.train),
        set(splits.val),
        set(splits.test_single),
        set(splits.test_combo),
    ]
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            assert parts[i].isdisjoint(parts[j])


def test_all_conditions_accounted_for(processed, split_cfg):
    splits = make_splits(processed, split_cfg)
    assigned = set(splits.all_conditions())
    assert assigned == set(processed.condition_labels())


def test_determinism_for_fixed_seed(processed, split_cfg):
    a = make_splits(processed, split_cfg)
    b = make_splits(processed, split_cfg)
    assert a.as_dict() == b.as_dict()
