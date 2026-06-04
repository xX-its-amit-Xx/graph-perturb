"""Tests for the Typer CLI, exercised offline via CliRunner."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from graph_perturb.cli import app

runner = CliRunner()


def test_graphs_command_lists_backends():
    result = runner.invoke(app, ["graphs"])
    assert result.exit_code == 0, result.output
    assert "Available graph backends:" in result.output
    assert "custom" in result.output
    assert "go_bp" in result.output


def test_synthetic_train_gnn_smoke():
    # Use go_bp's offline co-expression fallback (no network, no file needed).
    result = runner.invoke(
        app, ["train", "--synthetic", "--graph", "go_bp", "--model", "gnn", "--epochs", "1"]
    )
    assert result.exit_code == 0, result.output
    assert "Final metrics" in result.output
    assert "pearson_delta" in result.output


def test_synthetic_train_vae_smoke():
    result = runner.invoke(
        app, ["train", "--synthetic", "--graph", "go_bp", "--model", "vae", "--epochs", "1"]
    )
    assert result.exit_code == 0, result.output
    assert "pearson_delta" in result.output


def test_build_graph_synthetic_smoke():
    result = runner.invoke(app, ["build-graph", "--backend", "go_bp", "--synthetic"])
    assert result.exit_code == 0, result.output
    assert "PerturbationGraph" in result.output
