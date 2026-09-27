"""
Tests for Phase 2 (Pretrain) and Phase 3 (Finetune) progression visualization.
Ensures that plotting routines correctly generate high-resolution curve artifacts
for self-supervised pretraining and multi-task fine-tuning.
"""

import os
from pathlib import Path
import pytest
from PIL import Image

from pharma_gnn.visualization import (
    plot_pretrain_progression,
    plot_finetune_curves,
    BENCHMARK_PRETRAIN_HISTORY,
    BENCHMARK_FINETUNE_HISTORY
)


def test_plot_pretrain_progression_default(tmp_path):
    """Verify default benchmark pretraining curves render and save successfully."""
    out_file = tmp_path / "test_pretrain.png"
    result_path = plot_pretrain_progression(output_path=out_file, dpi=100)
    
    assert Path(result_path).exists()
    assert Path(result_path).stat().st_size > 5000
    
    # Check that image is valid and readable
    with Image.open(result_path) as img:
        assert img.format == "PNG"
        assert img.width > 500
        assert img.height > 200


def test_plot_pretrain_progression_custom(tmp_path):
    """Verify custom pretraining epochs and sub-losses render properly."""
    out_file = tmp_path / "custom_pretrain.png"
    epochs = [1, 2, 3]
    train_losses = [2.5, 1.8, 1.2]
    val_losses = [2.6, 1.9, 1.3]
    sub_losses = {
        "atom": [1.0, 0.7, 0.4],
        "bond": [0.6, 0.4, 0.3],
        "motif": [0.5, 0.4, 0.3],
        "context": [0.4, 0.3, 0.2]
    }
    
    result_path = plot_pretrain_progression(
        epochs=epochs,
        train_losses=train_losses,
        val_losses=val_losses,
        sub_losses=sub_losses,
        output_path=out_file,
        dpi=100
    )
    
    assert Path(result_path).exists()
    assert Path(result_path).stat().st_size > 5000


def test_plot_finetune_curves_default(tmp_path):
    """Verify fine-tuning loss and ROC-AUC curves render and save successfully."""
    loss_file = tmp_path / "test_ft_loss.png"
    auc_file = tmp_path / "test_ft_auc.png"
    
    results = plot_finetune_curves(
        loss_path=loss_file,
        auc_path=auc_file,
        dpi=100
    )
    
    assert Path(results["loss_curve"]).exists()
    assert Path(results["auc_curve"]).exists()
    assert Path(results["loss_curve"]).stat().st_size > 5000
    assert Path(results["auc_curve"]).stat().st_size > 5000

    with Image.open(results["loss_curve"]) as img:
        assert img.format == "PNG"
    with Image.open(results["auc_curve"]) as img:
        assert img.format == "PNG"


def test_plot_finetune_curves_custom(tmp_path):
    """Verify custom fine-tuning epoch history renders properly."""
    loss_file = tmp_path / "custom_ft_loss.png"
    auc_file = tmp_path / "custom_ft_auc.png"
    epochs = [1, 2, 3]
    train_losses = [0.8, 0.6, 0.4]
    val_losses = [0.85, 0.65, 0.45]
    val_aucs = [0.65, 0.75, 0.82]

    results = plot_finetune_curves(
        epochs=epochs,
        train_losses=train_losses,
        val_losses=val_losses,
        val_aucs=val_aucs,
        loss_path=loss_file,
        auc_path=auc_file,
        dpi=100
    )

    assert Path(results["loss_curve"]).exists()
    assert Path(results["auc_curve"]).exists()
