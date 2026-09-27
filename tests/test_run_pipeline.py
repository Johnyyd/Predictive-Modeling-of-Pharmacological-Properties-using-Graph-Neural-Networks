import pytest
import sys
import subprocess
import json
from pathlib import Path

# Import functions from run_pipeline
from run_pipeline import (
    parse_pipeline_args,
    resolve_stages,
    build_stage_command,
    should_skip_stage,
    format_duration,
    run_stage,
    STAGES_MAP
)


def test_parse_args_defaults():
    """Verify default CLI arguments."""
    args = parse_pipeline_args([])
    assert args.stages == [1, 2, 3, 4, 5]
    assert args.from_stage is None
    assert args.smoke_test is False
    assert args.force_pretrain is False
    assert args.force_dataset is False
    assert args.hidden_channels == 128
    assert args.num_layers == 4
    assert args.heads == 4
    assert args.pretrain_limit == 100000
    assert args.pretrain_epochs == 20


def test_parse_args_custom():
    """Verify custom CLI flags."""
    args = parse_pipeline_args([
        "--stages", "2", "3",
        "--smoke-test",
        "--force-pretrain",
        "--hidden-channels", "64",
        "--num-layers", "3",
        "--heads", "2"
    ])
    assert args.stages == [2, 3]
    assert args.smoke_test is True
    assert args.force_pretrain is True
    assert args.hidden_channels == 64
    assert args.num_layers == 3
    assert args.heads == 2


def test_resolve_stages_from_stage():
    """Verify --from-stage resolution."""
    assert resolve_stages(stages=None, from_stage=3) == [3, 4, 5]
    assert resolve_stages(stages=None, from_stage=1) == [1, 2, 3, 4, 5]
    assert resolve_stages(stages=None, from_stage=5) == [5]


def test_resolve_stages_explicit():
    """Verify explicit --stages list."""
    assert resolve_stages(stages=[4, 5], from_stage=None) == [4, 5]
    assert resolve_stages(stages=[1, 3, 5], from_stage=None) == [1, 3, 5]


def test_resolve_stages_invalid():
    """Verify ValueError on invalid stage numbers."""
    with pytest.raises(ValueError):
        resolve_stages(stages=[0, 1], from_stage=None)
    with pytest.raises(ValueError):
        resolve_stages(stages=[6], from_stage=None)
    with pytest.raises(ValueError):
        resolve_stages(stages=None, from_stage=7)


def test_build_stage_command():
    """Verify command string generation for all stages."""
    args = parse_pipeline_args(["--smoke-test"])
    
    # Stage 1
    cmd1 = build_stage_command(1, args)
    assert "phase1_build_dataset.py" in cmd1[1]

    # Stage 2
    cmd2 = build_stage_command(2, args)
    assert "phase2_pretrain.py" in cmd2[1]
    assert "--smoke-test" in cmd2
    assert "--hidden-channels" in cmd2
    assert "--limit" in cmd2
    assert "100000" in cmd2

    # Stage 3
    cmd3 = build_stage_command(3, args)
    assert "phase3_finetune.py" in cmd3[1]
    assert "--smoke-test" in cmd3

    # Stage 4
    cmd4 = build_stage_command(4, args)
    assert "phase4_calibration.py" in cmd4[1]
    assert "--smoke-test" in cmd4

    # Stage 5
    cmd5 = build_stage_command(5, args)
    assert "phase5_deployment.py" in cmd5[1]
    assert "--smoke-test" in cmd5


def test_format_duration():
    """Verify human-readable duration formatting."""
    assert format_duration(5.2) == "5.2s"
    assert format_duration(65) == "1m 05s"
    assert format_duration(3665) == "1h 01m 05s"


def test_should_skip_stage_logic(tmp_path):
    """Verify stage skip detection logic."""
    args = parse_pipeline_args([])
    # Fake args with force=False
    # If checkpoint exists, Stage 2 can be skipped unless forced
    can_skip_2, reason_2 = should_skip_stage(2, args, pretrained_exists=True)
    assert can_skip_2 is True
    assert "existing checkpoint" in reason_2.lower()

    # If force_pretrain is set, do not skip
    args_force = parse_pipeline_args(["--force-pretrain"])
    can_skip_2_f, _ = should_skip_stage(2, args_force, pretrained_exists=True)
    assert can_skip_2_f is False

    # Stage 1 skip when dataset exists
    can_skip_1, reason_1 = should_skip_stage(1, args, dataset_exists=True)
    assert can_skip_1 is True
    assert "already exists" in reason_1.lower()


def test_pipeline_smoke_run_subset():
    """Verify executing run_pipeline.py CLI with stages 4 and 5 in smoke-test mode."""
    cfg_target = Path("configs/model_config.json") if Path("configs/model_config.json").exists() else Path("model_config.json")
    backup_cfg = (cfg_target, cfg_target.read_text()) if cfg_target.exists() else None
    try:
        proc = subprocess.run(
            [sys.executable, "run_pipeline.py", "--smoke-test", "--stages", "4", "5"],
            capture_output=True,
            text=True
        )
        assert proc.returncode == 0, f"run_pipeline failed:\n{proc.stdout}\n{proc.stderr}"
        assert "PHARMAGNN FOUNDATION PIPELINE SUMMARY" in proc.stdout
        assert "Phase 4" in proc.stdout
        assert "Phase 5" in proc.stdout
    finally:
        if backup_cfg is not None:
            p, text = backup_cfg
            p.write_text(text)
