#!/usr/bin/env python3
"""
PharmaGNN v2 Foundation Model: Master End-to-End Orchestrator Pipeline
======================================================================
Coordinates and executes all 5 stages of the PharmaGNN lifecycle:
  Stage 1: Dataset Curation & ETL (scripts/phase1_build_dataset.py)
  Stage 2: Self-Supervised Pretraining (scripts/phase2_pretrain.py)
  Stage 3: Multi-Task Downstream Fine-Tuning (scripts/phase3_finetune.py)
  Stage 4: Model Calibration & Chemical Sanity (scripts/phase4_calibration.py)
  Stage 5: Production Deployment & Serving Verification (scripts/phase5_deployment.py)

Usage examples:
  # Run entire pipeline from start to finish
  python run_pipeline.py

  # Run quick smoke test across all stages
  python run_pipeline.py --smoke-test

  # Run from Stage 3 onwards (reuse existing pretrain & datasets)
  python run_pipeline.py --from-stage 3

  # Run specific stages (e.g., Calibration and Deployment only)
  python run_pipeline.py --stages 4 5

  # Force complete pretraining from scratch
  python run_pipeline.py --force-pretrain
"""

import sys
import os
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional


STAGES_MAP = {
    1: {
        "name": "Phase 1: Dataset Curation & ETL",
        "script": "scripts/phase1_build_dataset.py",
        "description": "Download, clean, and merge Tox21, ClinTox, and curated reference compounds.",
        "artifacts": [
            "data/processed/master_toxicity_dataset.csv",
            "data/processed/metadata.json"
        ]
    },
    2: {
        "name": "Phase 2: Self-Supervised Pretraining",
        "script": "scripts/phase2_pretrain.py",
        "description": "Multi-task self-supervised pretraining (atom/bond/motif/context) for GATv2 backbone.",
        "artifacts": [
            "pharma_gnn_pretrained_encoder.pt",
            "logs/phase2_pretrain.log"
        ]
    },
    3: {
        "name": "Phase 3: Multi-Task Fine-Tuning",
        "script": "scripts/phase3_finetune.py",
        "description": "Multi-task fine-tuning across 13 endpoints with focal loss & two-phase schedule.",
        "artifacts": [
            "pharma_gnn_finetuned.pt",
            "logs/phase3_finetune.log"
        ]
    },
    4: {
        "name": "Phase 4: Calibration & Chemical Sanity",
        "script": "scripts/phase4_calibration.py",
        "description": "Temperature scaling (ECE optimization) and 10-compound chemical sanity checks.",
        "artifacts": [
            "configs/calibration_info.json"
        ]
    },
    5: {
        "name": "Phase 5: Production Deployment & Serving",
        "script": "scripts/phase5_deployment.py",
        "description": "Production state_dict export, model config, OpenAPI spec, and live FastAPI validation.",
        "artifacts": [
            "configs/model_config.json",
            "configs/api_spec.json",
            "configs/monitoring_config.json",
            "configs/deployment_summary.json",
            "Dockerfile.production"
        ]
    }
}


def get_default_python_bin() -> str:
    """Return project virtualenv python if available, otherwise sys.executable."""
    repo_root = Path(__file__).parent.absolute()
    venv_py = repo_root / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def ensure_virtualenv():
    """
    If run under an external Python environment lacking required dependencies (e.g. torch),
    automatically re-exec under the project's local .venv Python environment.
    """
    repo_root = Path(__file__).parent.absolute()
    venv_py = repo_root / ".venv" / "bin" / "python"
    if venv_py.exists():
        try:
            import torch
        except ImportError:
            print(f"[!] Active Python ({sys.executable}) lacks project dependencies (torch).")
            print(f"[+] Auto-switching to project virtual environment: {venv_py}\n")
            env = os.environ.copy()
            env["VIRTUAL_ENV"] = str(repo_root / ".venv")
            env["PATH"] = f"{repo_root / '.venv' / 'bin'}:{env.get('PATH', '')}"
            os.execve(str(venv_py), [str(venv_py)] + sys.argv, env)


def parse_pipeline_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for the master pipeline orchestrator."""
    parser = argparse.ArgumentParser(
        description="PharmaGNN v2 Foundation Model: End-to-End Master Pipeline Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    group_stages = parser.add_argument_group("Stage Selection")
    group_stages.add_argument(
        "--stages",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Specific stage numbers to execute (e.g., --stages 3 4 5)"
    )
    group_stages.add_argument(
        "--from-stage",
        type=int,
        default=None,
        help="Execute all stages starting from this stage number up to 5"
    )

    group_mode = parser.add_argument_group("Execution Modes & Controls")
    group_mode.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run lightweight verification mode across selected stages"
    )
    group_mode.add_argument(
        "--force-pretrain",
        action="store_true",
        help="Force re-running Phase 2 pretraining even if checkpoint already exists"
    )
    group_mode.add_argument(
        "--force-dataset",
        action="store_true",
        help="Force re-building Phase 1 master dataset even if CSV already exists"
    )
    group_mode.add_argument(
        "--pretrain-limit",
        type=int,
        default=100000,
        help="Number of compounds to sample for Phase 2 pretraining (default: 100,000)"
    )
    group_mode.add_argument(
        "--pretrain-epochs",
        type=int,
        default=20,
        help="Number of Phase 2 pretraining epochs (default: 20)"
    )

    group_arch = parser.add_argument_group("Model Architecture Parameters")
    group_arch.add_argument(
        "--hidden-channels",
        type=int,
        default=128,
        help="GATv2 latent representation dimension (128 for v2 Foundation)"
    )
    group_arch.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of GATv2 message-passing layers (4 for v2 Foundation)"
    )
    group_arch.add_argument(
        "--heads",
        type=int,
        default=4,
        help="Number of multi-head graph attention heads (4 for v2 Foundation)"
    )

    group_paths = parser.add_argument_group("Weights & Checkpoints")
    group_paths.add_argument(
        "--pretrained-weights",
        type=str,
        default="pharma_gnn_pretrained_encoder.pt",
        help="Pretrained encoder checkpoint path"
    )
    group_paths.add_argument(
        "--finetuned-weights",
        type=str,
        default="pharma_gnn_finetuned.pt",
        help="Fine-tuned model checkpoint path"
    )
    group_paths.add_argument(
        "--python-bin",
        type=str,
        default=get_default_python_bin(),
        help="Python executable to invoke stage subprocesses"
    )
    group_paths.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory where stage logs are saved"
    )

    return parser.parse_args(argv)


def resolve_stages(stages: Optional[List[int]], from_stage: Optional[int]) -> List[int]:
    """Resolve the ordered list of stages to run."""
    valid_stages = set(STAGES_MAP.keys())

    if from_stage is not None:
        if from_stage not in valid_stages:
            raise ValueError(f"Invalid --from-stage: {from_stage}. Must be between 1 and {max(valid_stages)}.")
        return [s for s in range(from_stage, max(valid_stages) + 1)]

    if stages is not None:
        for s in stages:
            if s not in valid_stages:
                raise ValueError(f"Invalid stage '{s}' in --stages. Allowed stages: {sorted(list(valid_stages))}.")
        # Preserve user specified order or return unique stages
        resolved = []
        for s in stages:
            if s not in resolved:
                resolved.append(s)
        return resolved

    return [1, 2, 3, 4, 5]


def format_duration(seconds: float) -> str:
    """Format duration in seconds to clean human-readable representation."""
    if seconds < 0:
        return "0.0s"
    if seconds >= 3600:
        hrs = int(seconds // 3600)
        rem = seconds % 3600
        mins = int(rem // 60)
        secs = int(rem % 60)
        return f"{hrs}h {mins:02d}m {secs:02d}s"
    elif seconds >= 60:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs:02d}s"
    else:
        return f"{seconds:.1f}s"


def should_skip_stage(
    stage: int,
    args: argparse.Namespace,
    pretrained_exists: Optional[bool] = None,
    dataset_exists: Optional[bool] = None
) -> Tuple[bool, str]:
    """
    Determine if a stage can be safely skipped (e.g. checkpoint reuse or existing dataset).
    """
    if stage == 1:
        exists = dataset_exists if dataset_exists is not None else Path("data/processed/master_toxicity_dataset.csv").exists()
        if exists and not args.force_dataset:
            return True, "Master dataset already exists at data/processed/master_toxicity_dataset.csv"

    if stage == 2:
        exists = pretrained_exists if pretrained_exists is not None else Path(args.pretrained_weights).exists()
        if exists and not args.force_pretrain and not args.smoke_test:
            return True, f"Found existing checkpoint at {args.pretrained_weights}"

    return False, ""


def build_stage_command(stage: int, args: argparse.Namespace) -> List[str]:
    """Construct the command line arguments for invoking a specific stage script."""
    stage_info = STAGES_MAP[stage]
    script_path = stage_info["script"]

    if stage == 1:
        cmd = [args.python_bin, script_path]

    elif stage == 2:
        cmd = [
            args.python_bin, script_path,
            "--limit", str(args.pretrain_limit),
            "--epochs", str(args.pretrain_epochs),
            "--hidden-channels", str(args.hidden_channels),
            "--num-layers", str(args.num_layers),
            "--heads", str(args.heads),
            "--output-weights", args.pretrained_weights
        ]
        if args.smoke_test:
            cmd.append("--smoke-test")

    elif stage == 3:
        cmd = [
            args.python_bin, script_path,
            "--hidden-channels", str(args.hidden_channels),
            "--num-layers", str(args.num_layers),
            "--heads", str(args.heads),
            "--pretrained-weights", args.pretrained_weights,
            "--output-weights", args.finetuned_weights
        ]
        if args.smoke_test:
            cmd.append("--smoke-test")

    elif stage == 4:
        cmd = [
            args.python_bin, script_path,
            "--weights-path", args.finetuned_weights,
            "--hidden-channels", str(args.hidden_channels),
            "--num-layers", str(args.num_layers),
            "--heads", str(args.heads),
            "--output-json", "configs/calibration_info.json"
        ]
        if args.smoke_test:
            cmd.append("--smoke-test")

    elif stage == 5:
        cmd = [
            args.python_bin, script_path,
            "--weights-path", args.finetuned_weights,
            "--hidden-channels", str(args.hidden_channels),
            "--num-layers", str(args.num_layers),
            "--heads", str(args.heads)
        ]
        if args.smoke_test:
            cmd.append("--smoke-test")

    else:
        raise ValueError(f"Unknown stage {stage}")

    return cmd


def run_stage(stage: int, args: argparse.Namespace) -> Dict:
    """Execute a single pipeline stage and track metrics, artifacts, and outcome."""
    stage_info = STAGES_MAP[stage]
    stage_name = stage_info["name"]

    print("\n" + "=" * 80)
    print(f"▶ EXECUTING {stage_name.upper()}")
    print(f"  Description : {stage_info['description']}")
    print("=" * 80)

    # Check skip conditions
    skip, skip_reason = should_skip_stage(stage, args)
    if skip:
        print(f"\n[⏭] STAGE SKIPPED: {skip_reason}")
        print(f"    (Use appropriate --force-* flag to force execution)")
        return {
            "stage": stage,
            "name": stage_name,
            "status": "SKIPPED",
            "duration": 0.0,
            "reason": skip_reason,
            "artifacts": stage_info["artifacts"]
        }

    cmd = build_stage_command(stage, args)
    print(f"  Command: {' '.join(cmd)}\n")

    start_time = time.time()
    try:
        # Prepare environment ensuring virtualenv bin and root are present
        env = os.environ.copy()
        repo_root = Path(__file__).parent.absolute()
        venv_dir = repo_root / ".venv"
        if venv_dir.exists():
            env["VIRTUAL_ENV"] = str(venv_dir)
            env["PATH"] = f"{venv_dir / 'bin'}:{env.get('PATH', '')}"

        # Run process streaming output to stdout in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )

        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line)
            sys.stdout.flush()

        process.wait()
        duration = time.time() - start_time

        if process.returncode != 0:
            print(f"\n[✖] Stage {stage} failed with exit code {process.returncode}")
            return {
                "stage": stage,
                "name": stage_name,
                "status": "FAILED",
                "duration": duration,
                "reason": f"Non-zero exit code: {process.returncode}",
                "artifacts": []
            }

        # Check expected artifacts
        produced_artifacts = []
        for art in stage_info["artifacts"]:
            if Path(art).exists():
                produced_artifacts.append(art)

        print(f"\n[✓] Stage {stage} completed successfully in {format_duration(duration)}")
        return {
            "stage": stage,
            "name": stage_name,
            "status": "SUCCESS",
            "duration": duration,
            "reason": "OK",
            "artifacts": produced_artifacts
        }

    except Exception as exc:
        duration = time.time() - start_time
        print(f"\n[✖] Exception while executing Stage {stage}: {exc}")
        return {
            "stage": stage,
            "name": stage_name,
            "status": "FAILED",
            "duration": duration,
            "reason": str(exc),
            "artifacts": []
        }


def print_summary_table(results: List[Dict], total_duration: float, all_success: bool):
    """Print an executive summary table of the pipeline run."""
    print("\n\n" + "=" * 80)
    print("PHARMAGNN FOUNDATION PIPELINE SUMMARY".center(80))
    print("=" * 80)
    print(f"{'Stage':<8} {'Stage Name':<38} {'Status':<10} {'Duration':<10} {'Artifacts'}")
    print("-" * 80)

    for res in results:
        status_symbol = {
            "SUCCESS": "✓ SUCCESS",
            "SKIPPED": "⏭ SKIPPED",
            "FAILED":  "✖ FAILED"
        }.get(res["status"], res["status"])

        art_count = len(res.get("artifacts", []))
        art_str = f"{art_count} artifact(s)" if art_count > 0 else (res.get("reason", "-")[:18])

        print(
            f"Stage {res['stage']:<2} {res['name']:<38} {status_symbol:<10} "
            f"{format_duration(res['duration']):<10} {art_str}"
        )

    print("-" * 80)
    print(f"Total Pipeline Runtime : {format_duration(total_duration)}")
    if all_success:
        print("Pipeline Status        : ALL STAGES COMPLETED SUCCESSFULLY [✓]")
    else:
        print("Pipeline Status        : PIPELINE FAILED OR TERMINATED EARLY [✖]")
    print("=" * 80 + "\n")


def run_pipeline(args: argparse.Namespace) -> bool:
    """Execute the configured pipeline stages sequentially."""
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    stages = resolve_stages(args.stages, args.from_stage)

    # In smoke test mode, isolate intermediate checkpoints to avoid overwriting production weights
    backup_cfg = None
    if args.smoke_test:
        if args.pretrained_weights == "pharma_gnn_pretrained_encoder.pt" and 2 in stages:
            args.pretrained_weights = "pharma_gnn_smoke_encoder.pt"
        if args.finetuned_weights == "pharma_gnn_finetuned.pt" and 3 in stages:
            args.finetuned_weights = "pharma_gnn_smoke_finetuned.pt"
        cfg_target = Path("configs/model_config.json")
        if not cfg_target.exists() and Path("model_config.json").exists():
            cfg_target = Path("model_config.json")
        if cfg_target.exists():
            backup_cfg = (cfg_target, cfg_target.read_text())

    print("=" * 80)
    print(" PharmaGNN v2 Foundation Model: Master Pipeline Orchestrator ".center(80, "#"))
    print("=" * 80)
    print(f"  • Selected Stages   : {stages}")
    print(f"  • Smoke-Test Mode   : {args.smoke_test}")
    print(f"  • Hidden Channels   : {args.hidden_channels}")
    print(f"  • GATv2 Layers      : {args.num_layers}")
    print(f"  • Attention Heads   : {args.heads}")
    if 2 in stages:
        print(f"  • Pretrain Scale    : {args.pretrain_limit:,} compounds ({args.pretrain_epochs} epochs)")
    print(f"  • Pretrain Weights  : {args.pretrained_weights}")
    print(f"  • Finetune Weights  : {args.finetuned_weights}")
    print(f"  • Python Binary     : {args.python_bin}")
    print("=" * 80)

    pipeline_start = time.time()
    results = []
    all_success = True

    try:
        for stage_num in stages:
            res = run_stage(stage_num, args)
            results.append(res)

            if res["status"] == "FAILED":
                all_success = False
                print(f"\n[!] Aborting subsequent pipeline stages due to failure in Stage {stage_num}.")
                break
    finally:
        # In smoke-test mode, restore production model_config.json if it was backed up
        if args.smoke_test and backup_cfg is not None:
            target_p, content = backup_cfg
            target_p.write_text(content)

    total_duration = time.time() - pipeline_start
    print_summary_table(results, total_duration, all_success)
    return all_success


def main(argv: Optional[List[str]] = None):
    ensure_virtualenv()
    args = parse_pipeline_args(argv)
    success = run_pipeline(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
