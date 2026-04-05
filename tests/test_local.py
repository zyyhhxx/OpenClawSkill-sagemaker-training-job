#!/usr/bin/env python3
"""Test 1: Local — verify scripts work without hitting AWS.

Tests:
  - test_train.py runs locally and produces model.json
  - sagemaker_train.py --dry-run produces valid config
  - sagemaker_cost.py --instance-type estimation works
  - sagemaker_list.py runs without error (may return empty)

Cost: $0 (no AWS resources used, except list call).
"""

import json
import shutil
import tempfile
from pathlib import Path

from helpers import (
    SCRIPTS_DIR, PYTHON, TestResult,
    get_config, run_script, assert_file_exists, assert_json_field
)


def test_local_training_script():
    """test_train.py should run locally and produce model.json."""
    t = TestResult("Local training script")
    model_dir = Path(tempfile.mkdtemp())

    try:
        result = run_script("test_train.py", [
            "--epochs", "3",
            "--model-dir", str(model_dir),
        ], timeout=30)

        t.check(result.returncode == 0, f"Exit code 0 (got {result.returncode})")
        t.check(assert_file_exists(model_dir / "model.json", "model.json"), "model.json created")
        t.check(
            assert_json_field(model_dir / "model.json", "type", "smoke-test", "model type"),
            "Model type = smoke-test"
        )
        t.check(
            assert_json_field(model_dir / "model.json", "epochs", 3, "epochs"),
            "Epochs = 3"
        )
        t.check("Training complete" in result.stdout, "Training complete in stdout")
        t.check("test_loss=0.4200;" in result.stdout, "Metrics in parseable format")
    finally:
        shutil.rmtree(model_dir, ignore_errors=True)

    return t.summary()


def test_dry_run():
    """sagemaker_train.py --dry-run should produce valid config without hitting AWS."""
    t = TestResult("Dry run config generation")
    cfg = get_config()

    result = run_script("sagemaker_train.py", [
        "--job-name", "dry-run-test",
        "--script", str(SCRIPTS_DIR / "test_train.py"),
        "--source-dir", str(SCRIPTS_DIR),
        "--role", cfg["role"],
        "--bucket", cfg["bucket"],
        "--region", cfg["region"],
        "--instance-type", "ml.m5.large",
        "--framework", "sklearn",
        "--max-runtime", "300",
        "--hyperparameters", '{"epochs":"5"}',
        "--dry-run",
    ], timeout=30)

    t.check(result.returncode == 0, f"Exit code 0 (got {result.returncode})")
    t.check("DRY RUN" in result.stdout, "DRY RUN header present")
    t.check('"TrainingJobName": "dry-run-test"' in result.stdout, "Job name in config")
    t.check('"epochs": "5"' in result.stdout, "Hyperparameters in config")
    t.check("sagemaker_program" in result.stdout, "Entry point set")
    t.check("sourcedir.tar.gz" in result.stdout, "Source package referenced")

    return t.summary()


def test_dry_run_pytorch():
    """Dry run with PyTorch framework resolves correct image."""
    t = TestResult("Dry run — PyTorch image resolution")
    cfg = get_config()

    result = run_script("sagemaker_train.py", [
        "--job-name", "dry-run-pytorch",
        "--script", str(SCRIPTS_DIR / "test_train.py"),
        "--source-dir", str(SCRIPTS_DIR),
        "--role", cfg["role"],
        "--bucket", cfg["bucket"],
        "--region", cfg["region"],
        "--instance-type", "ml.g5.xlarge",
        "--framework", "pytorch",
        "--max-runtime", "300",
        "--dry-run",
    ], timeout=30)

    t.check(result.returncode == 0, f"Exit code 0 (got {result.returncode})")
    t.check("pytorch-training" in result.stdout, "PyTorch training image resolved")
    t.check("gpu" in result.stdout, "GPU image selected for g5 instance")

    return t.summary()


def test_dry_run_spot():
    """Dry run with --spot sets spot training config."""
    t = TestResult("Dry run — spot training config")
    cfg = get_config()

    result = run_script("sagemaker_train.py", [
        "--job-name", "dry-run-spot",
        "--script", str(SCRIPTS_DIR / "test_train.py"),
        "--source-dir", str(SCRIPTS_DIR),
        "--role", cfg["role"],
        "--bucket", cfg["bucket"],
        "--region", cfg["region"],
        "--instance-type", "ml.m5.large",
        "--framework", "sklearn",
        "--spot",
        "--spot-max-wait", "3600",
        "--dry-run",
    ], timeout=30)

    t.check(result.returncode == 0, f"Exit code 0 (got {result.returncode})")
    t.check('"EnableManagedSpotTraining": true' in result.stdout, "Spot training enabled")
    t.check('"MaxWaitTimeInSeconds": 3600' in result.stdout, "Spot max wait set")

    return t.summary()


def test_dry_run_input_data():
    """Dry run with --input-data sets input channels."""
    t = TestResult("Dry run — input data channels")
    cfg = get_config()

    result = run_script("sagemaker_train.py", [
        "--job-name", "dry-run-input",
        "--script", str(SCRIPTS_DIR / "test_train.py"),
        "--source-dir", str(SCRIPTS_DIR),
        "--role", cfg["role"],
        "--bucket", cfg["bucket"],
        "--region", cfg["region"],
        "--instance-type", "ml.m5.large",
        "--framework", "sklearn",
        "--input-data", f"train:s3://{cfg['bucket']}/data/train/",
        "--input-data", f"validation:s3://{cfg['bucket']}/data/val/",
        "--dry-run",
    ], timeout=30)

    t.check(result.returncode == 0, f"Exit code 0 (got {result.returncode})")
    t.check('"ChannelName": "train"' in result.stdout, "Train channel configured")
    t.check('"ChannelName": "validation"' in result.stdout, "Validation channel configured")

    return t.summary()


def test_dry_run_custom_image():
    """Dry run with --image-uri bypasses framework resolution."""
    t = TestResult("Dry run — custom image URI")
    cfg = get_config()

    custom_image = "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-custom-image:latest"
    result = run_script("sagemaker_train.py", [
        "--job-name", "dry-run-custom-image",
        "--script", str(SCRIPTS_DIR / "test_train.py"),
        "--source-dir", str(SCRIPTS_DIR),
        "--role", cfg["role"],
        "--bucket", cfg["bucket"],
        "--region", cfg["region"],
        "--instance-type", "ml.m5.large",
        "--image-uri", custom_image,
        "--dry-run",
    ], timeout=30)

    t.check(result.returncode == 0, f"Exit code 0 (got {result.returncode})")
    t.check(custom_image in result.stdout, "Custom image URI in config")

    return t.summary()


def test_dry_run_env_and_tags():
    """Dry run with --env and --tags passes them to config."""
    t = TestResult("Dry run — environment variables and tags")
    cfg = get_config()

    result = run_script("sagemaker_train.py", [
        "--job-name", "dry-run-env-tags",
        "--script", str(SCRIPTS_DIR / "test_train.py"),
        "--source-dir", str(SCRIPTS_DIR),
        "--role", cfg["role"],
        "--bucket", cfg["bucket"],
        "--region", cfg["region"],
        "--instance-type", "ml.m5.large",
        "--framework", "sklearn",
        "--env", '{"MY_VAR":"hello","DEBUG":"1"}',
        "--tags", '{"project":"test","team":"ml"}',
        "--dry-run",
    ], timeout=30)

    t.check(result.returncode == 0, f"Exit code 0 (got {result.returncode})")
    t.check('"MY_VAR": "hello"' in result.stdout, "Env var MY_VAR set")
    t.check('"project"' in result.stdout and '"test"' in result.stdout, "Tag 'project' set")

    return t.summary()


def test_cost_estimate():
    """sagemaker_cost.py should estimate cost from instance type + duration."""
    t = TestResult("Cost estimation (local)")

    result = run_script("sagemaker_cost.py", [
        "--instance-type", "ml.g5.xlarge",
        "--duration", "3600",
    ], timeout=10)

    t.check(result.returncode == 0, f"Exit code 0 (got {result.returncode})")
    t.check("Estimated cost: $" in result.stdout, "Cost estimate present")
    t.check("$1.41" in result.stdout or "$1.4" in result.stdout, "g5.xlarge rate ~$1.41/hr")

    # Spot estimate
    result_spot = run_script("sagemaker_cost.py", [
        "--instance-type", "ml.g5.xlarge",
        "--duration", "3600",
        "--spot",
    ], timeout=10)

    t.check(result_spot.returncode == 0, "Spot estimate succeeds")
    # Spot should be cheaper
    t.check("spot" in result_spot.stdout.lower(), "Spot mentioned in output")

    return t.summary()


def test_list_jobs():
    """sagemaker_list.py should run without error."""
    t = TestResult("List jobs")
    cfg = get_config()

    result = run_script("sagemaker_list.py", [
        "--max", "3",
        "--region", cfg["region"],
    ], timeout=30)

    t.check(result.returncode == 0, f"Exit code 0 (got {result.returncode})")

    return t.summary()


def main():
    tests = [
        test_local_training_script,
        test_dry_run,
        test_dry_run_pytorch,
        test_dry_run_spot,
        test_dry_run_input_data,
        test_dry_run_custom_image,
        test_dry_run_env_and_tags,
        test_cost_estimate,
        test_list_jobs,
    ]

    print("=" * 60)
    print("Test Suite 1: Local Tests (no SageMaker jobs submitted)")
    print("=" * 60)

    results = []
    for test_fn in tests:
        print(f"\n--- {test_fn.__doc__.strip().split(chr(10))[0]} ---")
        results.append(test_fn())

    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"Local tests: {passed}/{total} passed")
    if passed < total:
        exit(1)


if __name__ == "__main__":
    main()
