#!/usr/bin/env python3
"""Test 2: Integration — submit real training jobs to SageMaker.

Tests:
  - sklearn container on ml.m5.large (cheapest CPU)
  - Spot training on ml.m5.large
  - PyTorch container on ml.m5.large (CPU, verifies image resolution)
  - Input data channel (upload CSV to S3, pass via --input-data)
  - Resume flow (--no-wait then --resume)
  - Multiple hyperparameters and epochs
  - Cost check against completed jobs

Cost: ~$0.05-0.10 total (5 cheap jobs, ~2-3 min each).
Time: ~15-20 min (jobs run sequentially).
"""

import csv
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from helpers import (
    SCRIPTS_DIR, PYTHON, TestResult,
    get_config, job_name, run_script, run_script_live,
    upload_to_s3, cleanup_s3_prefix,
    assert_file_exists, assert_json_field,
)


def test_sklearn_basic():
    """Submit a basic sklearn training job and verify artifacts."""
    t = TestResult("sklearn basic job")
    cfg = get_config()
    name = job_name("test-sklearn")
    output_dir = Path(tempfile.mkdtemp())

    try:
        rc = run_script_live("sagemaker_train.py", [
            "--job-name", name,
            "--script", str(SCRIPTS_DIR / "test_train.py"),
            "--source-dir", str(SCRIPTS_DIR),
            "--role", cfg["role"],
            "--bucket", cfg["bucket"],
            "--region", cfg["region"],
            "--instance-type", "ml.m5.large",
            "--framework", "sklearn",
            "--max-runtime", "300",
            "--hyperparameters", '{"epochs":"2"}',
            "--output-dir", str(output_dir),
            "--poll-interval", "15",
        ], timeout=600)

        t.check(rc == 0, f"Job completed successfully (exit {rc})")
        t.check(
            assert_file_exists(output_dir / "model" / "model.json", "model.json"),
            "Model artifact downloaded"
        )
        t.check(
            assert_json_field(output_dir / "model" / "model.json", "type", "smoke-test"),
            "Model type correct"
        )
        t.check(
            assert_json_field(output_dir / "model" / "model.json", "epochs", 2),
            "Epochs = 2"
        )

        # Cost check
        result = run_script("sagemaker_cost.py", [
            "--job-name", name, "--region", cfg["region"],
        ], timeout=30)
        t.check(result.returncode == 0, "Cost check succeeded")
        t.check("Estimated cost: $" in result.stdout, "Cost reported")

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)

    return t.summary()


def test_spot_training():
    """Submit a job with --spot and verify it uses spot instances."""
    t = TestResult("Spot training")
    cfg = get_config()
    name = job_name("test-spot")
    output_dir = Path(tempfile.mkdtemp())

    try:
        rc = run_script_live("sagemaker_train.py", [
            "--job-name", name,
            "--script", str(SCRIPTS_DIR / "test_train.py"),
            "--source-dir", str(SCRIPTS_DIR),
            "--role", cfg["role"],
            "--bucket", cfg["bucket"],
            "--region", cfg["region"],
            "--instance-type", "ml.m5.large",
            "--framework", "sklearn",
            "--max-runtime", "300",
            "--spot",
            "--spot-max-wait", "600",
            "--hyperparameters", '{"epochs":"1"}',
            "--output-dir", str(output_dir),
            "--poll-interval", "15",
        ], timeout=600)

        t.check(rc == 0, f"Spot job completed (exit {rc})")
        t.check(
            assert_file_exists(output_dir / "model" / "model.json"),
            "Model artifact downloaded"
        )

        # Verify it was actually spot
        result = run_script("sagemaker_cost.py", [
            "--job-name", name, "--region", cfg["region"],
        ], timeout=30)
        t.check("Spot: Yes" in result.stdout, "Confirmed spot instance used")

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)

    return t.summary()


def test_pytorch_cpu():
    """Submit a job with PyTorch framework on CPU instance."""
    t = TestResult("PyTorch CPU job")
    cfg = get_config()
    name = job_name("test-pytorch")
    output_dir = Path(tempfile.mkdtemp())

    try:
        rc = run_script_live("sagemaker_train.py", [
            "--job-name", name,
            "--script", str(SCRIPTS_DIR / "test_train.py"),
            "--source-dir", str(SCRIPTS_DIR),
            "--role", cfg["role"],
            "--bucket", cfg["bucket"],
            "--region", cfg["region"],
            "--instance-type", "ml.m5.large",
            "--framework", "pytorch",
            "--max-runtime", "300",
            "--hyperparameters", '{"epochs":"1"}',
            "--output-dir", str(output_dir),
            "--poll-interval", "15",
        ], timeout=600)

        t.check(rc == 0, f"PyTorch job completed (exit {rc})")
        t.check(
            assert_file_exists(output_dir / "model" / "model.json"),
            "Model artifact downloaded"
        )

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)

    return t.summary()


def test_input_data_channel():
    """Upload data to S3, pass via --input-data, verify script receives it."""
    t = TestResult("Input data channel")
    cfg = get_config()
    name = job_name("test-input")
    output_dir = Path(tempfile.mkdtemp())
    local_data = Path(tempfile.mkdtemp())

    # Create a training script that reads input data
    train_script = local_data / "train_with_data.py"
    train_script.write_text('''#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
args = parser.parse_args()

train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
print(f"[test] Train dir: {train_dir}")
print(f"[test] Files: {os.listdir(train_dir)}")

# Read the CSV
data_file = os.path.join(train_dir, "data.csv")
with open(data_file) as f:
    lines = f.readlines()
row_count = len(lines) - 1  # subtract header
print(f"[test] Rows read: {row_count}")

os.makedirs(args.model_dir, exist_ok=True)
with open(os.path.join(args.model_dir, "model.json"), "w") as f:
    json.dump({"type": "input-data-test", "rows_read": row_count}, f)
print("[test] Training complete.")
''')

    # Create test CSV
    csv_path = local_data / "data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature1", "feature2", "target"])
        for i in range(100):
            writer.writerow([i * 0.1, i * 0.2, i % 2])

    try:
        # Upload CSV to S3
        s3_prefix = f"sagemaker/test-data/{name}"
        s3_uri = upload_to_s3(csv_path, cfg["bucket"], f"{s3_prefix}/data.csv", cfg["region"])
        s3_dir = f"s3://{cfg['bucket']}/{s3_prefix}/"
        print(f"  Uploaded test data to {s3_dir}")

        rc = run_script_live("sagemaker_train.py", [
            "--job-name", name,
            "--script", str(train_script),
            "--source-dir", str(local_data),
            "--role", cfg["role"],
            "--bucket", cfg["bucket"],
            "--region", cfg["region"],
            "--instance-type", "ml.m5.large",
            "--framework", "sklearn",
            "--max-runtime", "300",
            "--input-data", f"train:{s3_dir}",
            "--output-dir", str(output_dir),
            "--poll-interval", "15",
        ], timeout=600)

        t.check(rc == 0, f"Job completed (exit {rc})")
        t.check(
            assert_file_exists(output_dir / "model" / "model.json"),
            "Model artifact downloaded"
        )
        t.check(
            assert_json_field(output_dir / "model" / "model.json", "type", "input-data-test"),
            "Correct model type"
        )
        t.check(
            assert_json_field(output_dir / "model" / "model.json", "rows_read", 100),
            "All 100 rows read from S3 input"
        )

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
        shutil.rmtree(local_data, ignore_errors=True)
        cleanup_s3_prefix(cfg["bucket"], s3_prefix, cfg["region"])

    return t.summary()


def test_resume_flow():
    """Submit with --no-wait, then reconnect with --resume."""
    t = TestResult("Resume flow")
    cfg = get_config()
    name = job_name("test-resume")
    output_dir = Path(tempfile.mkdtemp())

    try:
        # Step 1: Submit without waiting
        print("  Step 1: Submit with --no-wait")
        result = run_script("sagemaker_train.py", [
            "--job-name", name,
            "--script", str(SCRIPTS_DIR / "test_train.py"),
            "--source-dir", str(SCRIPTS_DIR),
            "--role", cfg["role"],
            "--bucket", cfg["bucket"],
            "--region", cfg["region"],
            "--instance-type", "ml.m5.large",
            "--framework", "sklearn",
            "--max-runtime", "300",
            "--hyperparameters", '{"epochs":"1"}',
            "--no-wait",
        ], timeout=60)

        t.check(result.returncode == 0, f"Submit succeeded (exit {result.returncode})")
        t.check("Job submitted successfully" in result.stdout, "Submission confirmed")

        # Step 2: Resume
        print("  Step 2: Resume polling")
        rc = run_script_live("sagemaker_train.py", [
            "--resume", name,
            "--bucket", cfg["bucket"],
            "--region", cfg["region"],
            "--output-dir", str(output_dir),
            "--poll-interval", "15",
        ], timeout=600)

        t.check(rc == 0, f"Resume completed (exit {rc})")
        t.check(
            assert_file_exists(output_dir / "model" / "model.json"),
            "Model artifact downloaded via resume"
        )

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)

    return t.summary()


def test_multiple_hyperparameters():
    """Submit with many hyperparameters and verify all are passed."""
    t = TestResult("Multiple hyperparameters")
    cfg = get_config()
    name = job_name("test-hparams")
    output_dir = Path(tempfile.mkdtemp())

    # Training script that dumps hyperparameters
    src_dir = Path(tempfile.mkdtemp())
    script = src_dir / "train_hparams.py"
    script.write_text('''#!/usr/bin/env python3
import argparse, json, os, sys

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
args = parser.parse_args()

print(f"[test] epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}, optimizer={args.optimizer}")

os.makedirs(args.model_dir, exist_ok=True)
with open(os.path.join(args.model_dir, "model.json"), "w") as f:
    json.dump({
        "type": "hparams-test",
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
    }, f)
print("[test] Training complete.")
''')

    try:
        rc = run_script_live("sagemaker_train.py", [
            "--job-name", name,
            "--script", str(script),
            "--source-dir", str(src_dir),
            "--role", cfg["role"],
            "--bucket", cfg["bucket"],
            "--region", cfg["region"],
            "--instance-type", "ml.m5.large",
            "--framework", "sklearn",
            "--max-runtime", "300",
            "--hyperparameters", '{"epochs":"5","lr":"0.01","batch-size":"64","optimizer":"sgd"}',
            "--output-dir", str(output_dir),
            "--poll-interval", "15",
        ], timeout=600)

        t.check(rc == 0, f"Job completed (exit {rc})")
        model_path = output_dir / "model" / "model.json"
        t.check(assert_file_exists(model_path), "Model downloaded")

        if model_path.exists():
            data = json.loads(model_path.read_text())
            t.check(data.get("epochs") == 5, f"epochs=5 (got {data.get('epochs')})")
            t.check(data.get("lr") == 0.01, f"lr=0.01 (got {data.get('lr')})")
            t.check(data.get("batch_size") == 64, f"batch_size=64 (got {data.get('batch_size')})")
            t.check(data.get("optimizer") == "sgd", f"optimizer=sgd (got {data.get('optimizer')})")

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
        shutil.rmtree(src_dir, ignore_errors=True)

    return t.summary()


def main():
    tests = [
        test_sklearn_basic,
        test_spot_training,
        test_pytorch_cpu,
        test_input_data_channel,
        test_resume_flow,
        test_multiple_hyperparameters,
    ]

    print("=" * 60)
    print("Test Suite 2: Integration Tests (real SageMaker jobs)")
    print("  Cost: ~$0.05-0.10 | Time: ~15-20 min")
    print("=" * 60)

    results = []
    for test_fn in tests:
        print(f"\n{'='*60}")
        print(f"--- {test_fn.__doc__.strip().split(chr(10))[0]} ---")
        results.append(test_fn())

    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"Integration tests: {passed}/{total} passed")
    if passed < total:
        exit(1)


if __name__ == "__main__":
    main()
