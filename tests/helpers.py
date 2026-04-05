#!/usr/bin/env python3
"""Shared utilities for sagemaker-training-job skill tests.

All tests use these helpers to avoid duplication.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Resolve paths
REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = REPO_ROOT / "sagemaker-training-job"
SCRIPTS_DIR = SKILL_DIR / "scripts"
PYTHON = sys.executable


def get_config():
    """Get test config from environment variables."""
    role = os.environ.get("SM_TEST_ROLE")
    bucket = os.environ.get("SM_TEST_BUCKET")
    region = os.environ.get("SM_TEST_REGION", "us-east-1")

    if not role or not bucket:
        print("ERROR: Set SM_TEST_ROLE and SM_TEST_BUCKET environment variables.", file=sys.stderr)
        print("  export SM_TEST_ROLE=arn:aws:iam::ACCOUNT:role/SageMakerTrainingExecutionRole", file=sys.stderr)
        print("  export SM_TEST_BUCKET=my-sagemaker-bucket", file=sys.stderr)
        sys.exit(1)

    return {"role": role, "bucket": bucket, "region": region}


def job_name(prefix="test"):
    """Generate a unique job name."""
    return f"{prefix}-{int(time.time())}"


def run_script(script_name, args, timeout=600):
    """Run a skill script and return the result."""
    cmd = [PYTHON, str(SCRIPTS_DIR / script_name)] + args
    print(f"  $ {' '.join(cmd[:4])} ...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result


def run_script_live(script_name, args, timeout=600):
    """Run a skill script with live output and return the exit code."""
    cmd = [PYTHON, str(SCRIPTS_DIR / script_name)] + args
    print(f"  $ {' '.join(cmd[:4])} ...")
    result = subprocess.run(cmd, timeout=timeout)
    return result.returncode


def upload_to_s3(local_path, bucket, s3_key, region="us-east-1"):
    """Upload a file to S3 using boto3."""
    import boto3
    s3 = boto3.client("s3", region_name=region)
    s3.upload_file(str(local_path), bucket, s3_key)
    return f"s3://{bucket}/{s3_key}"


def cleanup_s3_prefix(bucket, prefix, region="us-east-1"):
    """Delete all objects under an S3 prefix."""
    import boto3
    s3 = boto3.client("s3", region_name=region)
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            s3.delete_object(Bucket=bucket, Key=obj["Key"])


def assert_file_exists(path, label=""):
    """Assert a file exists, print result."""
    if Path(path).exists():
        print(f"  ✅ {label or path} exists")
        return True
    else:
        print(f"  ❌ {label or path} NOT FOUND")
        return False


def assert_json_field(path, field, expected=None, label=""):
    """Assert a JSON file has a field, optionally with expected value."""
    try:
        data = json.loads(Path(path).read_text())
        value = data.get(field)
        if expected is not None:
            if value == expected:
                print(f"  ✅ {label or field} = {value}")
                return True
            else:
                print(f"  ❌ {label or field} = {value} (expected {expected})")
                return False
        else:
            if value is not None:
                print(f"  ✅ {label or field} = {value}")
                return True
            else:
                print(f"  ❌ {label or field} is missing")
                return False
    except Exception as e:
        print(f"  ❌ Failed to read {path}: {e}")
        return False


class TestResult:
    """Collects pass/fail results for a test."""

    def __init__(self, name):
        self.name = name
        self.checks = []
        self.start_time = time.time()

    def check(self, passed, label):
        self.checks.append((passed, label))
        return passed

    @property
    def passed(self):
        return all(ok for ok, _ in self.checks)

    @property
    def duration(self):
        return time.time() - self.start_time

    def summary(self):
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        failed = [(ok, l) for ok, l in self.checks if not ok]
        print(f"\n{status}: {self.name} ({self.duration:.0f}s)")
        if failed:
            for _, label in failed:
                print(f"  - {label}")
        return self.passed
