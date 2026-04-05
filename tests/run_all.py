#!/usr/bin/env python3
"""Run all test suites.

Usage:
    python3 tests/run_all.py              # Run all tests
    python3 tests/run_all.py --local      # Local tests only ($0, 30 sec)
    python3 tests/run_all.py --integration # Integration tests only (~$0.10, 20 min)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent


def run_suite(name, script):
    print(f"\n{'#' * 60}")
    print(f"# {name}")
    print(f"{'#' * 60}\n")
    start = time.time()
    result = subprocess.run(
        [sys.executable, str(TESTS_DIR / script)],
        cwd=str(TESTS_DIR),
    )
    elapsed = time.time() - start
    status = "PASSED" if result.returncode == 0 else "FAILED"
    return result.returncode == 0, elapsed, status


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--local", action="store_true", help="Run local tests only")
    p.add_argument("--integration", action="store_true", help="Run integration tests only")
    args = p.parse_args()

    # Default: run all
    run_local = not args.integration or args.local
    run_integration = not args.local or args.integration
    if not args.local and not args.integration:
        run_local = True
        run_integration = True

    results = []

    if run_local:
        ok, elapsed, status = run_suite("Local Tests (no AWS cost)", "test_local.py")
        results.append(("Local", ok, elapsed, status))

    if run_integration:
        ok, elapsed, status = run_suite("Integration Tests (real SageMaker jobs)", "test_integration.py")
        results.append(("Integration", ok, elapsed, status))

    # Summary
    print(f"\n{'#' * 60}")
    print(f"# Final Summary")
    print(f"{'#' * 60}")
    total_time = sum(e for _, _, e, _ in results)
    all_passed = all(ok for _, ok, _, _ in results)

    for name, ok, elapsed, status in results:
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name}: {status} ({elapsed:.0f}s)")

    print(f"\n  Total time: {total_time:.0f}s")
    print(f"  Overall: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
