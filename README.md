# sagemaker-training-job-skill

An [OpenClaw](https://openclaw.ai) / [ClawHub](https://clawhub.ai) skill for submitting ML training jobs to AWS SageMaker.

## What It Does

Packages your training code, uploads it to S3, launches a SageMaker training job on managed GPU/CPU instances, polls until completion, and downloads model artifacts — all from a single command.

## Features

- **Framework support:** PyTorch, TensorFlow, scikit-learn, XGBoost (auto-resolves container images)
- **Spot training:** `--spot` flag for up to 70% cost savings on GPU instances
- **Resume:** `--resume` to reconnect to a running job (closed laptop? switched machines?)
- **Cost estimation:** Check costs before and after training
- **Dry run:** Validate config without spending money
- **Works anywhere:** EC2 (instance profile), personal PC (access keys), or CI/CD

## Scripts

| Script | Purpose |
|--------|---------|
| `sagemaker_train.py` | Submit → poll → download artifacts |
| `sagemaker_list.py` | List recent training jobs |
| `sagemaker_cost.py` | Estimate or check actual costs |
| `sagemaker_smoke_test.py` | Quick end-to-end verification (~$0.01) |
| `test_train.py` | Minimal training script used by tests |

## Quick Start

```bash
# Install the skill
clawhub install sagemaker-training-job

# Submit a training job
python3 scripts/sagemaker_train.py \
  --job-name my-experiment-001 \
  --script ./train.py \
  --role arn:aws:iam::ACCOUNT:role/SageMakerTrainingExecutionRole \
  --bucket my-sagemaker-bucket \
  --instance-type ml.g5.xlarge \
  --spot \
  --framework pytorch \
  --input-data s3://my-bucket/data/train/ \
  --hyperparameters '{"epochs":"50","lr":"0.001"}' \
  --output-dir ./results
```

## Prerequisites

- Python 3.9+ with `boto3` (required) and `sagemaker` (recommended)
- AWS credentials (instance profile, access keys, or SSO)
- Two IAM roles + S3 bucket configured — see `references/setup.md` for exact policies
- Run `sagemaker_smoke_test.py` to verify the setup works

## Documentation

- `SKILL.md` — Agent-facing usage guide
- `references/setup.md` — IAM policies for both roles, S3 bucket setup, instance selection
- `references/training-scripts.md` — Training script contract, templates for PyTorch and LightGBM

## Testing

Tests live in `tests/` (not included in the published skill). Requires AWS credentials
and a configured environment.

### Setup

```bash
# Set required environment variables
export SM_TEST_ROLE=arn:aws:iam::ACCOUNT_ID:role/SageMakerTrainingExecutionRole
export SM_TEST_BUCKET=your-sagemaker-bucket
export SM_TEST_REGION=us-east-1  # optional, defaults to us-east-1
```

### Run tests

```bash
# All tests (local + integration)
python3 tests/run_all.py

# Local tests only — $0, ~30 seconds
# Validates scripts, dry runs, config generation, cost estimation
python3 tests/run_all.py --local

# Integration tests only — ~$0.10, ~20 minutes
# Submits real SageMaker jobs and verifies end-to-end
python3 tests/run_all.py --integration
```

### Test coverage

**Local tests** (`test_local.py`) — no AWS cost:
| Test | What it verifies |
|------|-----------------|
| Local training script | `test_train.py` runs and produces correct model.json |
| Dry run — basic | Config generation with sklearn framework |
| Dry run — PyTorch | GPU image resolution for PyTorch |
| Dry run — spot | `--spot` sets `EnableManagedSpotTraining` and `MaxWaitTimeInSeconds` |
| Dry run — input data | Multiple `--input-data` channels configured correctly |
| Dry run — custom image | `--image-uri` bypasses framework resolution |
| Dry run — env and tags | `--env` and `--tags` passed to job config |
| Cost estimation | Local cost calculation for different instance types + spot |
| List jobs | `sagemaker_list.py` runs without error |

**Integration tests** (`test_integration.py`) — real SageMaker jobs:
| Test | What it verifies | Instance |
|------|-----------------|----------|
| sklearn basic | Submit, poll, download, verify artifacts | ml.m5.large |
| Spot training | `--spot` flag works, confirmed in cost check | ml.m5.large |
| PyTorch CPU | PyTorch container resolves and runs on CPU | ml.m5.large |
| Input data channel | Upload CSV to S3, pass via `--input-data`, script reads it | ml.m5.large |
| Resume flow | `--no-wait` then `--resume` reconnects and downloads | ml.m5.large |
| Multiple hyperparameters | Many hyperparams passed correctly to training script | ml.m5.large |

## License

MIT
