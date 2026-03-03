# MLPerf Inference Harness - GPT-OSS-120B

This guide provides instructions for setting up and running MLPerf Inference benchmarks for GPT-OSS-120B using llm-d and the harness framework.

## Table of Contents

1. [LLM-D Setup](#llm-d-setup)
2. [Client Pod Setup](#client-pod-setup)
3. [Running Tests with run_submission.py](#running-tests-with-run_submissionpy)
4. [Creating Submission](#creating-submission)

## LLM-D Setup

The LLM-D framework is used to deploy the GPT-OSS-120B model server in Kubernetes.

### GPU Configurations

This repository provides llm-d configurations for two GPU types:
- **H200**: Located in `setup/llm-d/H200/`
- **B200**: Located in `setup/llm-d/B200/`

Both configurations include:
- Automated deployment script (`deploy_gptoss120b_v050.sh`)
- EPP configuration for optimized scheduling
- Server and Offline mode overrides

**Choose the directory matching your GPU type** when following the instructions below.

### Prerequisites

- Kubernetes cluster with Istio 1.28.1 GA or later
- At least 128 CPUs available on worker nodes (16 CPUs × 8 replicas)
- 8 NVIDIA GPUs available
- HuggingFace token secret `llm-d-hf-token` created in your namespace
- Model artifacts available at the PVC path specified in override files

### Deployment Steps

1. **Clone the repo**   
```bash 
git clone --recurse-submodule https://github.com/openshift-psap/mlperf-inference-6.0-redhat.git
```
2. **Navigate to the setup directory:**
   ```bash
   # For H200 GPUs:
   cd setup/llm-d/H200/

   # For B200 GPUs:
   cd setup/llm-d/B200/
   ```

3. **Deploy GPT-OSS-120B in server mode:**
   ```bash
   bash deploy_gptoss120b_v050.sh server
   ```

   For offline mode:
   ```bash
   bash deploy_gptoss120b_v050.sh offline
   ```

   The script will:
   - Clone the llm-d repository (if not already present)
   - Check out v0.5.0
   - Copy override files to the correct location
   - Deploy infrastructure, GAIE, and model service in order
   - Show pod status after deployment

4. **Update EPP Configuration**
   > ⚠️ Perform this step for Offline only
   
   The EPP (Endpoint Picker Plugin) configuration optimizes request routing for MLPerf workloads:

   ```bash
   # Update EPP with custom configuration
   bash update_epp_config.sh
   ```

   This script:
   - Applies custom EPP configuration with optimized scoring weights
   - Enables kv-cache-utilization-scorer and queue-scorer plugins
   - Restarts the EPP pod to apply changes
   - Sets log verbosity to 7 for better debugging

   For more details, see `EPP_CONFIG_COMPARISON.md` in your GPU directory.

5. **Verify deployment:**
   ```bash
   # Check pod status
   kubectl get pods -n llm-d-bench -l app.kubernetes.io/instance=ms-inference-scheduling
   
   # Watch pod status
   kubectl get pods -n llm-d-bench -l app.kubernetes.io/instance=ms-inference-scheduling -w
   
   # Check logs
   kubectl logs -n llm-d-bench -l app.kubernetes.io/component=decode --tail=50 -f
   ```

6. **Get the API server URL:**
   ```bash
   # Get the service URL
   kubectl get svc -n llm-d-bench
   ```

   The API server URL will typically be in the format:
   ```
   http://<service-name>.<namespace>.svc.cluster.local:8000
   Eg http://infra-inference-scheduling-inference-gateway-istio.llm-d-bench.svc.cluster.local/
   ```
7. **Openshift/Kubernetes and System checks**
   - Ensure ```ulimit -n 65536``` or to a higher value
   - Ensure pod pid limit is also set to a higher value
   
### Environment Variables

You can customize the deployment with environment variables:

- `LLMD_DIR` - Where to clone llm-d (default: `/tmp/llm-d`)
- `NAMESPACE` - Kubernetes namespace (default: `llm-d-bench`)
- `RELEASE_NAME_POSTFIX` - Helm release postfix (default: `inference-scheduling`)

Example with custom settings:
```bash
NAMESPACE=my-namespace LLMD_DIR=/home/user/llm-d bash deploy_gptoss120b_v050.sh server
```

## Client Pod Setup

The client pod is where the MLPerf harness tests will run. This pod needs to be able to connect to the LLM-D API server.

### Setting Up Environment Variables

1. **Clone the repo and environment setup**
   ```bash
   #Clone repo
   git clone --recurse-submodule https://github.com/openshift-psap/mlperf-inference-6.0-redhat.git
   #Setup environment
   #This script clones and repo and sets up  the environment 
   bash setup/client/client_setup.sh 
   ```
   ```bash
   #Set ulimit
   ulimit -n 65536 
   ```
2. **Source the environment variables script:**
   ```bash
   cd harness/scripts
   source set_env_vars.sh
   ```

3. **Set required environment variables:**
   ```bash
   export DATASET_DIR=/path/to/datasets
   export PERF_DATASET=/path/to/perf/perf_eval_ref.parquet
   export ACC_DATASET=/path/to/acc/acc_eval_ref.parquet
   export COMPLIANCE_DATASET=/path/to/acc/acc_eval_compliance_gpqa.parquet
   export OUTPUT_DIR=./harness_output
   export API_SERVER_URL=http://<service-name>.llm-d-bench.svc.cluster.local:8000
   export AWS_ACCESS_KEY_ID=<your-aws-key>
   export AWS_SECRET_ACCESS_KEY=<your-aws-secret>
   export MLFLOW_TRACKING_URI=http://mlflow-server:5000
   export MLFLOW_EXPERIMENT_NAME=<your-experiment-name>
   export SERVER_TARGET_QPS=3  # For Server scenario
   ```

4. **Verify environment variables:**
   ```bash
   # Print current configuration
   print_env_vars
   
   # Validate required variables
   validate_env_vars
   ```

### Additional Configuration

- `HF_HOME` - HuggingFace home directory (if needed)
- `MODEL_CATEGORY` - Model category (default: `gpt-oss-120b`)
- `MODEL` - Model name (default: `openai/gpt-oss-120b`)
- `BACKEND` - Backend type (default: `vllm`)
- `LG_MODEL_NAME` - LoadGen model name (default: `gpt-oss-120b`)

## Running Tests with run_submission.py

The `run_submission.py` script is used to run MLPerf Inference tests. After setting up LLM-D correctly, navigate to the harness directory and run the tests.

### Prerequisites

1. **LLM-D server must be deployed and running** (see [LLM-D Setup](#llm-d-setup))
2. **Navigate to harness directory:**
   ```bash
   cd harness
   ```

3. **Set up environment variables using set_env_vars.sh:**
   ```bash
   # Source the environment variables script
   source scripts/set_env_vars.sh
   
   # Set required environment variables
   export DATASET_DIR=/path/to/datasets
   export PERF_DATASET=/path/to/perf/perf_eval_ref.parquet
   export ACC_DATASET=/path/to/acc/acc_eval_ref.parquet
   export COMPLIANCE_DATASET=/path/to/acc/acc_eval_compliance_gpqa.parquet
   export OUTPUT_DIR=./harness_output
   export API_SERVER_URL=http://<service-name>.llm-d-bench.svc.cluster.local:8000
   export AWS_ACCESS_KEY_ID=<your-aws-key>
   export AWS_SECRET_ACCESS_KEY=<your-aws-secret>
   export MLFLOW_TRACKING_URI=http://mlflow-server:5000
   export MLFLOW_EXPERIMENT_NAME=<your-experiment-name>
   export SERVER_TARGET_QPS=3  # For Server scenario
   
   # Verify environment variables are set correctly
   print_env_vars
   validate_env_vars
   ```

4. **⚠️ IMPORTANT: Check for audit.config before running tests:**
   ```bash
   # Check if audit.config exists in harness directory (should be removed)
   if [ -f "audit.config" ]; then
       echo "WARNING: audit.config found in harness directory. Removing it..."
       rm -f audit.config
   fi
   ```
   
   The `run_submission.py` script automatically cleans up `audit.config` at the beginning, but it's good practice to verify it's not present before starting tests.

### Running Offline Tests

To run all offline tests (performance, accuracy, and compliance):

```bash
python3 scripts/run_submission.py --scenario Offline run-offline
```

This will run:
- Offline Performance test
- Offline Accuracy test
- Offline Compliance TEST07
- Offline Compliance TEST09

### Running Server Tests

To run all server tests (performance, accuracy, and compliance):

```bash
python3 scripts/run_submission.py --scenario Server --server-target-qps 3 run-server
```

This will run:
- Server Performance test
- Server Accuracy test
- Server Compliance TEST07
- Server Compliance TEST09

**Note:** `--server-target-qps` is required for Server scenario.

### Running All Tests

To run all tests for both Server and Offline scenarios:

```bash
python3 scripts/run_submission.py --server-target-qps 3 run-all
```

### Running Specific Test Types

**Performance tests only:**
```bash
# Offline performance
python3 scripts/run_submission.py --scenario Offline run-performance

# Server performance
python3 scripts/run_submission.py --scenario Server --server-target-qps 3 run-performance
```

**Accuracy tests only:**
```bash
# Offline accuracy
python3 scripts/run_submission.py --scenario Offline run-accuracy

# Server accuracy
python3 scripts/run_submission.py --scenario Server --server-target-qps 3 run-accuracy
```

**Compliance tests:**
```bash
# Run both TEST07 and TEST09 (default)
python3 scripts/run_submission.py --scenario Offline run-compliance

# Run only TEST07
python3 scripts/run_submission.py --scenario Offline run-compliance TEST07

# Run only TEST09
python3 scripts/run_submission.py --scenario Server --server-target-qps 3 run-compliance TEST09
```

### Additional Options

**Dry run (see commands without executing):**
```bash
# Dry run shows what commands would be executed without actually running them
# This is useful for verifying configuration before running actual tests
python3 scripts/run_submission.py --dry-run run-server
python3 scripts/run_submission.py --dry-run --scenario Offline run-offline

# Dry run also validates configuration in lenient mode
# It will show warnings about missing environment variables but won't fail
```

**Generate bash script:**
```bash
# Generate bash script for Server tests
python3 scripts/run_submission.py --print-bash run-server > run_tests.sh

# Generate bash script for Offline tests
python3 scripts/run_submission.py --print-bash --scenario Offline run-compliance > compliance_tests.sh
```

**Custom output directory:**
```bash
python3 scripts/run_submission.py --output-dir /path/to/output run-server
```

**Custom MLflow tags:**
```bash
python3 scripts/run_submission.py --tag submission=final,version=1.0 run-server
```

**Custom user configuration:**
```bash
python3 scripts/run_submission.py --user-conf /path/to/user.conf run-server
```

**Custom audit configuration:**
```bash
python3 scripts/run_submission.py --audit-config /path/to/audit.config run-compliance
```

### Command Line Arguments

| Argument | Description | Required |
|----------|-------------|----------|
| `--scenario` | Scenario: `Server` or `Offline` | Yes (for specific scenarios) |
| `--server-target-qps` | Target QPS for Server scenario | Yes (for Server scenario) |
| `--output-dir` | Output directory (default: `./harness_output`) | No |
| `--dataset-dir` | Dataset directory | No (can use env var) |
| `--perf-dataset` | Performance dataset path | No (can use env var) |
| `--acc-dataset` | Accuracy dataset path | No (can use env var) |
| `--compliance-dataset` | Compliance dataset path | No (can use env var) |
| `--api-server-url` | API server URL | No (can use env var) |
| `--mlflow-tracking-uri` | MLflow tracking URI | No (can use env var) |
| `--mlflow-experiment-name` | MLflow experiment name | No (can use env var) |
| `--tag`, `--mlflow-tag` | MLflow tags (format: `key1=value1,key2=value2`) | No |
| `--user-conf` | User config file | No |
| `--audit-config` | Audit config file for compliance tests | No |
| `--dry-run` | Print commands without executing | No |
| `--print-bash` | Generate bash script | No |

### Commands

| Command | Description |
|--------|-------------|
| `run-server` | Run all Server tests (performance, accuracy, compliance) |
| `run-offline` | Run all Offline tests (performance, accuracy, compliance) |
| `run-all` | Run all tests for both scenarios |
| `run-performance` | Run performance test only |
| `run-accuracy` | Run accuracy test only |
| `run-compliance` | Run compliance tests (TEST07 and TEST09 by default) |

## Creating Submission

After running all tests, create the MLPerf submission package using the `create_submission.sh` script.

### Steps

1. **Ensure all tests have completed successfully:**
   ```bash
   # Verify output directory exists and contains results
   ls -la harness_output/
   ```

2. **Run create_submission.sh:**
   ```bash
   cd harness
   bash create_submission.sh harness_output
   ```

   The script will:
   - Copy the output directory to `SUBMISSION_CHECK`
   - Run compliance checks
   - Check accuracy results
   - Convert to MLPerf submission structure
   - Truncate accuracy logs
   - Copy system JSON and config files
   - Run submission checker

3. **Verify submission:**
   ```bash
   # Check the submission structure
   ls -la SUBMISSION_TEST/_truncated_v6/closed/RedHat/
   ```

4. **Submission directory structure:**
   ```
   SUBMISSION_TEST/_truncated_v6/closed/RedHat/
   ├── results/
   │   └── 8xH200-LLM-D-Openshift/
   │       └── gpt-oss-120b/
   │           ├── Server/
   │           │   ├── performance/
   │           │   ├── accuracy/
   │           │   └── compliance/
   │           └── Offline/
   │               ├── performance/
   │               ├── accuracy/
   │               └── compliance/
   ├── systems/
   │   └── 8xH200-LLM-D-Openshift.json
   └── ...
   ```

### Troubleshooting

If the submission checker fails, check:
- All test results are present in the output directory
- Accuracy logs are valid
- Compliance test results are present
- System JSON file is correct
- Config files are in the right location

## Quick Reference

### Complete Workflow

```bash
# 1. Deploy LLM-D server (choose H200 or B200)
# For H200:
cd setup/llm-d/H200/
bash deploy_gptoss120b_v050.sh server

# For B200:
cd setup/llm-d/B200/
bash deploy_gptoss120b_v050.sh server

# 1b. Update EPP configuration (recommended)
bash update_epp_config.sh

# 2. Set up environment variables
cd ../../harness/scripts
source set_env_vars.sh
export DATASET_DIR=...
export API_SERVER_URL=...
# ... set other required variables

# 3. Navigate to harness directory
cd ../

# 4. Run tests
python3 scripts/run_submission.py --scenario Server --server-target-qps 3 run-server
python3 scripts/run_submission.py --scenario Offline run-offline

# 5. Create submission
bash create_submission.sh harness_output
```

## Additional Resources

- LLM-D documentation:
  - H200: See `setup/llm-d/H200/OVERRIDE_FILES_README.md`
  - B200: See `setup/llm-d/B200/OVERRIDE_FILES_README.md`
  - EPP Configuration: See `setup/llm-d/H200/EPP_CONFIG_COMPARISON.md` or `setup/llm-d/B200/EPP_CONFIG_COMPARISON.md`
- Environment variables script: `harness/scripts/set_env_vars.sh`
- Submission converter: `harness/scripts/convert_to_submission.py`
