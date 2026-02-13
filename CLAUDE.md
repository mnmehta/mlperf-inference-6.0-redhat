# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the **MLPerf Inference Benchmark Suite** - a benchmark suite for measuring how fast systems can run ML models in various deployment scenarios (mobile devices, edge, datacenter). The repository contains reference implementations, the LoadGen library, and tools for creating MLPerf inference submissions.

**Current Version**: MLPerf Inference v6.0 (submission deadline February 13, 2026)

## High-Level Architecture

### Core Components

1. **LoadGen (`loadgen/`)** - The reusable C++ library (with Python bindings) that:
   - Generates traffic for benchmark scenarios (SingleStream, MultiStream, Server, Offline)
   - Records all traffic and responses for verification
   - Is model-agnostic and data-format agnostic
   - Requires users to implement `SystemUnderTest` and `QuerySampleLibrary` interfaces
   - Must NOT be modified for submissions (all changes must be upstreamed)

2. **Benchmark Implementations** - Organized by task domain:
   - `language/` - NLP models (BERT, GPT-J, Llama2-70B, Llama3.1-405B/8B, Mixtral-8x7B, DeepSeek-R1, GPT-OSS-120B)
   - `vision/` - Computer vision (ResNet50, RetinaNet, 3D-UNet)
   - `recommendation/` - Recommendation models (DLRM-v2, DLRM-v3)
   - `graph/` - Graph models (R-GAT)
   - `automotive/` - Autonomous driving (PointPainting)
   - `speech2text/` - Speech recognition (Whisper)
   - `text_to_image/` - Stable Diffusion XL
   - `text_to_video/` - Wan2.2-T2V-A14B-Diffusers
   - `multimodal/` - Vision-Language Models (Qwen3-VL)

3. **Harness (`harness/`)** - Modular framework for LLM benchmarking:
   - **Backend Servers** (`backendserver/`) - Manages inference servers (vLLM, SGLang)
   - **Clients** (`Client/`) - LoadGen integration for Offline and Server scenarios
   - **Dataset Processing** (`data/`) - Generic dataset processor for JSON/Pickle/CSV
   - **Metrics** (`metrics/`) - Optional metrics collection and visualization
   - Architecture documented in `harness/ARCHITECTURE.md`

4. **Submission Infrastructure**:
   - `compliance/` - Compliance tests (TEST01, TEST04, TEST06, TEST07, TEST08, TEST09)
   - `tools/submission/` - Submission validation scripts
   - `setup/` - Deployment scripts (e.g., llm-d for Kubernetes)

### Key Concepts

- **Scenarios**: SingleStream (latency), MultiStream, Server (latency + throughput), Offline (throughput)
- **Divisions**: Closed (strict rules, specific models) vs. Open (relaxed constraints)
- **Categories**: Edge vs. Datacenter (different scenario requirements)
- **SUT (System Under Test)**: The inference system being benchmarked
- **QSL (Query Sample Library)**: Provides access to dataset samples

## Common Commands

### Building LoadGen

**Python Installation:**
```bash
cd loadgen
pip install absl-py numpy pybind11
python -m pip install .
# Or install pre-built wheels:
pip install mlcommons-loadgen
```

**C++ Build:**
```bash
cd loadgen
mkdir build && cd build
cmake .. && cmake --build .
cp libmlperf_loadgen.a ..
```

### Running Benchmarks

Each benchmark has its own setup process. General pattern:

**For reference implementations:**
```bash
cd <benchmark_directory>  # e.g., language/bert
make setup                # Download datasets and models
make build_docker         # Build Docker image
make launch_docker        # Launch container
# Inside container:
python3 run.py --backend=[tf|pytorch|onnxruntime] \
               --scenario=[Offline|SingleStream|MultiStream|Server] \
               [--accuracy]
```

**For harness-based benchmarks (LLMs):**
```bash
cd harness

# Option 1: Set environment variables manually
export API_SERVER_URL=http://localhost:8000
export DATASET_DIR=/path/to/datasets
export OUTPUT_DIR=./output
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export MLFLOW_TRACKING_URI=http://mlflow:5000
export MLFLOW_EXPERIMENT_NAME=your_experiment

# Option 2: Use the environment setup script (recommended)
source scripts/set_env_vars.sh
# Edit the script to set your values, or export before sourcing

# Run directly with harness file
python harness_llama3.1_8b.py --scenario Offline --test_mode performance

# Run using submission script (handles performance, accuracy, compliance)
python scripts/run_submission.py --scenario Server --server-target-qps 3 run-all
python scripts/run_submission.py --scenario Offline run-offline
python scripts/run_submission.py --scenario Server run-compliance TEST07

# Convert results to submission format
python scripts/convert_to_submission.py --results-dir ./harness_output
```

### Compliance Testing

Required for closed division submissions:

```bash
# 1. Copy audit.config to working directory
cp compliance/TEST01/audit.config .

# 2. Run benchmark normally (LoadGen will detect audit.config)
python3 run.py --scenario Offline

# 3. Verify and prepare submission files
cd compliance/TEST01
python run_verification.py --results_dir <path> --compliance_dir <output_path>

# 4. Remove audit.config to avoid unintended compliance runs
rm audit.config
```

### Submission Validation

```bash
# Run submission checker (validates directory structure, logs, accuracy)
python tools/submission/submission_checker.py \
    --input <submission_directory> \
    --version v6.0
```

### Running Tests

```bash
# Harness tests
cd harness/backendserver/tests
pytest test_inference_server.py
pytest test_server_integration.py

cd harness/metrics/test
pytest test_metrics_collector.py
pytest test_visualizer.py
```

### Environment Variables for Harness

The harness requires several environment variables. Use `harness/scripts/set_env_vars.sh` to set them:

```bash
# Source the environment setup script
cd harness
source scripts/set_env_vars.sh

# Or set manually
export DATASET_DIR=/path/to/datasets          # Required: Dataset directory
export API_SERVER_URL=http://localhost:8000   # Required: Inference server URL
export OUTPUT_DIR=./harness_output            # Optional: Output directory (default: ./harness_output)
export AWS_ACCESS_KEY_ID=your_key             # Required: AWS credentials for S3 datasets
export AWS_SECRET_ACCESS_KEY=your_secret
export MLFLOW_TRACKING_URI=http://mlflow:5000 # Required: MLflow tracking server
export MLFLOW_EXPERIMENT_NAME=experiment_name # Required: MLflow experiment name
export HF_HOME=/path/to/huggingface_cache     # Optional: HuggingFace cache location
```

### Submission Workflow (Harness)

The harness provides automated scripts for complete submission workflows:

```bash
cd harness

# 1. Set up environment
source scripts/set_env_vars.sh  # Configure required variables

# 2. Run performance + accuracy for a scenario
python scripts/run_submission.py --scenario Server --server-target-qps 3 run-server
python scripts/run_submission.py --scenario Offline run-offline

# 3. Run compliance tests (runs TEST07 and TEST09 by default)
python scripts/run_submission.py --scenario Offline run-compliance
python scripts/run_submission.py run-compliance TEST07  # Specific test only

# 4. Run all tests (performance, accuracy, compliance)
python scripts/run_submission.py --scenario Server --server-target-qps 3 run-all

# 5. Convert results to submission format
python scripts/convert_to_submission.py \
    --results-dir ./harness_output \
    --output-dir ./submission

# 6. Run compliance verification
bash scripts/run_compliance_checks.sh ./harness_output

# Options:
#   --dry-run: Print commands without executing
#   --print-bash: Generate bash script instead of running
#   --no-mlflow: Skip MLflow logging
```

**Available harness files by model:**
- `harness_llama2_70b.py` - Llama 2 70B
- `harness_llama3.1_8b.py` - Llama 3.1 8B
- `harness_gpt_oss_120b.py` - GPT-OSS-120B
- `harness_deepseek_r1.py` - DeepSeek-R1
- `harness_qwen3vl.py` - Qwen3-VL (multimodal)
- `harness_main.py` - Generic harness base

## Development Workflow

### Working with LoadGen

1. **DO NOT modify LoadGen** for submission purposes - all changes must be upstreamed
2. LoadGen reads configuration from (in order): `mlperf.conf` → `user.conf` → `audit.config`
3. Key files generated:
   - `mlperf_log_summary.txt` - Performance summary
   - `mlperf_log_detail.txt` - Detailed logs
   - `mlperf_log_trace.json` - Timeline (view in chrome://tracing)
   - `mlperf_log_accuracy.json` - Accuracy results (post-process with benchmark-specific scripts)

### Implementing a New Benchmark

1. Implement `SystemUnderTest` interface (handles queries from LoadGen)
2. Implement `QuerySampleLibrary` interface (provides dataset samples)
3. Call `QuerySampleComplete()` for every sample received
4. Process accuracy log with benchmark-specific script
5. Ensure proper `TestSettings` configuration for MLPerf compliance

### Choosing Between Harness and Direct SUT Implementation

**Use the Harness (`harness/`) when:**
- Working with LLM benchmarks (Llama, GPT-OSS, DeepSeek, Mixtral, etc.)
- Need modular server management (vLLM, SGLang backends)
- Want automated submission workflow scripts
- Using multiple dataset formats (JSON, Pickle, CSV)
- Need metrics collection and visualization

**Use direct SUT implementation when:**
- Working with non-LLM benchmarks (vision, recommendation, etc.)
- Need tight control over LoadGen integration
- Implementing highly specialized inference logic
- Following reference implementation patterns

**Harness Architecture Overview:**
```
Dataset (JSON/Pickle/CSV)
    ↓
DatasetProcessor (standardizes to input_ids format)
    ↓
LoadGen Client (OfflineClient/ServerClient)
    ↓
Backend Server (vLLM/SGLang) ← InferenceServer manages lifecycle
    ↓
Metrics Collection (optional)
```

See `harness/ARCHITECTURE.md` for detailed architecture documentation.

### Adding a Backend to Harness

```python
# In harness/backendserver/
class MyBackendServer(InferenceServer):
    def get_backend_name(self):
        return "mybackend"

    def get_launch_command(self):
        # Return command to launch server
        return ["my-server", "--port", str(self.port)]

    # Implement other abstract methods...
```

Register in `backendserver/__init__.py` and update factory function.

## Important Constraints

### Submission Requirements (Closed Division)

- **Accuracy**: Must be within 99% or 99.9% of reference (model-specific)
- **Model Weights**: No modifications except quantization
- **Scenarios**: All required scenarios for category (edge/datacenter)
- **Compliance Tests**: Must pass required tests per benchmark
- **Test Duration**: Minimum 10 minutes per scenario; Offline needs ≥24,756 queries
- **Datacenter Requirements**: ECC RAM + networking capabilities
- **Power Submissions**: Approved SPEC power analyzer + EULA signature

### Version-Specific Information

- Use master branch commits since [6.0 seed release](https://github.com/mlcommons/inference/commit/f131a0d29ccae9a967d93ffe96f66b1be3537d3b)
- Reference implementations are NOT optimized for performance - use vendor implementations for real benchmarking
- Framework choice in README is for reference only; submitters can use any framework

## File Organization

- Individual benchmarks have their own READMEs with detailed setup instructions
- Configuration files:
  - `mlperf.conf` (symlink to `loadgen/mlperf.conf`) - MLPerf defaults
  - `user.conf` - User overrides (per benchmark directory)
  - `audit.config` - Compliance test config (copy from `compliance/TEST*/`)
- Each benchmark directory typically contains:
  - Model download/setup scripts
  - SUT implementations (`*_SUT.py`)
  - QSL implementations (`*_QSL.py`)
  - Accuracy evaluation scripts (`accuracy-*.py`)
  - `run.py` or similar harness script

## Tools and Utilities

### Submission Tools (`tools/submission/`)

```bash
# Main submission checker - validates entire submission directory
python tools/submission/submission_checker.py \
    --input <submission_dir> \
    --version v6.0 \
    [--submitter <name>]

# Generate final submission report
python tools/submission/generate_final_report.py \
    --input <submission_dir> \
    --output report.html

# Preprocess submission (cleanup, formatting)
python tools/submission/preprocess_submission.py \
    --input <submission_dir> \
    --output <processed_dir>

# Restructure from v5.1 to v6.0 format (if needed)
python tools/submission/restructure_v5.1_to_v6.0.py \
    --input <v5.1_dir> \
    --output <v6.0_dir>

# Power submission checker
python tools/submission/power/power_checker.py \
    --input <power_submission_dir>
```

### Harness Utilities (`harness/scripts/`)

- **run_submission.py**: Automated test runner for performance/accuracy/compliance
- **convert_to_submission.py**: Converts harness output to MLPerf submission format
- **run_compliance_checks.sh**: Runs compliance verification on results
- **set_env_vars.sh**: Environment variable setup helper
- **upload_to_mlflow.py**: Upload results to MLflow tracking server
- **run_harness_with_vllm.sh**: Launch harness with vLLM backend

## Useful Resources

- [MLPerf Inference Rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)
- [Submission Guidelines](Submission_Guidelines.md) - Checklist for submissions
- [Power Measurement Rules](https://github.com/mlcommons/inference_policies/blob/master/power_measurement.adoc)
- [LoadGen FAQ](loadgen/README_FAQ.md)
- [MLCommons Documentation](https://docs.mlcommons.org/inference/benchmarks/)
- Past results: `https://github.com/mlcommons/inference_results_v{version}`

## Common Patterns

### Dataset Download

Many benchmarks support automated download via MLCFlow:
```bash
pip install mlc-scripts
mlcr get,dataset,squad,validation --outdirname=<path> -j
mlcr get,ml-model,bert-large,_pytorch --outdirname=<path> -j
```

### Accuracy Evaluation

```bash
# After accuracy run, process the log
python accuracy-squad.py \
    --mlperf-accuracy-file mlperf_log_accuracy.json \
    --squad-val-file <dataset_path>
```

### Docker Workflows

Reference implementations provide Docker support:
```bash
make build_docker   # Build container with dependencies
make launch_docker  # Interactive session
# Or for automated runs:
docker run <image> python3 run.py --scenario Offline --accuracy
```

## Debugging Tips

### Performance Analysis
- Use `chrome://tracing` to visualize `mlperf_log_trace.json` for timeline analysis
- Check `mlperf_log_summary.txt` for quick performance metrics
- Review `mlperf_log_detail.txt` for LoadGen events (query timing, audit.config detection)

### Server and Backend Issues
- Enable debug mode in harness backend servers for process cleanup verification:
  ```python
  server = VLLMServer(..., debug_mode=True)
  ```
- For tensor parallel/distributed setups, verify all processes are properly cleaned up
- Server retry issues: Recent fixes (commits a7f72c4, 5821561) improved retry handling - each thread manages retries independently
- Backend server heartbeat monitoring can be configured via YAML or constructor args

### Compliance and Submission
- Submission checker warnings should be investigated (some require submitter response)
- Compliance tests automatically detect `audit.config` in working directory - remove after tests
- For compliance TEST07/TEST09, ensure correct scenario mapping (recent fix: commit 2f836ea)
- Check `mlperf_log_detail.txt` for audit config detection messages

### Common Issues
- **Missing environment variables**: Run `validate_env_vars` from `set_env_vars.sh`
- **Dataset loading errors**: Verify DATASET_DIR points to correct location and AWS credentials are set
- **API server connection**: Check API_SERVER_URL and ensure server is running (port accessible)
- **Process cleanup**: Use debug mode to verify all worker processes terminate properly
- **Accuracy failures**: Ensure correct dataset version and post-processing script for the model
