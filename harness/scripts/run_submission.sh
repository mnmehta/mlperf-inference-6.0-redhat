#!/bin/bash
# ============================================================================
# run_submission.sh
# -----------------
# Simple bash script to run MLPerf harness tests
# ============================================================================
#
# USAGE:
# ------
# This script runs MLPerf inference harness tests for GPT-OSS models.
#
# Prerequisites:
#   - Required environment variables must be set (see below)
#   - Python 3 with harness_main.py available
#   - Datasets must be available at specified paths
#
# Required Environment Variables:
#   - DATASET_DIR: Base directory containing datasets
#   - API_SERVER_URL: URL of the API server (e.g., http://localhost:9999)
#   - AWS_ACCESS_KEY_ID: AWS access key for dataset/model access
#   - AWS_SECRET_ACCESS_KEY: AWS secret key for dataset/model access
#   - MLFLOW_TRACKING_URI: MLflow tracking server URI (e.g., http://host:port)
#   - MLFLOW_EXPERIMENT_NAME: Name of the MLflow experiment
#
# Optional Environment Variables:
#   - OUTPUT_DIR: Output directory (default: ./harness_output)
#   - SCENARIO: Test scenario - Server or Offline (default: Server)
#   - SERVER_TARGET_QPS: Target QPS for Server scenario (default: 3)
#   - HF_HOME: HuggingFace home directory
#
# Basic Usage Examples:
#   # Set required environment variables
#   export DATASET_DIR=/mnt/data/datasets/gpt-oss_data
#   export API_SERVER_URL=http://localhost:9999
#   export AWS_ACCESS_KEY_ID=your_key
#   export AWS_SECRET_ACCESS_KEY=your_secret
#   export MLFLOW_TRACKING_URI=http://mlflow-server:5000
#   export MLFLOW_EXPERIMENT_NAME=GPT-OSS-120B-Experiments
#
#   # Run all Server tests
#   ./run_submission.sh run-server
#
#   # Run only performance test
#   ./run_submission.sh run-performance
#
#   # Run with command-line options
#   ./run_submission.sh --scenario Server --server-target-qps 5 run-server
#
#   # Dry run (see commands without executing)
#   ./run_submission.sh --dry-run run-server
#
#   # Get help
#   ./run_submission.sh --help
#
# ============================================================================

# Check for --dry-run flag early to adjust error handling
DRY_RUN_EARLY="false"
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
        DRY_RUN_EARLY="true"
        break
    fi
done

# Set DRY_RUN from early detection
DRY_RUN="$DRY_RUN_EARLY"

# Log if dry-run is enabled
if [[ "$DRY_RUN" == "true" ]]; then
    echo "=========================================="
    echo "[DRY RUN MODE ENABLED]"
    echo "Commands will be displayed but NOT executed"
    echo "=========================================="
    echo ""
fi

# Only exit on error if not in dry-run mode
if [[ "$DRY_RUN" != "true" ]]; then
    set -e  # Exit on error
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ============================================================================
# Configuration Variables (can be overridden via command line or environment)
# ============================================================================

# Dataset configuration (required - no defaults)
DATASET_DIR="${DATASET_DIR:-}"
PERF_DATASET="${PERF_DATASET:-}"
ACC_DATASET="${ACC_DATASET:-}"
COMPLIANCE_DATASET="${COMPLIANCE_DATASET:-}"

# Output and API configuration
OUTPUT_DIR="${OUTPUT_DIR:-./harness_output}"
API_SERVER_URL="${API_SERVER_URL:-}"  # Required - no default

# AWS configuration
AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-}"
AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-}"

# MLflow configuration (required - no defaults)
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-}"
MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-}"  # Required - no default

# HuggingFace configuration
HF_HOME="${HF_HOME:-}"

# Model configuration
MODEL_CATEGORY="${MODEL_CATEGORY:-gpt-oss-120b}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
BACKEND="${BACKEND:-vllm}"
LG_MODEL_NAME="${LG_MODEL_NAME:-gpt-oss-120b}"

# Server configuration
SCENARIO="${SCENARIO:-Server}"
# Track if SERVER_TARGET_QPS was explicitly set (via env var or command line)
# Check if environment variable is set (even if empty)
if [[ -n "${SERVER_TARGET_QPS+x}" ]] && [[ -n "${SERVER_TARGET_QPS}" ]]; then
    SERVER_TARGET_QPS_SET="true"
    SERVER_TARGET_QPS="${SERVER_TARGET_QPS}"
else
    SERVER_TARGET_QPS_SET="false"
    SERVER_TARGET_QPS="${SERVER_TARGET_QPS:-3}"
fi

# Compliance test configuration
COMPLIANCE_TEST="${COMPLIANCE_TEST:-TEST07}"
AUDIT_CONFIG_SRC="${AUDIT_CONFIG_SRC:-}"
AUDIT_OVERRIDE_CONF="${AUDIT_OVERRIDE_CONF:-audit-override.cfg}"  # User conf for compliance tests

# Execution options
# DRY_RUN is set early above from command-line check, will be updated in parse_args if needed
USER_CONF="${USER_CONF:-}"
MLFLOW_USER_TAG="${MLFLOW_USER_TAG:-}"  # Optional user-specified MLflow tag (e.g., "group:mygroup")

# ============================================================================
# Helper Functions
# ============================================================================

# Function to validate necessary configuration
validate_config() {
    local errors=0
    local warnings=0
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Validating configuration (lenient mode)..."
    else
        echo "Validating configuration..."
    fi
    
    # Check required environment variables
    if [[ -z "${DATASET_DIR}" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "[DRY RUN] NOTE: DATASET_DIR would be required: <not set>"
        else
            echo "ERROR: DATASET_DIR environment variable is not set"
            echo "       Please set it via: export DATASET_DIR=/path/to/datasets"
            echo "       Or use: --dataset-dir /path/to/datasets"
            errors=$((errors + 1))
        fi
    fi
    
    if [[ -z "${API_SERVER_URL}" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "[DRY RUN] NOTE: API_SERVER_URL would be required: <not set>"
        else
            echo "ERROR: API_SERVER_URL environment variable is not set"
            echo "       Please set it via: export API_SERVER_URL=http://localhost:9999"
            echo "       Or use: --api-server-url http://localhost:9999"
            errors=$((errors + 1))
        fi
    fi
    
    if [[ -z "${AWS_ACCESS_KEY_ID}" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "[DRY RUN] NOTE: AWS_ACCESS_KEY_ID would be required: <not set>"
        else
            echo "ERROR: AWS_ACCESS_KEY_ID environment variable is not set"
            echo "       Please set it via: export AWS_ACCESS_KEY_ID=your_key"
            echo "       Or use: --aws-access-key-id your_key"
            errors=$((errors + 1))
        fi
    fi
    
    if [[ -z "${AWS_SECRET_ACCESS_KEY}" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "[DRY RUN] NOTE: AWS_SECRET_ACCESS_KEY would be required: <not set>"
        else
            echo "ERROR: AWS_SECRET_ACCESS_KEY environment variable is not set"
            echo "       Please set it via: export AWS_SECRET_ACCESS_KEY=your_secret"
            echo "       Or use: --aws-secret-access-key your_secret"
            errors=$((errors + 1))
        fi
    fi
    
    if [[ -z "${MLFLOW_TRACKING_URI}" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "[DRY RUN] NOTE: MLFLOW_TRACKING_URI would be required: <not set>"
        else
            echo "ERROR: MLFLOW_TRACKING_URI environment variable is not set"
            echo "       Please set it via: export MLFLOW_TRACKING_URI=http://host:port"
            echo "       Or use: --mlflow-tracking-uri http://host:port"
            errors=$((errors + 1))
        fi
    fi
    
    if [[ -z "${MLFLOW_EXPERIMENT_NAME}" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "[DRY RUN] NOTE: MLFLOW_EXPERIMENT_NAME would be required: <not set>"
        else
            echo "ERROR: MLFLOW_EXPERIMENT_NAME environment variable is not set"
            echo "       Please set it via: export MLFLOW_EXPERIMENT_NAME=ExperimentName"
            echo "       Or use: --mlflow-experiment-name ExperimentName"
            errors=$((errors + 1))
        fi
    fi
    
    # Check if datasets exist
    if [[ ! -f "${PERF_DATASET}" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "[DRY RUN] NOTE: Performance dataset would be checked: ${PERF_DATASET}"
        else
            echo "WARNING: Performance dataset not found: ${PERF_DATASET}"
            warnings=$((warnings + 1))
        fi
    fi
    
    if [[ ! -f "${ACC_DATASET}" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "[DRY RUN] NOTE: Accuracy dataset would be checked: ${ACC_DATASET}"
        else
            echo "WARNING: Accuracy dataset not found: ${ACC_DATASET}"
            warnings=$((warnings + 1))
        fi
    fi
    
    if [[ ! -f "${COMPLIANCE_DATASET}" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "[DRY RUN] NOTE: Compliance dataset would be checked: ${COMPLIANCE_DATASET}"
        else
            echo "WARNING: Compliance dataset not found: ${COMPLIANCE_DATASET}"
            warnings=$((warnings + 1))
        fi
    fi
    
    # Check if output directory is writable (skip in dry-run)
    if [[ "$DRY_RUN" != "true" ]]; then
        if [[ ! -w "$(dirname "${OUTPUT_DIR}")" ]] && [[ ! -d "${OUTPUT_DIR}" ]]; then
            echo "WARNING: Output directory may not be writable: ${OUTPUT_DIR}"
            warnings=$((warnings + 1))
        fi
    fi
    
    # Check if harness_main.py exists
    if [[ ! -f "${HARNESS_DIR}/harness_main.py" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "[DRY RUN] NOTE: harness_main.py would be checked: ${HARNESS_DIR}/harness_main.py"
        else
            echo "ERROR: harness_main.py not found at ${HARNESS_DIR}/harness_main.py"
            errors=$((errors + 1))
        fi
    fi
    
    # Check if user-conf file exists (if specified)
    if [[ -n "${USER_CONF}" ]] && [[ ! -f "${USER_CONF}" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "[DRY RUN] NOTE: User config file would be checked: ${USER_CONF}"
        else
            echo "WARNING: User config file not found: ${USER_CONF}"
            warnings=$((warnings + 1))
        fi
    fi
    
    # Print summary (skip error exit in dry-run mode)
    if [[ $errors -gt 0 ]]; then
        echo ""
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "[DRY RUN] NOTE: Would find $errors error(s) in actual run"
        else
            echo "ERROR: Found $errors error(s). Please fix before running."
            return 1
        fi
    fi
    
    if [[ $warnings -gt 0 ]] && [[ "$DRY_RUN" != "true" ]]; then
        echo ""
        echo "WARNING: Found $warnings warning(s). Continuing anyway..."
        echo ""
    elif [[ "$DRY_RUN" != "true" ]]; then
        echo "✓ Configuration validation passed"
        echo ""
    else
        echo "[DRY RUN] ✓ Configuration check completed"
        echo ""
    fi
    
    return 0
}

# Function to find audit.config for compliance tests
find_audit_config() {
    local test_name="$1"
    if [[ -n "$AUDIT_CONFIG_SRC" ]]; then
        echo "$AUDIT_CONFIG_SRC"
        return
    fi
    
    # Try to find in compliance directory
    local compliance_dir="${HARNESS_DIR}/../compliance"
    local test_dir="${compliance_dir}/${test_name}/${MODEL_CATEGORY}"
    
    if [[ -f "${test_dir}/audit.config" ]]; then
        echo "${test_dir}/audit.config"
        return
    fi
    
    # Try generic path
    test_dir="${compliance_dir}/${test_name}"
    if [[ -f "${test_dir}/audit.config" ]]; then
        echo "${test_dir}/audit.config"
        return
    fi
    
    echo ""
}

# ============================================================================
# Base Command Function
# ============================================================================

# Base function to build and run harness command
run_harness_base() {
    local scenario="$1"
    local test_mode="$2"
    local dataset_path="$3"
    local output_subdir="$4"
    local description="$5"
    local tags="$6"
    local audit_config_path="$7"
    
    # Build output directory
    local full_output_dir="${OUTPUT_DIR}/${scenario,,}/${test_mode}"
    if [[ -n "$output_subdir" ]]; then
        full_output_dir="${OUTPUT_DIR}/${scenario,,}/${test_mode}/${output_subdir}"
    fi
    
    # Build base command as a string (for proper dry-run handling)
    local cmd="python3 \"${HARNESS_DIR}/harness_main.py\""
    cmd="${cmd} --model-category \"${MODEL_CATEGORY}\""
    cmd="${cmd} --model \"${MODEL}\""
    cmd="${cmd} --dataset-path \"${dataset_path}\""
    cmd="${cmd} --backend \"${BACKEND}\""
    cmd="${cmd} --lg-model-name \"${LG_MODEL_NAME}\""
    cmd="${cmd} --test-mode \"${test_mode}\""
    cmd="${cmd} --api-server-url \"${API_SERVER_URL}\""
    cmd="${cmd} --scenario \"${scenario}\""
    cmd="${cmd} --output-dir \"${full_output_dir}\""
    cmd="${cmd} --mlflow-experiment-name \"${MLFLOW_EXPERIMENT_NAME}\""
    
    # Add MLflow tracking URI if provided
    if [[ -n "$MLFLOW_TRACKING_URI" ]]; then
        # Extract host and port from URI
        local mlflow_host_port="${MLFLOW_TRACKING_URI#http://}"
        mlflow_host_port="${mlflow_host_port#https://}"
        local mlflow_host="${mlflow_host_port%%:*}"
        local mlflow_port="${mlflow_host_port##*:}"
        cmd="${cmd} --mlflow-host \"${mlflow_host}\""
        if [[ "$mlflow_port" != "$mlflow_host" ]]; then
            cmd="${cmd} --mlflow-port \"${mlflow_port}\""
        fi
    fi
    
    # Add server-target-qps for Server scenario
    if [[ "$scenario" == "Server" ]]; then
        cmd="${cmd} --server-target-qps \"${SERVER_TARGET_QPS}\""
    fi
    
    # Add MLflow description if provided
    if [[ -n "$description" ]]; then
        cmd="${cmd} --mlflow-description \"${description}\""
    fi
    
    # Add MLflow tags if provided
    # Merge user-specified tags with existing tags
    local final_tags="$tags"
    if [[ -n "$MLFLOW_USER_TAG" ]]; then
        if [[ -n "$final_tags" ]]; then
            final_tags="${final_tags},${MLFLOW_USER_TAG}"
        else
            final_tags="$MLFLOW_USER_TAG"
        fi
    fi
    
    if [[ -n "$final_tags" ]]; then
        cmd="${cmd} --mlflow-tag \"${final_tags}\""
    fi
    
    # Add user-conf for performance and accuracy tests (if specified)
    # Note: Compliance tests use test-mode="performance" but we detect them by presence of audit_config_path
    if [[ -n "$USER_CONF" ]] && [[ "$test_mode" == "performance" || "$test_mode" == "accuracy" ]]; then
        # Only add USER_CONF if this is not a compliance test (compliance tests use AUDIT_OVERRIDE_CONF)
        if [[ -z "$audit_config_path" ]]; then
            if [[ -f "$USER_CONF" ]]; then
                cmd="${cmd} --user-conf \"${USER_CONF}\""
            else
                echo "WARNING: User config file not found: ${USER_CONF}, skipping --user-conf"
            fi
        fi
    fi
    
    # Add audit-override.cfg for compliance tests (detected by presence of audit_config_path)
    if [[ -n "$audit_config_path" && -f "$audit_config_path" ]]; then
        if [[ -f "$AUDIT_OVERRIDE_CONF" ]] || [[ "$DRY_RUN" == "true" ]]; then
            # In dry-run, add it even if file doesn't exist (for display purposes)
            cmd="${cmd} --user-conf \"${AUDIT_OVERRIDE_CONF}\""
        else
            echo "WARNING: Audit override config file not found: ${AUDIT_OVERRIDE_CONF}, skipping --user-conf"
        fi
    fi
    
    # Add audit config if provided (for compliance tests)
    local audit_dest=""
    if [[ -n "$audit_config_path" && -f "$audit_config_path" ]]; then
        # Copy audit.config to harness directory (not output directory)
        audit_dest="${HARNESS_DIR}/audit.config"
        # Use just "audit.config" in the command (not full path)
        cmd="${cmd} --audit-config \"audit.config\""
        
        if [[ "$DRY_RUN" != "true" ]]; then
            # Copy audit.config to harness directory (only if not dry-run)
            cp "${audit_config_path}" "${audit_dest}"
            
            # Cleanup function to remove audit.config from harness directory after run
            cleanup_audit() {
                if [[ -f "${audit_dest}" ]]; then
                    rm -f "${audit_dest}"
                    echo "Cleaned up audit.config from harness directory"
                fi
            }
            trap cleanup_audit EXIT
        fi
    fi
    
    # Print command
    echo "=========================================="
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Would run: ${scenario} scenario, ${test_mode} mode"
    else
        echo "Running: ${scenario} scenario, ${test_mode} mode"
    fi
    echo "Dataset: ${dataset_path}"
    echo "Output: ${full_output_dir}"
    
    # Show audit.config operations in dry-run mode
    if [[ "$DRY_RUN" == "true" ]] && [[ -n "$audit_config_path" && -f "$audit_config_path" ]]; then
        echo "[DRY RUN] Would copy audit.config:"
        echo "  Source: ${audit_config_path}"
        echo "  Destination: ${audit_dest}"
        echo "[DRY RUN] Would remove audit.config from harness directory after run"
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Command would be:"
    else
        echo "Command:"
    fi
    # Print command in a readable multi-line format
    # Split on " --" to put each argument on a new line with proper indentation
    echo "${cmd}" | sed 's/ --/ \\\n    --/g' | sed 's/^/  /'
    echo "=========================================="
    
    # Skip execution if dry run - CRITICAL: Never execute commands in dry-run mode
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Command not executed"
        return 0
    fi
    
    # Final safety check before eval - abort if DRY_RUN is somehow true
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] ERROR: DRY_RUN is true, aborting command execution to prevent accidental execution"
        return 0
    fi
    
    # Run command using eval (only when not in dry-run mode)
    if eval "${cmd}"; then
        echo "✓ ${scenario} ${test_mode} test completed successfully"
        return 0
    else
        echo "✗ ${scenario} ${test_mode} test failed"
        return 1
    fi
}

# ============================================================================
# Test Run Functions
# ============================================================================

# Run performance test
run_performance() {
    local scenario="$1"
    local dataset_path="${PERF_DATASET}"
    local description="${scenario} Performance"
    local tags="test_type:performance,scenario:${scenario}"
    
    if [[ "$scenario" == "Server" ]]; then
        description="${scenario} Performance QPS${SERVER_TARGET_QPS}"
        tags="${tags},qps:${SERVER_TARGET_QPS}"
    fi
    
    run_harness_base "${scenario}" "performance" "${dataset_path}" "" "${description}" "${tags}" ""
}

# Run accuracy test
run_accuracy() {
    local scenario="$1"
    local dataset_path="${ACC_DATASET}"
    local description="${scenario} Accuracy"
    local tags="test_type:accuracy,scenario:${scenario}"
    
    if [[ "$scenario" == "Server" ]]; then
        description="${scenario} Accuracy QPS${SERVER_TARGET_QPS}"
        tags="${tags},qps:${SERVER_TARGET_QPS}"
    fi
    
    run_harness_base "${scenario}" "accuracy" "${dataset_path}" "" "${description}" "${tags}" ""
}

# Run compliance test
run_compliance() {
    local scenario="$1"
    local compliance_test="${2:-${COMPLIANCE_TEST}}"
    local dataset_path=""
    local audit_config_path=$(find_audit_config "${compliance_test}")
    
    if [[ -z "$audit_config_path" ]]; then
        echo "Error: Could not find audit.config for ${compliance_test}"
        return 1
    fi
    
    # Select dataset based on compliance test type
    # TEST07 uses accuracy dataset, TEST09 uses performance dataset
    if [[ "${compliance_test}" == "TEST07" ]]; then
        dataset_path="${COMPLIANCE_DATASET}"
        if [[ -z "$dataset_path" ]]; then
            echo "Error: COMPLIANCE_DATASET not set (required for TEST07)"
            return 1
        fi
    elif [[ "${compliance_test}" == "TEST09" ]]; then
        dataset_path="${PERF_DATASET}"
        if [[ -z "$dataset_path" ]]; then
            echo "Error: PERF_DATASET not set (required for TEST09)"
            return 1
        fi
    else
        # Default to compliance dataset for other tests
        dataset_path="${COMPLIANCE_DATASET}"
        echo "WARNING: Unknown compliance test ${compliance_test}, using COMPLIANCE_DATASET"
    fi
    
    local description="${scenario} Compliance ${compliance_test}"
    local tags="test_type:compliance,scenario:${scenario},compliance_test:${compliance_test}"
    
    if [[ "$scenario" == "Server" ]]; then
        description="${scenario} Compliance ${compliance_test} QPS${SERVER_TARGET_QPS}"
        tags="${tags},qps:${SERVER_TARGET_QPS}"
    fi
    
    # Compliance tests use test-mode="performance" with audit.config
    run_harness_base "${scenario}" "performance" "${dataset_path}" "${compliance_test,,}" "${description}" "${tags}" "${audit_config_path}"
}

# Run all tests for a scenario (performance, accuracy, compliance)
run_all_tests() {
    local scenario="$1"
    
    echo "=========================================="
    echo "Running all tests for ${scenario} scenario"
    echo "=========================================="
    echo ""
    
    # Run performance
    echo ">>> Running performance test..."
    run_performance "${scenario}" || return 1
    echo ""
    
    # Run accuracy
    echo ">>> Running accuracy test..."
    run_accuracy "${scenario}" || return 1
    echo ""
    
    # Run compliance tests (both TEST07 and TEST09)
    echo ">>> Running compliance test TEST07..."
    run_compliance "${scenario}" "TEST07" || return 1
    echo ""
    
    echo ">>> Running compliance test TEST09..."
    run_compliance "${scenario}" "TEST09" || return 1
    echo ""
    
    echo "✓ All ${scenario} tests completed successfully"
}

# ============================================================================
# Command Line Argument Parsing
# ============================================================================

parse_args() {
    local command=""
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dataset-dir)
                DATASET_DIR="$2"
                # Re-derive dataset paths
                PERF_DATASET="${DATASET_DIR}/perf/perf_eval_ref.parquet"
                ACC_DATASET="${DATASET_DIR}/acc/acc_eval_ref.parquet"
                COMPLIANCE_DATASET="${DATASET_DIR}/acc/acc_eval_compliance_gpqa.parquet"
                shift 2
                ;;
            --perf-dataset)
                PERF_DATASET="$2"
                shift 2
                ;;
            --acc-dataset)
                ACC_DATASET="$2"
                shift 2
                ;;
            --compliance-dataset)
                COMPLIANCE_DATASET="$2"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --api-server-url)
                API_SERVER_URL="$2"
                shift 2
                ;;
            --aws-access-key-id)
                AWS_ACCESS_KEY_ID="$2"
                shift 2
                ;;
            --aws-secret-access-key)
                AWS_SECRET_ACCESS_KEY="$2"
                shift 2
                ;;
            --mlflow-tracking-uri)
                MLFLOW_TRACKING_URI="$2"
                shift 2
                ;;
            --mlflow-experiment-name)
                MLFLOW_EXPERIMENT_NAME="$2"
                shift 2
                ;;
            --tag|--mlflow-tag)
                MLFLOW_USER_TAG="$2"
                shift 2
                ;;
            --hf-home)
                HF_HOME="$2"
                shift 2
                ;;
            --scenario)
                SCENARIO="$2"
                # Ensure scenario is one of the valid values
                if [[ "$SCENARIO" != "Server" ]] && [[ "$SCENARIO" != "Offline" ]]; then
                    echo "ERROR: Invalid scenario: $SCENARIO. Must be 'Server' or 'Offline'"
                    exit 1
                fi
                shift 2
                ;;
            --server-target-qps)
                if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                    echo "ERROR: --server-target-qps requires a value"
                    echo "       Usage: --server-target-qps <value>"
                    exit 1
                fi
                SERVER_TARGET_QPS="$2"
                SERVER_TARGET_QPS_SET="true"
                shift 2
                ;;
            --compliance-test)
                COMPLIANCE_TEST="$2"
                shift 2
                ;;
            --audit-config)
                AUDIT_CONFIG_SRC="$2"
                shift 2
                ;;
            --audit-override-conf)
                AUDIT_OVERRIDE_CONF="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            --user-conf)
                USER_CONF="$2"
                shift 2
                ;;
            --help)
                cat << EOF
Usage: $0 [OPTIONS] [COMMAND]

Configuration Options:
  --dataset-dir DIR              Dataset directory (default: /mnt/data/datasets/gpt-oss_data)
  --perf-dataset PATH             Performance dataset path
  --acc-dataset PATH              Accuracy dataset path
  --compliance-dataset PATH       Compliance dataset path
  --output-dir DIR                Output directory (default: ./harness_output)
  --api-server-url URL            API server URL (default: http://localhost:9999)
  --aws-access-key-id KEY         AWS access key ID
  --aws-secret-access-key KEY     AWS secret access key
  --mlflow-tracking-uri URI       MLflow tracking URI (e.g., http://host:port)
  --mlflow-experiment-name NAME   MLflow experiment name
  --tag TAG                       Additional MLflow tag to merge with existing tags (e.g., 'group:mygroup')
                                  (alias: --mlflow-tag)
  --hf-home PATH                  HuggingFace home directory
  --scenario SCENARIO             Scenario: Server or Offline (default: Server)
  --server-target-qps QPS         Target QPS for Server scenario (default: 3)
  --compliance-test TEST          Compliance test name (default: TEST07)
  --audit-config PATH             Path to audit.config file (auto-detected if not provided)
  --audit-override-conf PATH      Path to audit-override.cfg file for compliance tests (default: audit-override.cfg)
  --dry-run                        Print commands without executing them
  --user-conf PATH                 User config file for performance/accuracy tests (useful for testing)

Commands:
  run-server                      Run all tests for Server scenario (includes TEST07 and TEST09)
  run-offline                     Run all tests for Offline scenario (includes TEST07 and TEST09)
  run-all                         Run all tests for both Server and Offline scenarios
  run-performance [SCENARIO]      Run performance test (default: Server, use --scenario to override)
  run-accuracy [SCENARIO]         Run accuracy test (default: Server, use --scenario to override)
  run-compliance [SCENARIO] [TEST] Run compliance test (default: Server, can specify: Offline TEST07, Server TEST09, etc.)

Examples:
  # Run all Server tests
  $0 --scenario Server --server-target-qps 3 run-server

  # Run all Offline tests
  $0 --scenario Offline run-offline

  # Run all tests for both Server and Offline scenarios
  $0 --server-target-qps 3 run-all

  # Run only performance test for Server
  $0 --scenario Server --server-target-qps 3 run-performance

  # Run compliance test with custom dataset
  $0 --compliance-dataset /path/to/compliance.parquet run-compliance

  # Run specific compliance test (TEST07 or TEST09) - uses default Server scenario
  $0 run-compliance TEST07
  $0 run-compliance TEST09

  # Run compliance test for specific scenario
  $0 run-compliance Offline TEST07
  $0 run-compliance Server TEST09
  $0 run-compliance Offline TEST09

  # Run with additional MLflow tag
  $0 --tag "group:mygroup" run-server

  # Set environment variables
  export DATASET_DIR=/mnt/data/datasets/gpt-oss_data
  export API_SERVER_URL=http://localhost:9999
  export SERVER_TARGET_QPS=3
  $0 run-server

  # Dry run to see commands without executing
  $0 --dry-run run-server

  # Run with user config for testing
  $0 --user-conf /path/to/user.conf run-performance
EOF
                exit 0
                ;;
            run-server|run-offline|run-performance|run-accuracy|run-compliance|run-all)
                command="$1"
                shift
                # For run-compliance, check if next arguments are scenario and/or test name
                if [[ "$command" == "run-compliance" ]] && [[ $# -gt 0 ]]; then
                    # Check if first argument is a scenario (Server or Offline)
                    # Only override SCENARIO if explicitly provided in command
                    # Otherwise, preserve SCENARIO that was set via --scenario option
                    if [[ "$1" == "Server" ]] || [[ "$1" == "Offline" ]]; then
                        SCENARIO="$1"
                        shift
                    fi
                    # Check if next argument is a test name (TEST07 or TEST09)
                    # If first arg was a test name (not a scenario), use it as test
                    if [[ $# -gt 0 ]] && [[ "$1" =~ ^TEST[0-9]+$ ]]; then
                        COMPLIANCE_TEST="$1"
                        shift
                    fi
                    # SCENARIO should already be set via --scenario option or default
                    # Don't reset it here to preserve the value from --scenario
                fi
                break
                ;;
            *)
                echo "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Return command via global variable or echo
    echo "$command"
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    # Parse command line arguments and get command
    local command=$(parse_args "$@")
    
    # Ensure SCENARIO is set (should be set via --scenario or default to Server)
    # Don't reset it if it was already set by --scenario option
    SCENARIO="${SCENARIO:-Server}"
    
    # Ensure DRY_RUN is set correctly (from early detection, parse_args, or environment)
    # Priority: command line > environment variable > default
    if [[ "$DRY_RUN_EARLY" == "true" ]] || [[ "$DRY_RUN" == "true" ]]; then
        DRY_RUN="true"
    else
        # Check environment variable if not set via command line
        DRY_RUN="${DRY_RUN:-false}"
    fi
    
    # Log if dry-run is enabled (if it wasn't already logged early)
    if [[ "$DRY_RUN" == "true" ]] && [[ "$DRY_RUN_EARLY" != "true" ]]; then
        echo "=========================================="
        echo "[DRY RUN MODE ENABLED]"
        echo "Commands will be displayed but NOT executed"
        echo "=========================================="
        echo ""
    fi
    
    # Derive dataset paths from DATASET_DIR if not explicitly set
    if [[ -n "$DATASET_DIR" ]]; then
        [[ -z "$PERF_DATASET" ]] && PERF_DATASET="${DATASET_DIR}/perf/perf_eval_ref.parquet"
        [[ -z "$ACC_DATASET" ]] && ACC_DATASET="${DATASET_DIR}/acc/acc_eval_ref.parquet"
        [[ -z "$COMPLIANCE_DATASET" ]] && COMPLIANCE_DATASET="${DATASET_DIR}/acc/acc_eval_compliance_gpqa.parquet"
    fi
    
    # Export environment variables if set
    [[ -n "$AWS_ACCESS_KEY_ID" ]] && export AWS_ACCESS_KEY_ID
    [[ -n "$AWS_SECRET_ACCESS_KEY" ]] && export AWS_SECRET_ACCESS_KEY
    [[ -n "$HF_HOME" ]] && export HF_HOME
    
    # Check if server-target-qps is required but not specified
    # Only check for Server scenario - Offline scenario doesn't need server-target-qps
    # Verify SCENARIO is set correctly before checking
    if [[ -z "$SCENARIO" ]]; then
        SCENARIO="Server"
    fi
    
    # Only require server-target-qps for Server scenario
    # Explicitly check that SCENARIO is "Server" (not "Offline" or anything else)
    # Check both that it was set AND that it has a value
    if [[ "$SCENARIO" = "Server" ]]; then
        if [[ "$SERVER_TARGET_QPS_SET" != "true" ]] || [[ -z "$SERVER_TARGET_QPS" ]]; then
            echo "ERROR: --server-target-qps is required for Server scenario"
            echo "       Please specify it via: --server-target-qps <value>"
            echo "       Or set it via environment variable: export SERVER_TARGET_QPS=<value>"
            exit 1
        fi
    fi
    # For Offline scenario, server-target-qps is not needed, so ignore it if specified
    
    # Print configuration
    echo "=========================================="
    echo "Configuration:"
    echo "  DATASET_DIR: ${DATASET_DIR}"
    echo "  PERF_DATASET: ${PERF_DATASET}"
    echo "  ACC_DATASET: ${ACC_DATASET}"
    echo "  COMPLIANCE_DATASET: ${COMPLIANCE_DATASET}"
    echo "  OUTPUT_DIR: ${OUTPUT_DIR}"
    echo "  API_SERVER_URL: ${API_SERVER_URL}"
    echo "  SCENARIO: ${SCENARIO}"
    if [[ "$SCENARIO" == "Server" ]]; then
        echo "  SERVER_TARGET_QPS: ${SERVER_TARGET_QPS}"
    fi
    echo "  MLFLOW_EXPERIMENT_NAME: ${MLFLOW_EXPERIMENT_NAME}"
    [[ -n "$MLFLOW_TRACKING_URI" ]] && echo "  MLFLOW_TRACKING_URI: ${MLFLOW_TRACKING_URI}"
    [[ -n "$AWS_ACCESS_KEY_ID" ]] && echo "  AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID:0:4}..."
    [[ -n "$HF_HOME" ]] && echo "  HF_HOME: ${HF_HOME}"
    [[ "$DRY_RUN" == "true" ]] && echo "  DRY_RUN: true"
    [[ -n "$USER_CONF" ]] && echo "  USER_CONF: ${USER_CONF}"
    echo "=========================================="
    echo ""
    
    # Validate configuration (skip validation in dry-run mode for datasets, but still check critical vars)
    if ! validate_config; then
        echo "ERROR: Configuration validation failed. Exiting."
        exit 1
    fi
    
    # Handle commands (default to run-server if no command specified)
    command="${command:-run-server}"
    
    case "$command" in
        run-server)
            run_all_tests "Server"
            ;;
        run-offline)
            run_all_tests "Offline"
            ;;
        run-performance)
            run_performance "${SCENARIO}"
            ;;
        run-accuracy)
            run_accuracy "${SCENARIO}"
            ;;
        run-compliance)
            # Ensure SCENARIO is set correctly (should be set via --scenario or default to Server)
            # SCENARIO should already be set by --scenario option or default initialization
            # Verify it's a valid value
            if [[ "$SCENARIO" != "Server" ]] && [[ "$SCENARIO" != "Offline" ]]; then
                echo "ERROR: Invalid scenario: $SCENARIO. Must be 'Server' or 'Offline'"
                exit 1
            fi
            run_compliance "${SCENARIO}" "${COMPLIANCE_TEST}"
            ;;
        run-all)
            echo "=========================================="
            echo "Running all tests for both Server and Offline scenarios"
            echo "=========================================="
            echo ""
            
            echo ">>> Running all Server tests..."
            run_all_tests "Server" || return 1
            echo ""
            
            echo ">>> Running all Offline tests..."
            run_all_tests "Offline" || return 1
            echo ""
            
            echo "✓ All tests for both scenarios completed successfully"
            ;;
        "")
            # No command specified, default to run-server
            run_all_tests "Server"
            ;;
        *)
            echo "Unknown command: $command"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
