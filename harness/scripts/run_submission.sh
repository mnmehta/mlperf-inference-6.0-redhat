#!/bin/bash
# ============================================================================
# run_submission.sh
# -----------------
# Simple bash script to run MLPerf harness tests
# ============================================================================

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ============================================================================
# Configuration Variables (can be overridden via command line or environment)
# ============================================================================

# Dataset configuration
DATASET_DIR="${DATASET_DIR:-/mnt/data/datasets/gpt-oss_data}"
PERF_DATASET="${PERF_DATASET:-${DATASET_DIR}/perf/perf_eval_ref.parquet}"
ACC_DATASET="${ACC_DATASET:-${DATASET_DIR}/acc/acc_eval_ref.parquet}"
COMPLIANCE_DATASET="${COMPLIANCE_DATASET:-${DATASET_DIR}/acc/acc_eval_compliance_gpqa.parquet}"

# Output and API configuration
OUTPUT_DIR="${OUTPUT_DIR:-./harness_output}"
API_SERVER_URL="${API_SERVER_URL:-http://localhost:9999}"

# AWS configuration
AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-}"
AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-}"

# MLflow configuration
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-}"
MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-GPT-OSS-120B-Experiments}"

# HuggingFace configuration
HF_HOME="${HF_HOME:-}"

# Model configuration
MODEL_CATEGORY="${MODEL_CATEGORY:-gpt-oss-120b}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
BACKEND="${BACKEND:-vllm}"
LG_MODEL_NAME="${LG_MODEL_NAME:-gpt-oss-120b}"

# Server configuration
SCENARIO="${SCENARIO:-Server}"
SERVER_TARGET_QPS="${SERVER_TARGET_QPS:-3}"

# Compliance test configuration
COMPLIANCE_TEST="${COMPLIANCE_TEST:-TEST07}"
AUDIT_CONFIG_SRC="${AUDIT_CONFIG_SRC:-}"

# Execution options
DRY_RUN="${DRY_RUN:-false}"
USER_CONF="${USER_CONF:-}"

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
    
    # Check if API server URL is set
    if [[ -z "${API_SERVER_URL}" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "[DRY RUN] NOTE: API_SERVER_URL would be required: ${API_SERVER_URL:-<not set>}"
        else
            echo "ERROR: API_SERVER_URL is not set"
            errors=$((errors + 1))
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
    
    # Build base command
    local cmd=(
        python3 "${HARNESS_DIR}/harness_main.py"
        --model-category "${MODEL_CATEGORY}"
        --model "${MODEL}"
        --dataset-path "${dataset_path}"
        --backend "${BACKEND}"
        --lg-model-name "${LG_MODEL_NAME}"
        --test-mode "${test_mode}"
        --api-server-url "${API_SERVER_URL}"
        --scenario "${scenario}"
        --output-dir "${full_output_dir}"
        --mlflow-experiment-name "${MLFLOW_EXPERIMENT_NAME}"
    )
    
    # Add MLflow tracking URI if provided
    if [[ -n "$MLFLOW_TRACKING_URI" ]]; then
        # Extract host and port from URI
        local mlflow_host_port="${MLFLOW_TRACKING_URI#http://}"
        mlflow_host_port="${mlflow_host_port#https://}"
        local mlflow_host="${mlflow_host_port%%:*}"
        local mlflow_port="${mlflow_host_port##*:}"
        cmd+=(--mlflow-host "${mlflow_host}")
        if [[ "$mlflow_port" != "$mlflow_host" ]]; then
            cmd+=(--mlflow-port "${mlflow_port}")
        fi
    fi
    
    # Add server-target-qps for Server scenario
    if [[ "$scenario" == "Server" ]]; then
        cmd+=(--server-target-qps "${SERVER_TARGET_QPS}")
    fi
    
    # Add MLflow description if provided
    if [[ -n "$description" ]]; then
        cmd+=(--mlflow-description "$description")
    fi
    
    # Add MLflow tags if provided
    if [[ -n "$tags" ]]; then
        cmd+=(--mlflow-tag "$tags")
    fi
    
    # Add user-conf for performance and accuracy tests (if specified)
    if [[ -n "$USER_CONF" ]] && [[ "$test_mode" == "performance" || "$test_mode" == "accuracy" ]]; then
        if [[ -f "$USER_CONF" ]]; then
            cmd+=(--user-conf "${USER_CONF}")
        else
            echo "WARNING: User config file not found: ${USER_CONF}, skipping --user-conf"
        fi
    fi
    
    # Add audit config if provided (for compliance tests)
    local audit_dest=""
    if [[ -n "$audit_config_path" && -f "$audit_config_path" ]]; then
        audit_dest="${full_output_dir}/audit.config"
        cmd+=(--audit-config "${audit_dest}")
        
        if [[ "$DRY_RUN" != "true" ]]; then
            # Copy audit.config to output directory (only if not dry-run)
            mkdir -p "${full_output_dir}"
            cp "${audit_config_path}" "${audit_dest}"
            
            # Cleanup function
            cleanup_audit() {
                if [[ -f "${audit_dest}" ]]; then
                    rm -f "${audit_dest}"
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
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Command would be:"
    else
        echo "Command:"
    fi
    # Print command in a readable format (properly quoted)
    echo "  ${cmd[*]}"
    echo "=========================================="
    
    # Skip execution if dry run
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Command not executed"
        return 0
    fi
    
    # Run command
    if "${cmd[@]}"; then
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
    local dataset_path="${COMPLIANCE_DATASET}"
    local audit_config_path=$(find_audit_config "${compliance_test}")
    
    if [[ -z "$audit_config_path" ]]; then
        echo "Error: Could not find audit.config for ${compliance_test}"
        return 1
    fi
    
    local description="${scenario} Compliance ${compliance_test}"
    local tags="test_type:compliance,scenario:${scenario},compliance_test:${compliance_test}"
    
    if [[ "$scenario" == "Server" ]]; then
        description="${scenario} Compliance ${compliance_test} QPS${SERVER_TARGET_QPS}"
        tags="${tags},qps:${SERVER_TARGET_QPS}"
    fi
    
    run_harness_base "${scenario}" "compliance" "${dataset_path}" "${compliance_test,,}" "${description}" "${tags}" "${audit_config_path}"
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
    
    # Run compliance
    echo ">>> Running compliance test..."
    run_compliance "${scenario}" || return 1
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
            --hf-home)
                HF_HOME="$2"
                shift 2
                ;;
            --scenario)
                SCENARIO="$2"
                shift 2
                ;;
            --server-target-qps)
                SERVER_TARGET_QPS="$2"
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
  --mlflow-tag TAGS               MLflow tags in format 'tag1:value1,tag2:value2' (comma-separated)
  --hf-home PATH                  HuggingFace home directory
  --scenario SCENARIO             Scenario: Server or Offline (default: Server)
  --server-target-qps QPS         Target QPS for Server scenario (default: 3)
  --compliance-test TEST          Compliance test name (default: TEST07)
  --audit-config PATH             Path to audit.config file (auto-detected if not provided)
  --dry-run                        Print commands without executing them
  --user-conf PATH                 User config file for performance/accuracy tests (useful for testing)

Commands:
  run-server                      Run all tests for Server scenario
  run-offline                     Run all tests for Offline scenario
  run-performance [SCENARIO]      Run performance test (default: Server)
  run-accuracy [SCENARIO]         Run accuracy test (default: Server)
  run-compliance [SCENARIO]       Run compliance test (default: Server)

Examples:
  # Run all Server tests
  $0 --scenario Server --server-target-qps 3 run-server

  # Run all Offline tests
  $0 --scenario Offline run-offline

  # Run only performance test for Server
  $0 --scenario Server --server-target-qps 3 run-performance

  # Run compliance test with custom dataset
  $0 --compliance-dataset /path/to/compliance.parquet run-compliance

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
            run-server|run-offline|run-performance|run-accuracy|run-compliance)
                command="$1"
                shift
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
    
    # Export environment variables if set
    [[ -n "$AWS_ACCESS_KEY_ID" ]] && export AWS_ACCESS_KEY_ID
    [[ -n "$AWS_SECRET_ACCESS_KEY" ]] && export AWS_SECRET_ACCESS_KEY
    [[ -n "$HF_HOME" ]] && export HF_HOME
    
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
            run_compliance "${SCENARIO}"
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
