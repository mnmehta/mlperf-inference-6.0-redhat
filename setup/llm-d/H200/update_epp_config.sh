#!/bin/bash
#
# Update EPP Configuration Script
#
# This script applies custom EPP configuration to the GAIE (Gateway API Inference Extension)
# deployment and restarts the EPP pod to pick up the new configuration.
#
# Usage: ./update_epp_config.sh [namespace]
#   namespace: Target namespace (default: llm-d-bench)
#

set -e

# Configuration
NAMESPACE="${1:-llm-d-bench}"
RELEASE_POSTFIX="${RELEASE_NAME_POSTFIX:-inference-scheduling}"
LLMD_DIR="${LLMD_DIR:-/tmp/llm-d}"
WORK_DIR="${LLMD_DIR}/guides/inference-scheduling"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EPP_CONFIG_FILE="${SCRIPT_DIR}/epp_config.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Main function
main() {
    log_info "========================================="
    log_info "EPP Configuration Update"
    log_info "========================================="
    log_info "Namespace: $NAMESPACE"
    log_info "Release Postfix: $RELEASE_POSTFIX"
    log_info ""

    # Check if EPP config file exists
    if [ ! -f "$EPP_CONFIG_FILE" ]; then
        log_error "EPP config file not found: $EPP_CONFIG_FILE"
        exit 1
    fi

    # Check if llm-d directory exists
    if [ ! -d "$WORK_DIR" ]; then
        log_error "LLM-D work directory not found: $WORK_DIR"
        log_info "Please ensure llm-d repository is cloned at: $LLMD_DIR"
        exit 1
    fi

    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
        log_error "Namespace $NAMESPACE does not exist"
        exit 1
    fi

    # Copy EPP config to work directory
    log_info "Copying EPP config to work directory..."
    cp "$EPP_CONFIG_FILE" "$WORK_DIR/"

    # Change to work directory
    cd "$WORK_DIR"

    # Export release name postfix for helmfile
    export RELEASE_NAME_POSTFIX="$RELEASE_POSTFIX"

    # Get current EPP pod name before update
    log_info "Finding current EPP pod..."
    EPP_POD=$(kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=gaie-${RELEASE_POSTFIX}-epp" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

    if [ -z "$EPP_POD" ]; then
        log_warn "No EPP pod found. Proceeding with helm update anyway."
    else
        log_info "Current EPP pod: $EPP_POD"
    fi

    # Apply the updated GAIE configuration
    log_info "Updating GAIE deployment with custom EPP config..."
    EPP_CONFIG_BASENAME="$(basename "$EPP_CONFIG_FILE")"

    helmfile -f helmfile.yaml.gotmpl -l name=gaie-${RELEASE_POSTFIX} apply -n "$NAMESPACE" \
        --args "--values ${EPP_CONFIG_BASENAME}"

    log_info "Helm update complete"
    echo ""

    # Delete the EPP pod to force recreation with new config
    if [ -n "$EPP_POD" ]; then
        log_info "Deleting EPP pod to apply new configuration..."
        kubectl delete pod "$EPP_POD" -n "$NAMESPACE"

        # Wait for new pod to be ready
        log_info "Waiting for new EPP pod to be ready..."
        sleep 5

        # Wait for pod to be running
        local retries=0
        local max_retries=30
        while [ $retries -lt $max_retries ]; do
            NEW_EPP_POD=$(kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=gaie-${RELEASE_POSTFIX}-epp" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
            POD_STATUS=$(kubectl get pod "$NEW_EPP_POD" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo "")

            if [ "$POD_STATUS" = "Running" ]; then
                log_info "New EPP pod is running: $NEW_EPP_POD"
                break
            fi

            sleep 2
            retries=$((retries + 1))
        done

        if [ $retries -eq $max_retries ]; then
            log_warn "Timeout waiting for new EPP pod to be ready"
        fi
    else
        log_info "Waiting for EPP pod to be created..."
        sleep 10

        NEW_EPP_POD=$(kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=gaie-${RELEASE_POSTFIX}-epp" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
        if [ -n "$NEW_EPP_POD" ]; then
            log_info "EPP pod created: $NEW_EPP_POD"
        fi
    fi

    echo ""
    log_info "========================================="
    log_info "Update Complete!"
    log_info "========================================="
    echo ""

    # Show current EPP pod status
    log_info "Current EPP pod status:"
    kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=gaie-${RELEASE_POSTFIX}-epp"
    echo ""

    # Show EPP configuration
    log_info "EPP configuration applied:"
    echo "  - Log verbosity: 7"
    echo "  - Plugins config file: mlperf-epp-config.yaml"
    echo "  - Plugins:"
    echo "    * kv-cache-utilization-scorer (weight: 2.0)"
    echo "    * queue-scorer (weight: 2.0)"
    echo "    * max-score-picker"
    echo ""

    log_info "To check EPP logs:"
    echo "  kubectl logs -n $NAMESPACE -l app.kubernetes.io/name=gaie-${RELEASE_POSTFIX}-epp --tail=50 -f"
    echo ""
}

# Run main function
main
